"""Exported classifiers and a calibration-selected ONNX conversion strategy."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import platform
import shutil
import time
import zipfile
from pathlib import Path

from app.services import benchmark as b
from app.services.models import verify_model, public_model
from app.services.metrics import comparison_summary


def _single(value):
    if isinstance(value, (tuple, list)) and len(value) == 1:
        value = value[0]
    return value


def load_images(dataset, spec):
    import numpy as np
    from PIL import Image
    channels = spec["input_shape"][1 if spec["layout"] == "NCHW" else 3]
    height, width = spec["input_shape"][2:] if spec["layout"] == "NCHW" else spec["input_shape"][1:3]
    if len(dataset["entries"]) * height * width * channels * 4 > 512 * 1024 * 1024:
        raise ValueError("Preprocessed dataset exceeds 512 MiB; reduce image count or input dimensions")
    if b._hash_file(Path(dataset["path"])) != dataset["sha256"]:
        raise ValueError("Dataset checksum changed")
    arrays, labels = [], []
    with zipfile.ZipFile(dataset["path"]) as archive:
        for entry in dataset["entries"]:
            if not 0 <= entry["label"] < spec["class_count"]:
                raise ValueError(f"Label {entry['label']} is outside model class order 0..{spec['class_count'] - 1}")
            with Image.open(io.BytesIO(archive.read(entry["path"]))) as image:
                if image.width * image.height > b.MAX_IMAGE_PIXELS:
                    raise ValueError("Image exceeds the pixel limit")
                image = image.convert("RGB" if channels == 3 else "L")
                if spec["resize"] == "shortest_center_crop":
                    scale = spec["resize_shorter"] / min(image.size)
                    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
                    left, top = (image.width - width) // 2, (image.height - height) // 2
                    image = image.crop((left, top, left + width, top + height))
                else:
                    image = image.resize((width, height), Image.Resampling.BILINEAR)
                value = np.asarray(image, dtype=np.float32).reshape(height, width, channels)
                value = (value * spec["scale"] - np.asarray(spec["mean"], dtype=np.float32)) / np.asarray(spec["std"], dtype=np.float32)
                if spec["layout"] == "NCHW":
                    value = value.transpose(2, 0, 1)
                arrays.append(np.ascontiguousarray(value[None], dtype=np.float32))
                labels.append(entry["label"])
    return arrays, labels


def disjoint_inputs(left, right):
    """Catch duplicated decoded/preprocessed images even when archive names differ."""
    digests = {hashlib.sha256(x.tobytes()).digest() for x in left}
    if any(hashlib.sha256(x.tobytes()).digest() in digests for x in right):
        raise ValueError("Calibration and test datasets overlap after preprocessing. Supply separate images.")


def onnx_session(path, threads, optimize=True):
    import onnx
    import onnxruntime as ort
    graph = onnx.load(str(path), load_external_data=False)
    def inspect_graph(value):
        for tensor in list(value.initializer) + [s.values for s in value.sparse_initializer] + [s.indices for s in value.sparse_initializer]:
            if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
                raise ValueError("Upload a single-file ONNX model with embedded weights; external data is unsupported")
        for node in value.node:
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    inspect_graph(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for child in attribute.graphs:
                        inspect_graph(child)
                elif attribute.type in (onnx.AttributeProto.TENSOR, onnx.AttributeProto.TENSORS):
                    for tensor in ([attribute.t] if attribute.type == onnx.AttributeProto.TENSOR else attribute.tensors):
                        if tensor.external_data or tensor.data_location == onnx.TensorProto.EXTERNAL:
                            raise ValueError("External tensor data is unsupported")
    inspect_graph(graph.graph)
    from types import SimpleNamespace
    for function in graph.functions:
        inspect_graph(SimpleNamespace(initializer=[], sparse_initializer=[], node=function.node))
    session = b._ort_session(ort, graph.SerializeToString(), threads, diagnostic=not optimize)
    if len(session.get_inputs()) != 1 or len(session.get_outputs()) != 1 or session.get_inputs()[0].type != "tensor(float)":
        raise ValueError("Supported ONNX classifiers need one float32 image input and one class-score output")
    name = session.get_inputs()[0].name
    return lambda value: session.run(None, {name: value})[0], graph


def tflite_session(path, threads):
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(path), num_threads=threads)
    interpreter.allocate_tensors()
    ins, outs = interpreter.get_input_details(), interpreter.get_output_details()
    if len(ins) != 1 or len(outs) != 1:
        raise ValueError("Expected one image input and one class-score output")
    if any(x["dtype"] not in (np.float32, np.int8, np.uint8) for x in (ins[0], outs[0])):
        raise ValueError("TFLite input/output dtype must be float32, int8 or uint8")
    def quantize(value, detail):
        if np.issubdtype(detail["dtype"], np.integer):
            scale, zero = detail["quantization"]
            if scale <= 0:
                raise ValueError("A quantized input needs per-tensor scale and zero point")
            limits = np.iinfo(detail["dtype"])
            return np.clip(np.rint(value / scale + zero), limits.min, limits.max).astype(detail["dtype"])
        return value.astype(detail["dtype"])
    def infer(value):
        interpreter.set_tensor(ins[0]["index"], quantize(value, ins[0]))
        interpreter.invoke()
        result = interpreter.get_tensor(outs[0]["index"])
        if np.issubdtype(result.dtype, np.integer):
            scale, zero = outs[0]["quantization"]
            if scale <= 0:
                raise ValueError("A quantized output needs per-tensor scale and zero point")
            result = (result.astype(np.float32) - zero) * scale
        return result
    return infer, interpreter, lambda value: quantize(value, ins[0])


def checked(infer, spec):
    def run(value):
        import numpy as np
        result = b._numpy_output(_single(infer(value)), np)
        if result.shape != (1, spec["class_count"]):
            raise ValueError(f"Expected class scores [1, {spec['class_count']}], got {list(result.shape)}")
        return result
    return run


def rank_fidelity(metric):
    return (metric["tolerance_failure_count"], metric["output_max_abs"], metric["output_mae"])


def _export(ep, sample, path, optimize):
    import torch
    started = time.perf_counter()
    program = torch.onnx.export(copy.deepcopy(ep), (sample,), dynamo=True, optimize=optimize,
                                input_names=["image"], output_names=["logits"], external_data=False, verbose=False)
    program.save(str(path), external_data=False)
    return time.perf_counter() - started


def _diagnostics(ep, first, profiles, spec, settings):
    """Each exported PyTorch operation is listed; only proven boundaries get errors."""
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    module = ep.module()
    captures, budget = {}, [0]
    class Capture(torch.fx.Interpreter):
        def run_node(self, node):
            value = super().run_node(node)
            if node.op == "call_function" and isinstance(value, torch.Tensor):
                size = value.numel() * value.element_size()
                if budget[0] + size <= 64 * 1024 * 1024:
                    captures[node.name] = value.detach().clone().cpu().numpy()
                    budget[0] += size
            return value
    reference_output = _single(Capture(module).run(torch.from_numpy(first)))
    reference_output = b._numpy_output(reference_output, np)
    nodes = [n for n in module.graph.nodes if n.op == "call_function"]
    rows = []
    allowed = {"convolution": {"Conv"}, "conv2d": {"Conv"}, "relu": {"Relu"}, "linear": {"Gemm"}, "add": {"Add"}, "mul": {"Mul"}, "sigmoid": {"Sigmoid"}}
    for profile in profiles:
        values, names = {}, {}
        if not profile.get("export_optimize", True):
            try:
                graph = onnx.shape_inference.infer_shapes(onnx.load(str(profile["path"])))
                infos = {v.name: v for v in list(graph.graph.value_info) + list(graph.graph.output)}
                for node in nodes:
                    matching = []
                    for target in graph.graph.node:
                        props = {x.key: x.value for x in target.metadata_props}
                        origin = props.get("pkg.torch.onnx.fx_node", "")
                        # Match the exact FX definition, not a substring of another node's name.
                        if origin.startswith(f"%{node.name} :"):
                            matching.append(target)
                    op = str(node.target).split(".")[-2] if "." in str(node.target) else ""
                    if len(matching) == 1 and matching[0].domain in ("", "ai.onnx") and matching[0].op_type in allowed.get(op, set()):
                        target = matching[0]
                        if len(target.output) == 1 and target.output[0] in infos and node.name in captures:
                            names[node.name] = target.output[0]
                del graph.graph.output[:]
                # Tensor capture and extra graph outputs are bounded independently.
                for name in dict.fromkeys(names.values()):
                    graph.graph.output.append(infos[name])
                if names:
                    session = b._ort_session(ort, graph.SerializeToString(), settings["threads"], True)
                    keys = list(dict.fromkeys(names.values()))
                    values = dict(zip(keys, session.run(keys, {session.get_inputs()[0].name: first})))
            except Exception:
                names, values = {}, {}
        candidates = [(n.name, str(n.target), captures.get(n.name), values.get(names.get(n.name))) for n in nodes]
        candidates.append(("output.logits", "ClassifierOutput", reference_output, profile["infer"](first)))
        for name, op, ref, candidate in candidates:
            row = {"profile": profile["profile"], "name": name, "operation": op, "status": "unmapped", "mae": None, "max_abs": None,
                   "detail": "No proven one-to-one tensor mapping; fusion/decomposition or capture limit may apply.", "sample_index": 0}
            if ref is not None and candidate is not None and ref.shape == candidate.shape:
                diff = np.abs(candidate.astype(np.float64) - ref.astype(np.float64))
                finite = np.isfinite(diff).all()
                row.update(status=("pass" if np.allclose(candidate, ref, atol=settings["atol"], rtol=settings["rtol"]) else "drift") if finite else "nonfinite",
                           mae=float(diff.mean()) if finite else None, max_abs=float(diff.max()) if finite else None,
                           expected_shape=list(ref.shape), actual_shape=list(candidate.shape),
                           detail="Final output in measured runtime." if name == "output.logits" else "Exact FX node and single compatible operator; diagnostic runtime disables graph optimization. First test image only.")
                if finite:
                    index = np.unravel_index(np.argmax(diff), diff.shape)
                    row.update(max_error_index=[int(x) for x in index], expected_value=float(ref[index]), actual_value=float(candidate[index]))
            rows.append(row)
    return rows


def run_developer_benchmark(request, dataset, output_dir):
    import numpy as np
    settings = b._settings(request)
    source = request["_model"]
    spec = public_model(source)
    path = verify_model(source)
    arrays, labels = load_images(dataset, spec)
    strategy = request.get("strategy", "fixed_profiles")
    calibration = request.get("_calibration")
    selection, layers, profiles, artifacts = None, [], [], []
    torch, old_threads, ep = None, None, None
    try:
        source_path = output_dir / ("reference." + source["format"])
        shutil.copyfile(path, source_path)
        artifacts.append(b._artifact("reference", source["format"], source_path))
        if source["format"] == "pt2":
            import torch
            old_threads = torch.get_num_threads()
            torch.set_num_threads(settings["threads"])
            ep = torch.export.load(str(path))
            module = ep.module()
            def reference_infer(value):
                with torch.no_grad():
                    return _single(module(torch.from_numpy(value)))
            reference_infer = checked(reference_infer, spec)
            profiles.append({"profile": "pytorch", "label": "Uploaded PyTorch reference", "infer": reference_infer, "conversion_seconds": None, "path": source_path})
            sample = torch.from_numpy(arrays[0])
            strategy_started = time.perf_counter()
            if request["format"] == "onnx":
                standard_path, conservative_path = output_dir / "standard.onnx", output_dir / "unoptimized.onnx"
                standard_time = _export(ep, sample, standard_path, True)
                unoptimized_time = _export(ep, sample, conservative_path, False)
                standard_infer, _ = onnx_session(standard_path, settings["threads"])
                standard = {"profile": "standard", "label": "Standard PyTorch ONNX export", "infer": checked(standard_infer, spec),
                            "conversion_seconds": standard_time, "path": standard_path, "export_optimize": True, "runtime_optimize": True}
                profiles.append(standard)
                candidates = [dict(standard, id="standard", export_seconds=standard_time)]
                for runtime_opt in (True, False):
                    infer, _ = onnx_session(conservative_path, settings["threads"], runtime_opt)
                    candidates.append({"id": "preserve_graph" if runtime_opt else "preserve_graph_no_fusion", "path": conservative_path,
                                       "infer": checked(infer, spec), "export_optimize": False, "runtime_optimize": runtime_opt,
                                       "conversion_seconds": unoptimized_time, "export_seconds": unoptimized_time})
                if strategy == "fidelity_search":
                    if not calibration:
                        raise ValueError("Fidelity search requires a separate calibration dataset")
                    calibration_arrays, calibration_labels = load_images(calibration, spec)
                    disjoint_inputs(calibration_arrays, arrays)
                    reference_outputs = [reference_infer(x) for x in calibration_arrays]
                    evidence = []
                    for candidate in candidates:
                        metric = b._evaluate(candidate["infer"], calibration_arrays, reference_outputs, calibration_labels, settings, np)
                        metric.pop("outputs")
                        candidate["rank"] = rank_fidelity(metric)
                        evidence.append({"id": candidate["id"], "export_optimize": candidate["export_optimize"], "runtime_optimize": candidate["runtime_optimize"],
                                         "export_seconds": candidate["export_seconds"], **metric})
                    winner = min(candidates, key=lambda x: x["rank"])
                    selection = {"algorithm": "EdgeLens calibration-guided fidelity search v1", "selected": winner["id"],
                                 "objective": "Minimize tolerance-failing samples, then maximum absolute error, then MAE; exact ties choose standard.",
                                 "calibration": {k: calibration[k] for k in ("id", "name", "sha256", "image_count")},
                                 "candidates": evidence, "test_data_used_for_selection": False, "total_strategy_seconds": time.perf_counter() - strategy_started}
                    duration = selection["total_strategy_seconds"]
                else:
                    winner, duration = candidates[1], unoptimized_time
                selected_path = output_dir / "dashboard.onnx"
                shutil.copyfile(winner["path"], selected_path)
                profiles.append({**winner, "path": selected_path, "profile": "dashboard", "label": "EdgeLens selected ONNX" if selection else "EdgeLens preserve-graph ONNX",
                                 "conversion_seconds": duration})
            else:
                if strategy == "fidelity_search":
                    raise ValueError("Fidelity search currently supports PT2 → ONNX only")
                import litert_torch
                for profile in ("standard", "dashboard"):
                    target = output_dir / f"{profile}.tflite"
                    started = time.perf_counter()
                    litert_torch.convert(module, (sample,)).export(str(target))
                    seconds = time.perf_counter() - started
                    infer, _, _ = tflite_session(target, settings["threads"])
                    profiles.append({"profile": profile, "label": f"{profile.title()} LiteRT FP32 (identical control)", "path": target,
                                     "infer": checked(infer, spec), "conversion_seconds": seconds, "export_optimize": True})
            artifacts += [{**b._artifact(p["profile"], request["format"], p["path"]),
                           "runtime_optimize": p.get("runtime_optimize"), "export_optimize": p.get("export_optimize")} for p in profiles[1:]]
            if request["format"] == "onnx":
                diagnostic = {**candidates[2], "profile": "diagnostic_unoptimized", "label": "Unoptimized diagnostic graph only"}
                with torch.no_grad():
                    layers = _diagnostics(ep, arrays[0], profiles[1:] + [diagnostic], spec, settings)
                artifacts.append({**b._artifact("diagnostic_unoptimized", "onnx", conservative_path),
                                  "role": "diagnostics_only", "export_optimize": False, "runtime_optimize": False})
            else:
                layers = [{"profile": p["profile"], "name": n.name, "operation": str(n.target), "status": "unmapped", "mae": None, "max_abs": None,
                           "detail": "TFLite intermediate tensors are not aligned to exported PyTorch nodes."} for p in profiles[1:] for n in ep.graph.nodes if n.op == "call_function"]
        else:
            if strategy == "fidelity_search" or request["format"] != source["format"]:
                raise ValueError("Imported ONNX/TFLite models are benchmarked in their existing format. Supply PT2 for conversion comparisons.")
            if source["format"] == "onnx":
                infer, graph = onnx_session(path, settings["threads"])
                layers = [{"profile": "imported", "name": n.name or f"node_{i}", "operation": n.op_type, "status": "unmapped", "mae": None, "max_abs": None,
                           "detail": "Operator inventory only. Original PyTorch tensors were not supplied."} for i, n in enumerate(graph.graph.node)]
            else:
                infer, interpreter, quantize = tflite_session(path, settings["threads"])
                # Runtime tensor inventory uses the public API; it is not a layer equivalence claim.
                layers = [{"profile": "imported", "name": x["name"], "operation": "TFLite tensor", "status": "unmapped", "mae": None, "max_abs": None,
                           "detail": f"Tensor shape {x['shape'].tolist()}; dtype {np.dtype(x['dtype']).name}; quantization {x['quantization']}. Original PyTorch tensors unavailable."} for x in interpreter.get_tensor_details()]
            profiles = [{"profile": "imported", "label": f"Uploaded {source['format'].upper()} classifier", "path": source_path,
                         "infer": checked(infer, spec), "conversion_seconds": None}]
        reference_outputs = [profiles[0]["infer"](x) for x in arrays]
        timings = b._time_profiles([(p["profile"], p["infer"], arrays[0]) for p in profiles], settings)
        metrics, predictions = [], []
        for p in profiles:
            metric = b._evaluate(p["infer"], arrays, reference_outputs, labels, settings, np)
            outputs = metric.pop("outputs")
            for i, (out, ref, label) in enumerate(zip(outputs, reference_outputs, labels)):
                diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
                predictions.append({"profile": p["profile"], "sample_index": i, "image": dataset["entries"][i]["path"], "label": label,
                                    "prediction": int(out.argmax()), "reference_prediction": int(ref.argmax()), "max_abs": float(diff.max()),
                                    "within_tolerance": bool(np.allclose(out, ref, atol=settings["atol"], rtol=settings["rtol"]))})
            if ep is None:
                for key in ("accuracy_delta_pp", "agreement_pct", "output_mae", "output_max_abs", "within_tolerance", "tolerance_failure_count"):
                    metric[key] = None
            metric.update(profile=p["profile"], label=p["label"], conversion_seconds=p["conversion_seconds"], size_bytes=p["path"].stat().st_size, **timings[p["profile"]])
            metrics.append(metric)
        if ep is None:
            for item in predictions:
                item.update(reference_prediction=None, max_abs=None, within_tolerance=None)
        summary = comparison_summary(metrics[1], metrics[2]) if ep is not None else {"conclusion": "Measured uploaded classifier only. No original PyTorch reference: conversion loss and converter superiority cannot be established."}
        if ep is not None:
            summary["output_mae_delta"] = metrics[2]["output_mae"] - metrics[1]["output_mae"]
            summary["output_max_abs_delta"] = metrics[2]["output_max_abs"] - metrics[1]["output_max_abs"]
        versions = {}
        for name in ("torch", "onnx", "onnxruntime", "numpy", "Pillow", "ai-edge-litert", "litert-torch"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        return {"schema_version": 2, "source": "measured", "created_at": datetime.now(timezone.utc).isoformat(),
                "model": spec, "dataset": {k: dataset[k] for k in ("id", "name", "sha256", "image_count", "class_count")},
                "summary": summary, "selection": selection, "accuracy_resolution_pp": 100 / len(labels),
                "environment": {"benchmark_scope": "host_cpu", "platform": platform.platform(), "python": platform.python_version(), "versions": versions, "settings": settings},
                "target": b._target(request["target"], request["format"]), "metrics": metrics, "layers": layers, "predictions": predictions,
                "artifacts": artifacts, "edge_results": [],
                "methodology": ["Same uploaded model, declared preprocessing and labelled test images across all profiles.",
                                "Preprocessing: resize with bilinear interpolation; (pixel * scale - mean) / std; declared RGB/grayscale and NCHW/NHWC layout.",
                                "Calibration-guided selection uses separate images and no held-out labels or outputs; exact preprocessed overlap is rejected." if selection else "No calibration-guided selection was requested.",
                                "Accuracy is labelled top-1; agreement compares with the supplied PyTorch reference when available. Deltas use percentage points.",
                                "Timing uses one fixed preprocessed image, warm-ups and rotating profile order; excludes preprocessing, loading and conversion; includes invocation/output copy.",
                                "EdgeLens search time includes both exports, session creation and calibration inference. Per-export times are retained in selection evidence.",
                                "Layer evidence uses the first test image. An additional unoptimized diagnostic profile provides conservative operator mappings even when the selected artifact is optimized. It is not an extra accuracy/latency competitor. Unmapped operations are unverified, not passed. Final predictions cover every test image."],
                "limitations": ["Custom classifiers currently require one fixed batch-one float32 image input and one finite [1, class_count] score output; imported TFLite also supports per-tensor int8/uint8 quantization.",
                                "ONNX and TFLite uploads alone do not provide evidence of conversion loss. Upload PT2 to compare against PyTorch.",
                                "Results may be equal or worse. This is an orchestration and selection algorithm built on standard exporters, not a new low-level compiler.",
                                f"Accuracy step on this dataset is {100 / len(labels):.6g} percentage points. A 0.001 pp step would require at least 100,000 test images; this prototype accepts up to 200.",
                                "Small latency differences can be timer/OS noise; inspect raw timing samples and repeat independent runs. No significance claim is made.",
                                "Host CPU timings are not ESP32 or Raspberry Pi timings. Physical results and provider estimates are stored separately.",
                                "No peak process RAM or energy measurement. Stored bytes and tensor arena usage are different quantities."]}
    finally:
        if torch is not None and old_threads is not None:
            torch.set_num_threads(old_threads)
