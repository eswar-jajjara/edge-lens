"""Exported classifiers and a calibration-selected ONNX conversion strategy."""
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
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


def disjoint_inputs(left, right, left_role="Calibration", right_role="test"):
    """Catch duplicated decoded/preprocessed images even when archive names differ."""
    digests = {hashlib.sha256(x.tobytes()).digest() for x in left}
    if any(hashlib.sha256(x.tobytes()).digest() in digests for x in right):
        raise ValueError(f"{left_role} and {right_role} datasets overlap after preprocessing. Supply separate images.")


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


def evaluate_profile(profile, arrays, reference_outputs, labels, dataset, settings, comparison_available=True):
    """Shared labelled evaluation for existing conversions and precision candidates."""
    import numpy as np
    metric = b._evaluate(profile["infer"], arrays, reference_outputs, labels, settings, np)
    outputs = metric.pop("outputs")
    predictions = []
    for i, (out, ref, label) in enumerate(zip(outputs, reference_outputs, labels)):
        diff = np.abs(out.astype(np.float64) - ref.astype(np.float64))
        predictions.append({"profile": profile["profile"], "sample_index": i, "image": dataset["entries"][i]["path"], "label": label,
                            "prediction": int(out.argmax()), "reference_prediction": int(ref.argmax()) if comparison_available else None,
                            "mae": float(diff.mean()) if comparison_available else None,
                            "max_abs": float(diff.max()) if comparison_available else None,
                            "within_tolerance": bool(np.allclose(out, ref, atol=settings["atol"], rtol=settings["rtol"])) if comparison_available else None})
    if not comparison_available:
        for key in ("accuracy_delta_pp", "agreement_pct", "output_mae", "output_max_abs", "within_tolerance", "tolerance_failure_count"):
            metric[key] = None
    metric.update(profile=profile["profile"], label=profile["label"], conversion_seconds=profile["conversion_seconds"],
                  size_bytes=profile["path"].stat().st_size,
                  artifact_sha256=b._hash_file(profile["path"]), dataset_sha256=dataset["sha256"], evaluation_split="test")
    return metric, predictions


def _versions():
    versions = {}
    for name in ("torch", "onnx", "onnxruntime", "numpy", "Pillow", "ai-edge-litert", "litert-torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _split_metadata(dataset, arrays, role):
    digest = hashlib.sha256()
    for array, entry in zip(arrays, dataset["entries"]):
        digest.update(hashlib.sha256(array.tobytes()).digest())
        digest.update(str(entry["label"]).encode("ascii") + b"\0")
    return {**{k: dataset[k] for k in ("id", "name", "sha256", "image_count", "class_count", "labels")},
            "role": role, "preprocessed_sha256": digest.hexdigest()}


def _provenance(metric, split):
    """Each reported metric declares both evidence origin and interpretation."""
    accuracy = {"accuracy_pct", "accuracy_delta_pp", "agreement_pct", "correct_count", "sample_count", "output_mae", "output_max_abs", "tolerance_failure_count", "within_tolerance", "accuracy_delta_vs_fp32_pp"}
    timing = {"latency_mean_ms", "latency_p50_ms", "latency_p95_ms", "latency_samples_ms"}
    keys = accuracy | timing | {"size_bytes", "conversion_seconds", "peak_ram_bytes", "device_latency_ms"}
    metric["provenance"] = {}
    for key in sorted(keys):
        value = metric.get(key)
        origin = {"status": "UNAVAILABLE" if value is None else "MEASURED", "scope": "host_cpu"}
        if key in accuracy:
            origin.update(split=split, sample_count=metric.get("sample_count"), method="Labelled top-1 or raw output comparison on identical preprocessed inputs")
        elif key in timing:
            origin.update(method="Invocation and output copy on one fixed preprocessed image; warm-ups excluded")
        elif key == "size_bytes":
            origin.update(method="Serialized artifact bytes on disk; not runtime RAM or total firmware flash")
        elif key == "conversion_seconds":
            origin.update(method="FP32 export plus candidate transformation where applicable; dependency export time retained separately")
        elif key == "device_latency_ms":
            origin.update(scope="physical_device", reason="No physical-device run in this experiment")
        else:
            origin.update(reason="Peak process RAM is not measured")
        metric["provenance"][key] = origin
    return metric


def _run_quantization_comparison(request, dataset, output_dir):
    """Precision comparisons and deployment search share one evaluator and timer."""
    import numpy as np
    from app.services import quantization as q
    optimizing = request.get("strategy") == "deployment_search"
    optimization = None
    source, settings = request["_model"], b._settings(request)
    if source["format"] not in {"pt2", "onnx"} or request["format"] != "onnx":
        raise ValueError("FP32/INT8 experiments require an uploaded PT2 or FP32 ONNX classifier")
    calibration, validation = request.get("_calibration"), request.get("_validation")
    if not calibration or not validation:
        raise ValueError("Supply separate calibration, validation and test datasets")
    split_data = {"calibration": calibration, "validation": validation, "test": dataset}
    if len({x["sha256"] for x in split_data.values()}) != 3:
        raise ValueError("Calibration, validation and test datasets must be separate archives")
    spec, source_path = public_model(source), verify_model(source)
    pixels = np.prod(spec["input_shape"]) * 4
    if sum(x["image_count"] for x in split_data.values()) * pixels > 512 * 1024 * 1024:
        raise ValueError("Combined preprocessed calibration, validation and test inputs exceed 512 MiB")
    split_inputs = {role: load_images(value, spec) for role, value in split_data.items()}
    roles = list(split_inputs)
    for i, role in enumerate(roles):
        for other in roles[i + 1:]:
            disjoint_inputs(split_inputs[role][0], split_inputs[other][0], role.capitalize(), other)
    class_order = {}
    for value in split_data.values():
        for folder, index in value["labels"].items():
            if folder in class_order and class_order[folder] != index:
                raise ValueError("Class folder indices must match across all three datasets")
            class_order[folder] = index
    splits = {role: _split_metadata(value, split_inputs[role][0], role) for role, value in split_data.items()}
    config = request.get("quantization", {"calibration_method": "MinMax", "per_channel": True})
    candidates, profiles, artifacts, layers = [], [], [], []
    baseline_path, int8_path = output_dir / "fp32.onnx", output_dir / "static-int8.onnx"
    fixed_config = {"runtime": "onnxruntime", "execution_provider": "CPUExecutionProvider", "graph_optimization": "ORT_ENABLE_ALL",
                    "threads": settings["threads"], "inter_op_threads": 1, "execution_mode": "ORT_SEQUENTIAL"}
    baseline = {"id": "fp32", "name": "FP32 ONNX baseline", "candidate_type": "baseline", "precision": "FP32", "format": "onnx",
                "status": "pending", "failure": None, "artifact_sha256": None, "configuration": {**fixed_config, "export_optimize": True if source["format"] == "pt2" else None},
                "validation_metrics": None, "test_metrics": None}
    quantized = {"id": "static_int8", "name": "Static INT8 ONNX (QDQ)", "candidate_type": "static_quantization", "precision": "INT8 QDQ with floating-point operations",
                 "format": "onnx", "status": "pending", "failure": None, "artifact_sha256": None,
                 "configuration": {**fixed_config, "quant_format": "QDQ", "activation_type": "QInt8", "weight_type": "QInt8", "op_types": q.OP_TYPES,
                                   "reduce_range": False, "shape_inference": True, "preprocess_graph_optimization": False, "WeightSymmetric": True, **config},
                 "validation_metrics": None, "test_metrics": None}
    candidates.extend([baseline, quantized])
    started, torch, old_threads = time.perf_counter(), None, None

    def fail(candidate, stage, exc):
        message = str(exc).replace(str(output_dir), "[experiment]").replace(str(source_path), "[source model]")
        candidate.update(status="failed", failure={"stage": stage, "type": type(exc).__name__, "message": message[:1000]})

    def register(candidate, path, conversion_seconds):
        infer, graph = onnx_session(path, settings["threads"])
        q.require_fp32(graph) if candidate is baseline else None
        infer = checked(infer, spec)
        infer(split_inputs["calibration"][0][0])
        artifact = b._artifact(candidate["id"], "onnx", path)
        artifacts.append(artifact)
        candidate.update(status="built", artifact_sha256=artifact["sha256"], size_bytes=artifact["size_bytes"],
                         conversion_seconds=conversion_seconds, inventory=q.quantization_inventory(graph))
        profiles.append({"profile": candidate["id"], "label": candidate["name"], "path": path, "infer": infer,
                         "conversion_seconds": conversion_seconds, "candidate": candidate})
        for i, node in enumerate(graph.graph.node):
            layers.append({"profile": candidate["id"], "name": node.name or f"node_{i}", "operation": node.op_type,
                           "status": "unmapped", "mae": None, "max_abs": None,
                           "detail": "Operator inventory only; numerical correspondence was not established for this entry. See measured diagnostic entries and controlled sensitivity evidence where available."})

    try:
        if source["format"] == "pt2":
            import torch
            old_threads = torch.get_num_threads()
            torch.set_num_threads(settings["threads"])
            ep = torch.export.load(str(source_path))
            module = ep.module()
            def reference(value):
                with torch.no_grad():
                    return _single(module(torch.from_numpy(value)))
            reference = checked(reference, spec)
            copied = output_dir / "reference.pt2"
            shutil.copyfile(source_path, copied)
            artifacts.append(b._artifact("pytorch", "pt2", copied))
            profiles.append({"profile": "pytorch", "label": "Original PyTorch reference", "path": copied, "infer": reference,
                             "conversion_seconds": None})
            try:
                baseline["incremental_build_seconds"] = _export(ep, torch.from_numpy(split_inputs["calibration"][0][0]), baseline_path, True)
                register(baseline, baseline_path, baseline["incremental_build_seconds"])
            except Exception as exc:
                fail(baseline, "build_or_runtime_validation", exc)
        else:
            shutil.copyfile(source_path, baseline_path)
            try:
                baseline["incremental_build_seconds"] = None
                register(baseline, baseline_path, None)
            except Exception as exc:
                fail(baseline, "baseline_validation", exc)
        if baseline["status"] == "built":
            try:
                seconds, _ = q.build_static_int8(baseline_path, int8_path, split_inputs["calibration"][0], config)
                quantized["incremental_build_seconds"] = seconds
                quantized["dependency_build_seconds"] = baseline.get("conversion_seconds") or 0
                register(quantized, int8_path, seconds + quantized["dependency_build_seconds"])
                cache = int8_path.with_name("calibration-ranges.json")
                if cache.is_file():
                    artifact = b._artifact("calibration_ranges", "json", cache)
                    artifact["role"] = "calibration_statistics"
                    artifacts.append(artifact)
                    quantized["calibration_cache_sha256"] = artifact["sha256"]
            except Exception as exc:
                fail(quantized, "quantization_or_runtime_validation", exc)
        else:
            fail(quantized, "dependency", ValueError("FP32 baseline failed; quantization was not attempted"))

        if optimizing:
            from app.services.deployment_search import explore
            from app.schemas.runs import SearchSettings, DeploymentConstraints
            val_arrays, val_labels = split_inputs["validation"]
            validation_reference = [profiles[0]["infer"](array) for array in val_arrays] if profiles else []
            def evaluate_validation(profile):
                metric, _ = evaluate_profile(profile, val_arrays, validation_reference, val_labels, validation, settings)
                metric.update(peak_ram_bytes=None, device_latency_ms=None, latency_mean_ms=None, latency_p50_ms=None,
                              latency_p95_ms=None, latency_samples_ms=None, reference_profile=profiles[0]["profile"], evaluation_split="validation")
                return _provenance(metric, "validation")
            optimization = explore(baseline_path=baseline_path, candidates=candidates, profiles=profiles, artifacts=artifacts,
                                   output_dir=output_dir, calibration_arrays=split_inputs["calibration"][0], validation_arrays=val_arrays,
                                   settings=settings, search=SearchSettings.model_validate(request.get("search", {})).model_dump(),
                                   constraints=DeploymentConstraints.model_validate(request.get("constraints", {})).model_dump(),
                                   initial_config=config, register=register, evaluate=evaluate_validation, fail=fail)
            for candidate in candidates:
                value = candidate.get("validation_metrics")
                base_value = baseline.get("validation_metrics")
                if value:
                    value["accuracy_delta_vs_fp32_pp"] = value["accuracy_pct"] - base_value["accuracy_pct"] if base_value else None
                    _provenance(value, "validation")
            lookup = {(row["profile"], row["name"]): row for row in layers}
            for row in optimization["diagnostics"].get("rows", []):
                measured = dict(row, profile=optimization["diagnostics"].get("profile"))
                if (measured["profile"], measured["name"]) in lookup:
                    lookup[(measured["profile"], measured["name"])].update(measured)
                else:
                    layers.append(measured)
            diagnostics_file = output_dir / "layer-diagnostics.json"
            diagnostics_file.write_text(json.dumps({"diagnostics": optimization["diagnostics"], "sensitivity": optimization["sensitivity"]}, indent=2), encoding="utf-8")
            artifact = b._artifact("layer_diagnostics", "json", diagnostics_file)
            artifact["role"] = "diagnostic_report"
            artifacts.append(artifact)

        # Selection is frozen before any held-out inference. Phase 1 stays fixed.
        metrics, predictions = [], []
        reference_profile = profiles[0] if profiles else None
        final_ids = {"pytorch", "fp32", "static_int8", optimization["selection"]["selected"]} if optimizing else None
        valid_profiles = [profile for profile in profiles if not optimizing or (profile["profile"] in final_ids and profile.get("candidate", {}).get("status") != "failed")]
        for role in (("test",) if optimizing else ("validation", "test")):
            arrays, labels = split_inputs[role]
            reference_outputs = [reference_profile["infer"](x) for x in arrays] if reference_profile else []
            for profile in list(valid_profiles):
                candidate = profile.get("candidate")
                try:
                    metric, per_image = evaluate_profile(profile, arrays, reference_outputs, labels, split_data[role], settings)
                    metric.update(peak_ram_bytes=None, device_latency_ms=None,
                                  latency_mean_ms=None, latency_p50_ms=None, latency_p95_ms=None, latency_samples_ms=None,
                                  reference_profile=reference_profile["profile"], evaluation_split=role)
                    _provenance(metric, role)
                    if candidate:
                        candidate[role + "_metrics"] = metric
                    if role == "test":
                        metrics.append(metric)
                        predictions.extend(per_image)
                except Exception as exc:
                    if not candidate:
                        raise
                    fail(candidate, role + "_evaluation", exc)
                    valid_profiles.remove(profile)
            baseline_metrics = baseline.get(role + "_metrics")
            if baseline_metrics:
                for candidate in candidates:
                    value = candidate.get(role + "_metrics")
                    if value:
                        value["accuracy_delta_vs_fp32_pp"] = value["accuracy_pct"] - baseline_metrics["accuracy_pct"]
                        _provenance(value, role)
        if valid_profiles:
            timing_failures = {}
            timings = b._time_profiles([(p["profile"], p["infer"], split_inputs["test"][0][0]) for p in valid_profiles], settings, timing_failures)
            for metric in metrics:
                metric.update(timings.get(metric["profile"], {}))
                if metric["profile"] in timing_failures:
                    exc = timing_failures[metric["profile"]]
                    metric["benchmark_failure"] = str(exc)[:1000]
                    candidate = next((c for c in candidates if c["id"] == metric["profile"]), None)
                    if candidate:
                        fail(candidate, "host_benchmark", exc)
                _provenance(metric, "test")
        for candidate in candidates:
            if candidate["status"] == "built":
                candidate["status"] = "completed"
        fp32 = next((m for m in metrics if m["profile"] == "fp32"), None)
        int8 = next((m for m in metrics if m["profile"] == "static_int8"), None)
        if fp32:
            for metric in metrics:
                metric["accuracy_delta_vs_fp32_pp"] = metric["accuracy_pct"] - fp32["accuracy_pct"]
                _provenance(metric, "test")
        if fp32 and int8:
            delta = int8["accuracy_pct"] - fp32["accuracy_pct"]
            summary = {"conclusion": f"Static INT8 versus FP32: {delta:+.6g} percentage points on the held-out test set. Inspect size and host latency before choosing a deployment configuration; no automatic winner is selected.",
                       "accuracy_delta_pp": delta, "latency_delta_ms": int8["latency_p50_ms"] - fp32["latency_p50_ms"] if int8["latency_p50_ms"] is not None and fp32["latency_p50_ms"] is not None else None,
                       "size_delta_bytes": int8["size_bytes"] - fp32["size_bytes"]}
        else:
            summary = {"conclusion": "FP32/INT8 comparison is incomplete. Failed candidates and their reasons are retained; no deployment recommendation is established."}
        for metric in metrics:
            row = next((x for x in predictions if x["profile"] == metric["profile"]), None)
            if row:
                layers.append({"profile": metric["profile"], "name": "output.logits", "operation": "ClassifierOutput", "sample_index": 0,
                               "status": "pass" if row["within_tolerance"] else "drift", "mae": row["mae"], "max_abs": row["max_abs"],
                               "detail": f"Final output on the first test image compared to {metric['reference_profile']}. This is not a layer root-cause diagnosis."})
        report = {"schema_version": 3, "source": "measured", "experiment_type": "fp32_static_int8", "model": spec,
                "created_at": datetime.now(timezone.utc).isoformat(), "dataset": splits["test"], "datasets": splits,
                "summary": summary, "selection": None, "test_data_used_for_selection": False,
                "search": {"candidate_limit": 2, "attempted_candidates": 2, "automatic_selection": False, "elapsed_seconds": time.perf_counter() - started},
                "accuracy_resolution_pp": 100 / dataset["image_count"], "candidates": candidates,
                "environment": {"benchmark_scope": "host_cpu", "platform": platform.platform(), "architecture": platform.machine(),
                                "processor": platform.processor() or "not reported", "logical_cpus": os.cpu_count(),
                                "python": platform.python_version(), "versions": _versions(), "settings": settings, "provider": "CPUExecutionProvider"},
                "target": b._target(request["target"], "onnx"), "metrics": metrics, "layers": layers, "predictions": predictions,
                "artifacts": artifacts, "edge_results": [],
                "methodology": ["Fixed FP32 and static S8S8 QDQ configurations; standard PyTorch ONNX exporter optimize=True when source is PT2. No additional FP32 graph rewriting.",
                                "Only calibration inputs determine INT8 ranges. Shape inference precedes quantization; graph optimization during preprocessing is disabled.",
                                "Calibration, validation and held-out test archives and exact preprocessed images must be disjoint. Identical declared preprocessing and class order are used.",
                                "Configurations are fixed before validation and test evaluation. Validation metrics are recorded; no candidate is selected or tuned using test results.",
                                "Accuracy and output metrics cover every image of the corresponding split. Numerical reference is PyTorch for PT2 uploads or the uploaded FP32 ONNX baseline otherwise.",
                                "Both ONNX sessions use CPUExecutionProvider, ORT_ENABLE_ALL, sequential execution and identical thread counts.",
                                "Latency measures one fixed test image with warm-ups and rotating profile order. Decode, preprocessing, calibration, conversion and diagnostics are excluded; raw samples are saved.",
                                "Conversion seconds include the FP32 dependency export plus each candidate's transformation; incremental and dependency times are retained separately."],
                "limitations": ["A QDQ graph can retain floating-point operations. Reported coverage describes graph boundaries, not a hardware kernel execution trace or fully integer firmware.",
                                "Quantized layer mapping, sensitivity search, mixed precision and automatic constraint selection are future milestones.",
                                "Imported ONNX cannot establish original PyTorch conversion loss; it can establish quantization differences against its FP32 baseline.",
                                f"One changed prediction = {100 / dataset['image_count']:.6g} pp. Small datasets are smoke tests, not reliable model-quality evidence." if dataset["image_count"] < 300 else f"One changed prediction = {100 / dataset['image_count']:.6g} pp; this is sample accuracy, without a statistical significance claim.",
                                "Host timings are not ESP32 or Raspberry Pi timings. No physical-device latency or peak RAM is measured. Serialized bytes are not runtime RAM or total firmware flash.",
                                "Equal, slower, larger and less accurate candidates remain visible. A 0.001 pp accuracy step needs at least 100,000 test images; this bounded prototype does not support that scale."]}
        if optimizing:
            selection = optimization["selection"]
            selected = next((candidate for candidate in candidates if candidate["id"] == selection["selected"]), None)
            if selected and selected.get("test_metrics") and selected["status"] == "completed":
                value = selected["test_metrics"]
                report["summary"] = {"conclusion": f"Selected {selected['name']} using validation {selection['objective']}. Held-out accuracy: {value['accuracy_pct']:.6g}%. Selection was frozen before test evaluation.",
                                     "accuracy_delta_pp": value.get("accuracy_delta_vs_fp32_pp"), "selected": selected["id"]}
            else:
                report["summary"] = {"conclusion": "No final deployment recommendation: " + ("trade-offs only; no optimization objective requested." if selection["decision"] == "tradeoffs_only" else "no measured candidate satisfies the constraints, or the selected candidate's final test failed.")}
            report.update(schema_version=4, experiment_type="deployment_optimization", selection=selection,
                          search={**optimization["search"], "total_strategy_seconds": time.perf_counter() - started},
                          sensitivity=optimization["sensitivity"], diagnostics=optimization["diagnostics"])
            report["methodology"] = ["Build bounded static INT8 calibration/granularity candidates, then exclude sensitivity-ranked operations from quantization; reuse the same FP32 export, quantizer and evaluator.",
                                     "Calibration alone determines ranges and activation diagnostics. Validation alone determines controlled recovery, constraints, Pareto membership and selection.",
                                     "One-operation probes hold other quantizer settings and calibration ranges fixed. Cumulative exclusions use operations with observed validation improvement in accuracy or, on ties, output MAE.",
                                     "Selection is frozen before held-out inference. PyTorch, FP32, the initial INT8 control and the selected configuration receive final test results; other candidates retain validation evidence only.",
                                     "Host latency used for selection is measured on the first validation image with rotating candidate order. Final host latency uses the first test image. Diagnostic unoptimized sessions are never timed as deployment artifacts."] + report["methodology"][-3:]
            report["limitations"][1] = "Diagnostics compare preserved ONNX operation outputs on bounded calibration samples, not arbitrary PyTorch layer identity. Propagated activation drift is not causal proof; control effects are specific to this validation subset. Excluded operations can still receive quantized neighbouring activations."
        return report
    finally:
        if torch is not None and old_threads is not None:
            torch.set_num_threads(old_threads)


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
    if request.get("strategy") in {"quantization_compare", "deployment_search"}:
        return _run_quantization_comparison(request, dataset, output_dir)
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
            metric, per_image = evaluate_profile(p, arrays, reference_outputs, labels, dataset, settings, ep is not None)
            predictions.extend(per_image)
            metric.update(**timings[p["profile"]])
            _provenance(metric, "test")
            metrics.append(metric)
        summary = comparison_summary(metrics[1], metrics[2]) if ep is not None else {"conclusion": "Measured uploaded classifier only. No original PyTorch reference: conversion loss and converter superiority cannot be established."}
        if ep is not None:
            summary["output_mae_delta"] = metrics[2]["output_mae"] - metrics[1]["output_mae"]
            summary["output_max_abs_delta"] = metrics[2]["output_max_abs"] - metrics[1]["output_max_abs"]
        versions = _versions()
        return {"schema_version": 2, "source": "measured", "created_at": datetime.now(timezone.utc).isoformat(),
                "model": spec, "dataset": _split_metadata(dataset, arrays, "test"),
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
                                f"Accuracy step on this dataset is {100 / len(labels):.6g} percentage points. A 0.001 pp step would require at least 100,000 test images; this prototype accepts up to 1,000, subject to the preprocessing memory budget.",
                                "Small latency differences can be timer/OS noise; inspect raw timing samples and repeat independent runs. No significance claim is made.",
                                "Host CPU timings are not ESP32 or Raspberry Pi timings. Physical results and provider estimates are stored separately.",
                                "No peak process RAM or energy measurement. Stored bytes and tensor arena usage are different quantities."]}
    finally:
        if torch is not None and old_threads is not None:
            torch.set_num_threads(old_threads)
