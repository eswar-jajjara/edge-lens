"""Optional real CPU benchmark worker; importing this module needs no ML packages.

This module handles registered torchvision architectures. Developer exports use
the separate, explicitly trusted developer_benchmark worker. The API must serialize jobs: torch
thread settings are process-global. A cloud CPU run is not a physical edge test.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import os
import platform
import time
import warnings
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from app.services.metrics import classification_metrics, comparison_summary, percentile


class RuntimeUnavailable(RuntimeError):
    """Optional ML installation is missing, unsupported, or cannot be imported."""


MODELS = {"mobilenet_v2": "MobileNetV2", "resnet18": "ResNet18"}
MAX_IMAGES = 1000
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


def _present(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def runtime_capabilities() -> dict:
    """Presence check only; execution performs the authoritative import/API check."""
    base = ["torch", "torchvision", "numpy", "PIL"]
    output = {}
    for format_name, dependencies in {
        "onnx": base + ["onnx", "onnxscript", "onnxruntime"],
        "tflite": base + ["litert_torch", "ai_edge_litert"],
    }.items():
        missing = [name for name in dependencies if not _present(name)]
        supported = format_name != "tflite" or platform.system() == "Linux"
        reason = "Dependencies detected; model execution has not been tested."
        if not supported:
            reason = "LiteRT Torch conversion requires a Linux worker; use Python 3.11."
        elif missing:
            reason = f"Install requirements-{ 'ml' if format_name == 'onnx' else 'tflite' }.txt in the backend environment."
        output[format_name] = {"available": not missing and supported, "missing": missing, "reason": reason}
    return output


def _imports(format_name: str) -> dict:
    caps = runtime_capabilities()[format_name]
    if not caps["available"]:
        raise RuntimeUnavailable(caps["reason"] + (" Missing: " + ", ".join(caps["missing"]) if caps["missing"] else ""))
    names = ["torch", "torchvision", "numpy", "PIL.Image"]
    names += ["onnx", "onnxruntime"] if format_name == "onnx" else ["litert_torch", "ai_edge_litert.interpreter"]
    try:
        return {name: importlib.import_module(name) for name in names}
    except Exception as error:
        raise RuntimeUnavailable(
            f"The {format_name} worker could not import its runtime ({type(error).__name__}). "
            "Use a compatible torch/torchvision pair in a clean Python 3.11 or 3.12 environment."
        ) from error


def _hash_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _artifact(profile: str, format_name: str, path: Path) -> dict:
    return {"profile": profile, "format": format_name, "path": str(path),
            "sha256": _hash_file(path), "size_bytes": path.stat().st_size}


def _settings(request: dict) -> dict:
    values = {"warmup_runs": 3, "measured_runs": 10, "threads": 1, "atol": 0.0001, "rtol": 0.001}
    values.update(request.get("settings", {}))
    for key, low, high in (("warmup_runs", 0, 100), ("measured_runs", 3, 1000), ("threads", 1, 16)):
        if type(values[key]) is not int or not low <= values[key] <= high:
            raise ValueError(f"{key} must be an integer from {low} to {high}")
    import math
    for key in ("atol", "rtol"):
        if not isinstance(values[key], (float, int)) or not math.isfinite(values[key]) or not 0 <= values[key] <= 1:
            raise ValueError(f"{key} must be finite and from 0 to 1")
    return values


def _load_model(model_id: str, modules: dict):
    vision = modules["torchvision"]
    weights = (vision.models.MobileNet_V2_Weights.IMAGENET1K_V2 if model_id == "mobilenet_v2"
               else vision.models.ResNet18_Weights.IMAGENET1K_V1)
    try:
        model = getattr(vision.models, model_id)(weights=weights, progress=False).cpu().eval()
    except Exception as error:
        raise RuntimeUnavailable(
            "Could not load the official pretrained weights. The first run needs download access "
            "to download.pytorch.org or a populated TORCH_HOME cache."
        ) from error
    return model, weights


def _load_inputs(dataset: dict, weights, modules: dict):
    entries = dataset.get("entries", [])
    if not 1 <= len(entries) <= MAX_IMAGES:
        raise ValueError(f"Dataset must contain 1 to {MAX_IMAGES} labelled images")
    if len(entries) * 3 * 224 * 224 * 4 > 512 * 1024 * 1024:
        raise ValueError("Preprocessed dataset exceeds 512 MiB; use fewer images")
    tensors, labels = [], []
    image_module = modules["PIL.Image"]
    transform = weights.transforms()
    archive_path = Path(dataset["path"])
    if dataset.get("sha256") and _hash_file(archive_path) != dataset["sha256"]:
        raise ValueError("Stored dataset checksum changed; upload the dataset again")
    with zipfile.ZipFile(archive_path) as archive:
        for entry in entries:
            path = str(entry["path"])
            member = PurePosixPath(path)
            if member.is_absolute() or ".." in member.parts or "\\" in path or ":" in path:
                raise ValueError("Dataset contains an unsafe image path")
            label = entry["label"]
            if type(label) is not int or not 0 <= label < 1000:
                raise ValueError("Labels must be ImageNet-1K class indices from 0 to 999")
            info = archive.getinfo(path)
            if not 0 < info.file_size <= MAX_IMAGE_BYTES:
                raise ValueError(f"Image must contain at most {MAX_IMAGE_BYTES // (1024 * 1024)} MB: {path}")
            with archive.open(info) as source:
                content = source.read(MAX_IMAGE_BYTES + 1)
            if len(content) > MAX_IMAGE_BYTES:
                raise ValueError("Image exceeds the byte limit")
            with warnings.catch_warnings():
                warnings.simplefilter("error", image_module.DecompressionBombWarning)
                with image_module.open(io.BytesIO(content)) as image:
                    if image.width * image.height > MAX_IMAGE_PIXELS or min(image.size) < 1:
                        raise ValueError("Image dimensions exceed the 20-million-pixel limit")
                    tensors.append(transform(image.convert("RGB")).unsqueeze(0).contiguous())
            labels.append(label)
    return tensors, labels


def _numpy_output(output, np):
    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    output = np.asarray(output)
    if output.ndim != 2 or output.shape[0] != 1 or output.shape[1] < 2 or not np.isfinite(output).all():
        raise ValueError("Inference must return finite class scores with shape [1, class_count]")
    return output


def _evaluate(infer, inputs, reference_outputs, labels, settings, np) -> dict:
    outputs = [_numpy_output(infer(value), np) for value in inputs]
    reference = reference_outputs if reference_outputs is not None else outputs
    predictions = [int(value.argmax(axis=1)[0]) for value in outputs]
    reference_predictions = [int(value.argmax(axis=1)[0]) for value in reference]
    differences = [np.abs(candidate.astype("float64") - baseline.astype("float64"))
                   for candidate, baseline in zip(outputs, reference)]
    result = classification_metrics(reference_predictions, predictions, labels)
    result["output_mae"] = float(sum(value.sum() for value in differences) / sum(value.size for value in differences))
    result["output_max_abs"] = float(max(value.max() for value in differences))
    result["within_tolerance"] = bool(all(np.allclose(candidate, baseline, atol=settings["atol"], rtol=settings["rtol"])
                                          for candidate, baseline in zip(outputs, reference)))
    result["tolerance_failure_count"] = sum(not np.allclose(candidate, baseline, atol=settings["atol"], rtol=settings["rtol"]) for candidate, baseline in zip(outputs, reference))
    result["outputs"] = outputs
    return result


def _time_profiles(profiles, settings, failures=None) -> dict:
    """Rotate profile order across rounds, using the identical first input."""
    failed = failures if failures is not None else {}
    for name, infer, sample in profiles:
        try:
            for _ in range(settings["warmup_runs"]):
                infer(sample)
        except Exception as exc:
            if failures is None:
                raise
            failed[name] = exc
    timings = {name: [] for name, _, _ in profiles}
    for round_index in range(settings["measured_runs"]):
        offset = round_index % len(profiles)
        for name, infer, sample in profiles[offset:] + profiles[:offset]:
            if name in failed:
                continue
            try:
                started = time.perf_counter_ns()
                infer(sample)
                timings[name].append((time.perf_counter_ns() - started) / 1_000_000)
            except Exception as exc:
                if failures is None:
                    raise
                failed[name] = exc
    return {name: {"latency_mean_ms": sum(values) / len(values), "latency_p50_ms": percentile(values, 50), "latency_p95_ms": percentile(values, 95),
                   "latency_samples_ms": values} for name, values in timings.items() if name not in failed}


def _ort_session(ort, path_or_bytes, threads: int, diagnostic: bool = False):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = (ort.GraphOptimizationLevel.ORT_DISABLE_ALL if diagnostic
                                        else ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
    return ort.InferenceSession(path_or_bytes, sess_options=options, providers=["CPUExecutionProvider"])


def _convert_onnx(model, sample, output_dir: Path, modules: dict, settings: dict):
    torch, ort = modules["torch"], modules["onnxruntime"]
    if "optimize" not in inspect.signature(torch.onnx.export).parameters:
        raise RuntimeUnavailable("This worker requires torch.onnx.export with optimize support; install torch>=2.7.")
    converted, artifacts = [], []
    for profile, optimize, label in (("standard", True, "Standard ONNX export (optimize=True)"),
                                     ("dashboard", False, "Dashboard ONNX configuration (optimize=False)")):
        path = output_dir / f"{profile}.onnx"
        # A converter must not change the source used by the next profile.
        conversion_model = copy.deepcopy(model).eval()
        started = time.perf_counter()
        program = torch.onnx.export(conversion_model, (sample,), dynamo=True, optimize=optimize,
                                    input_names=["image"], output_names=["logits"],
                                    external_data=False, verbose=False)
        if program is None:
            raise RuntimeUnavailable("The ONNX exporter did not return an ONNXProgram")
        program.save(str(path), external_data=False)
        elapsed = time.perf_counter() - started
        session = _ort_session(ort, str(path), settings["threads"])
        input_name = session.get_inputs()[0].name
        infer = lambda value, session=session, input_name=input_name: session.run(None, {input_name: value})[0]
        converted.append({"profile": profile, "label": label, "infer": infer, "path": path,
                          "conversion_seconds": elapsed, "optimize": optimize})
        artifacts.append(_artifact(profile, "onnx", path))
    return converted, artifacts


def _convert_tflite(model, sample, output_dir: Path, modules: dict, settings: dict):
    converter, interpreter_module = modules["litert_torch"], modules["ai_edge_litert.interpreter"]
    converted, artifacts = [], []
    for profile, label in (("standard", "Standard LiteRT Torch FP32"),
                           ("dashboard", "Dashboard LiteRT Torch FP32 (identical control)")):
        path = output_dir / f"{profile}.tflite"
        conversion_model = copy.deepcopy(model).eval()
        started = time.perf_counter()
        edge_model = converter.convert(conversion_model, (sample,))
        edge_model.export(str(path))
        elapsed = time.perf_counter() - started
        interpreter = interpreter_module.Interpreter(model_path=str(path), num_threads=settings["threads"])
        interpreter.allocate_tensors()
        inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1 or tuple(inputs[0]["shape"]) != tuple(sample.shape):
            raise ValueError("Converted LiteRT tensor signature differs from the supported NCHW input")
        input_index, output_index = inputs[0]["index"], outputs[0]["index"]
        def infer(value, runner=interpreter, input_index=input_index, output_index=output_index):
            runner.set_tensor(input_index, value)
            runner.invoke()
            return runner.get_tensor(output_index)
        converted.append({"profile": profile, "label": label, "infer": infer, "path": path,
                          "conversion_seconds": elapsed, "optimize": None})
        artifacts.append(_artifact(profile, "tflite", path))
    return converted, artifacts


def _capture_leaves(model, sample, torch):
    captures = defaultdict(list)
    handles = []
    leaves = [(name, layer) for name, layer in model.named_modules() if name and not list(layer.children())]
    for name, layer in leaves:
        def hook(_, inputs, output, name=name):
            if isinstance(output, torch.Tensor):
                # Clone at the boundary: later in-place activations must not alter it.
                captures[name].append(output.detach().clone().cpu().numpy())
        handles.append(layer.register_forward_hook(hook))
    try:
        model(sample)
    finally:
        for handle in handles:
            handle.remove()
    return leaves, captures


def _scope_names(node) -> list[str]:
    metadata = {item.key: item.value for item in node.metadata_props}
    raw = metadata.get("pkg.torch.onnx.name_scopes", "[]")
    if len(raw) > 20_000:
        return []
    try:
        scopes = ast.literal_eval(raw)
        # The exporter appends the FX node name after the module names. It can
        # equal an unrelated leaf name (e.g. functional "relu"); never map it.
        return [value for value in scopes[:-1] if isinstance(value, str)] if isinstance(scopes, list) else []
    except (ValueError, SyntaxError, TypeError):
        return []


def _layer_rows(model_id, leaves, captures, converted, format_name, first_input, modules, settings):
    """Conservative mapping: inspect only exact, unfused single-op leaf scopes.

    Optimized graphs can retain stale origin names after fusion. We deliberately do
    not treat those names as a one-to-one mapping. Final logits are always known.
    """
    np = modules["numpy"]
    final_layer = "classifier.1" if model_id == "mobilenet_v2" else "fc"
    result = []
    allowed_ops = {"Conv2d": {"Conv"}, "BatchNorm2d": {"BatchNormalization"},
                   "ReLU": {"Relu"}, "ReLU6": {"Clip"}, "Linear": {"Gemm"},
                   "AdaptiveAvgPool2d": {"GlobalAveragePool"}, "MaxPool2d": {"MaxPool"}}
    for converted_model in converted:
        profile = converted_model["profile"]
        rows = {name: {"name": name, "operation": type(layer).__name__, "profile": profile,
                       "status": "unmapped", "mae": None, "max_abs": None,
                       "detail": "No proven one-to-one tensor mapping; fusion, decomposition, or missing metadata may apply."}
                for name, layer in leaves}
        final_output = _numpy_output(converted_model["infer"](first_input), np)
        candidates = {}
        diagnostic_values = {}
        if format_name == "onnx" and converted_model["optimize"] is False:
            try:
                onnx = modules["onnx"]
                graph = onnx.load(str(converted_model["path"]))
                graph = onnx.shape_inference.infer_shapes(graph)
                value_info = {value.name: value for value in list(graph.graph.value_info) + list(graph.graph.output)}
                scoped_nodes = defaultdict(list)
                for node in graph.graph.node:
                    matches = [name for name in _scope_names(node) if name in rows]
                    if matches:
                        scoped_nodes[max(matches, key=len)].append(node)
                for name, layer in leaves:
                    nodes = scoped_nodes[name]
                    if len(nodes) != 1 or len(captures[name]) != 1:
                        continue
                    node = nodes[0]
                    if (node.domain not in {"", "ai.onnx"}
                            or node.op_type not in allowed_ops.get(type(layer).__name__, set())
                            or len(node.output) != 1):
                        continue
                    value_name = node.output[0]
                    if value_name in value_info:
                        candidates[name] = value_name
                # Bound diagnostic memory to the declared small architectures and one image.
                diagnostic = copy.deepcopy(graph)
                del diagnostic.graph.output[:]
                unique_outputs = list(dict.fromkeys(candidates.values()))
                for name in unique_outputs:
                    diagnostic.graph.output.append(value_info[name])
                if unique_outputs:
                    session = _ort_session(modules["onnxruntime"], diagnostic.SerializeToString(), settings["threads"], True)
                    values = session.run(unique_outputs, {session.get_inputs()[0].name: first_input})
                    diagnostic_values = dict(zip(unique_outputs, values))
            except Exception as error:
                for row in rows.values():
                    row["detail"] = f"Intermediate extraction unavailable ({type(error).__name__}); not classified as a numerical failure."
        for name, row in rows.items():
            if len(captures[name]) != 1:
                row["detail"] = "Layer is reused or has no single tensor output; invocation alignment is required."
                continue
            candidate = final_output if name == final_layer else diagnostic_values.get(candidates.get(name))
            if candidate is None:
                continue
            reference = captures[name][0]
            if candidate.shape != reference.shape:
                row["detail"] = "Candidate tensor shape does not match; mapping withheld."
                continue
            if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
                row.update(status="nonfinite", detail="Mapped diagnostic tensor contains NaN or infinity.")
                continue
            diff = np.abs(candidate.astype("float64") - reference.astype("float64"))
            row.update(mae=float(diff.mean()), max_abs=float(diff.max()),
                       status="pass" if np.allclose(candidate, reference, atol=settings["atol"], rtol=settings["rtol"]) else "drift",
                       detail=("Final classifier output on the first dataset image." if name == final_layer else
                               "Exact leaf scope and single operator; first dataset image; diagnostic graph disables ORT optimizations."))
        result.extend(rows.values())
    return result


def _target(target: str, format_name: str) -> dict:
    if target == "esp32":
        return {"id": target, "name": "ESP32", "status": "micro_compatibility_pending", "latency_ms": None,
                "notes": ["Goal only. No physical ESP32 or remote hardware was measured.",
                          "Model size, supported Micro operators and tensor arena allocation must be validated on your exact ESP32.",
                          "A small model, board-specific memory budget, supported Micro operators and firmware validation are required; int8 is often appropriate for MCU deployment.",
                          "ONNX is not a direct TFLite Micro artifact." if format_name == "onnx" else
                          "A .tflite extension does not establish TFLite Micro compatibility."]}
    return {"id": target, "name": "Raspberry Pi", "status": "device_validation_pending", "latency_ms": None,
            "notes": ["Goal only. Host CPU timing cannot be reported as Raspberry Pi latency.",
                      "Confirm Pi model, RAM, operating system and runtime build before deployment.",
                      "Operator compatibility, memory use and actual board timing remain unverified."]}


def run_benchmark(request: dict, dataset: dict, output_dir: Path) -> dict:
    """Run a real, reproducible host-CPU comparison or raise an actionable error."""
    model_id = request.get("model_id")
    format_name = request.get("format")
    target_id = request.get("target")
    if model_id not in MODELS or format_name not in {"onnx", "tflite"} or target_id not in {"raspberry_pi", "esp32"}:
        raise ValueError("Select a supported model, conversion format and deployment goal")
    settings = _settings(request)
    modules = _imports(format_name)
    torch, np = modules["torch"], modules["numpy"]
    old_threads = torch.get_num_threads()
    torch.set_num_threads(settings["threads"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model, weights = _load_model(model_id, modules)
        inputs, labels = _load_inputs(dataset, weights, modules)
        arrays = [value.numpy() for value in inputs]
        with torch.no_grad():
            reference_infer = lambda value: model(value)
            reference = _evaluate(reference_infer, inputs, None, labels, settings, np)
            reference_outputs = reference.pop("outputs")
            source_path = output_dir / "reference-state-dict.pt"
            torch.save(model.state_dict(), source_path)
            source_artifact = _artifact("pytorch", "pytorch_state_dict", source_path)
            convert = _convert_onnx if format_name == "onnx" else _convert_tflite
            converted, artifacts = convert(model, inputs[0], output_dir, modules, settings)
            profiles = [("pytorch", reference_infer, inputs[0])] + [(item["profile"], item["infer"], arrays[0]) for item in converted]
            timings = _time_profiles(profiles, settings)
            reference.update(profile="pytorch", label="PyTorch reference", conversion_seconds=None,
                             size_bytes=source_artifact["size_bytes"], **timings["pytorch"])
            metrics = [reference]
            for item, artifact in zip(converted, artifacts):
                measured = _evaluate(item["infer"], arrays, reference_outputs, labels, settings, np)
                measured.pop("outputs")
                measured.update(profile=item["profile"], label=item["label"], conversion_seconds=item["conversion_seconds"],
                                size_bytes=artifact["size_bytes"], **timings[item["profile"]])
                metrics.append(measured)
            leaves, captures = _capture_leaves(model, inputs[0], torch)
            layers = _layer_rows(model_id, leaves, captures, converted, format_name, arrays[0], modules, settings)
    finally:
        torch.set_num_threads(old_threads)
    distributions = ["torch", "torchvision", "numpy", "Pillow", "onnx", "onnxruntime", "onnxscript", "litert-torch", "ai-edge-litert"]
    versions = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "schema_version": 1, "source": "measured", "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": comparison_summary(metrics[1], metrics[2]),
        "model": {"id": model_id, "name": MODELS[model_id], "weights": str(weights),
                  "input_shape": [1, 3, 224, 224], "dtype": "float32", "class_count": 1000},
        "dataset": {**{key: dataset.get(key) for key in ("id", "name", "sha256")}, "image_count": len(labels), "class_count": len(set(labels))},
        "environment": {"benchmark_scope": "host_cpu", "execution_location": "backend_host",
                        "platform": platform.platform(), "architecture": platform.machine(),
                        "processor": platform.processor() or "not reported", "logical_cpus": os.cpu_count(),
                        "python": platform.python_version(), "versions": versions, "settings": settings,
                        "provider": "CPUExecutionProvider" if format_name == "onnx" else "LiteRT CPU interpreter"},
        "target": _target(target_id, format_name), "metrics": metrics, "layers": layers,
        "artifacts": [source_artifact] + artifacts,
        "methodology": [
            "All profiles use the same official pretrained weights, labelled images and torchvision weight-specific preprocessing.",
            "Top-1 accuracy uses the uploaded ImageNet-1K integer labels; agreement compares predictions with PyTorch. These are different metrics.",
            "Accuracy deltas are percentage points. Output errors compare raw logits against PyTorch, using absolute and relative tolerance.",
            f"CPU batch=1; {settings['threads']} intra-op thread(s); {settings['warmup_runs']} warm-ups per profile and {settings['measured_runs']} timed calls per profile.",
            "Timing uses the first preprocessed image, rotates profile order, and excludes image decoding, preprocessing, model load, conversion and diagnostics. Runtime invocation/output-copy overhead is included.",
            "Conversion time includes exporter execution and artifact serialization. Model download, profile-isolation copy, dataset loading and runtime session initialization are excluded.",
            "ONNX profiles use the same torch.onnx dynamo exporter; standard optimize=True, dashboard optimize=False; both benchmark sessions enable all ORT graph optimizations." if format_name == "onnx" else
            "Both TFLite profiles use identical litert_torch.convert FP32 settings. The second is a repeat control, not a distinct improved converter.",
            "Leaf-module diagnostics use the first labelled image in a separate pass after timing; reused modules and unproven mappings are explicitly unavailable.",
            "Stored model size is serialized bytes on disk; it is not peak runtime RAM.",
        ],
        "limitations": [
            "These are backend-host CPU measurements, including when the host is a cloud VM. No Raspberry Pi or ESP32 measurements or simulated device latencies are reported.",
            "No guarantee of higher accuracy: equal or worse configurations are retained. These two profiles do not represent all traditional converters.",
            "Accuracy describes this uploaded sample only. It is not published full ImageNet accuracy or a statistical significance claim; one changed prediction moves accuracy by 100/N percentage points.",
            "Conversion profiles run standard first, dashboard second; compiler/cache warm-up can affect single conversion-time measurements. Repeat whole jobs before drawing timing conclusions.",
            "Intermediate diagnostics cover only conservative one-to-one mappings in the unoptimized ONNX artifact. Fused optimized tensors and TFLite internal tensors are not aligned in this MVP.",
            "Diagnostic graph outputs can prevent fusions, so intermediate diagnostic values are distinct from optimized runtime execution. Final-output fidelity is measured using the actual benchmark runtime.",
            "Peak RAM, energy use, hardware accelerators, quantization and board firmware are not benchmarked.",
        ],
    }
