"""Bounded FP32/QDQ operation comparisons; tensor drift is not causal proof."""
from collections import Counter
import copy
import math

from app.services.quantization import OP_TYPES
from app.services import benchmark as b


def compare_activations(fp32_path, quantized_path, arrays, settings, sample_limit=3, node_limit=128):
    """Match preserved unique ONNX nodes and capture their operation outputs.

    No PyTorch-to-ONNX layer identity is inferred. Diagnostic sessions disable
    graph optimizations; deployment latency uses the ordinary optimized graph.
    Capture eight operations at a time and aggregate, rather than storing every
    intermediate activation for the entire dataset.
    """
    import numpy as np
    import onnx
    import onnxruntime as ort
    float_graph = onnx.shape_inference.infer_shapes(onnx.load(fp32_path, load_external_data=False))
    quant_graph = onnx.shape_inference.infer_shapes(onnx.load(quantized_path, load_external_data=False))
    float_counts = Counter(n.name for n in float_graph.graph.node)
    quant_counts = Counter(n.name for n in quant_graph.graph.node)
    quant_nodes = {n.name: n for n in quant_graph.graph.node}
    rows, pairs = [], []
    for node in float_graph.graph.node:
        if node.op_type not in OP_TYPES:
            continue
        row = {"name": node.name, "operation": node.op_type, "status": "unmapped", "mae": None, "max_abs": None,
               "nrmse": None, "sample_count": 0, "scope": "calibration_diagnostic", "provenance": "UNAVAILABLE",
               "mapping": "Unique preserved ONNX node name, operation type and output boundary; no PyTorch layer mapping."}
        other = quant_nodes.get(node.name)
        if not node.name or float_counts[node.name] != 1 or quant_counts[node.name] != 1 or other is None or other.op_type != node.op_type or len(node.output) != 1 or len(other.output) != 1:
            row["reason_code"] = "ambiguous_or_missing_preserved_operation"
            row["detail"] = "No unambiguous preserved operation correspondence was established."
        elif len(pairs) >= node_limit:
            row["reason_code"] = "diagnostic_node_budget"
            row["detail"] = "Operation was outside the bounded diagnostic node budget."
        else:
            row.update(float_tensor=node.output[0], quantized_tensor=other.output[0])
            pairs.append(row)
        rows.append(row)

    def capture_session(graph, names):
        augmented = copy.deepcopy(graph)
        infos = {v.name: v for v in list(graph.graph.value_info) + list(graph.graph.input) + list(graph.graph.output)}
        existing = {v.name for v in augmented.graph.output}
        known_bytes = 0
        for name in names:
            info = infos.get(name)
            if info and info.type.tensor_type.elem_type not in (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16):
                raise ValueError("Diagnostic operation output is not floating-point")
            if info:
                dims = [d.dim_value for d in info.type.tensor_type.shape.dim]
                if dims and all(d > 0 for d in dims):
                    known_bytes += math.prod(dims) * 4
            if name not in existing:
                augmented.graph.output.append(copy.deepcopy(info) if info else onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
                existing.add(name)
        if known_bytes > 128 * 1024 * 1024:
            raise ValueError("Diagnostic capture exceeds the 128 MiB chunk budget")
        session = b._ort_session(ort, augmented.SerializeToString(), settings["threads"], diagnostic=True)
        input_name = session.get_inputs()[0].name
        return lambda value: session.run(names, {input_name: value})

    samples = arrays[:sample_limit]
    for offset in range(0, len(pairs), 8):
        chunk = pairs[offset:offset + 8]
        try:
            float_infer = capture_session(float_graph, [x["float_tensor"] for x in chunk])
            quant_infer = capture_session(quant_graph, [x["quantized_tensor"] for x in chunk])
            stats = [{"sum_abs": 0., "sum_squared": 0., "reference_squared": 0., "elements": 0, "max": 0., "failures": 0} for _ in chunk]
            for sample in samples:
                expected, actual = float_infer(sample), quant_infer(sample)
                if sum(x.nbytes for x in expected + actual) > 256 * 1024 * 1024:
                    raise ValueError("Observed diagnostic capture exceeds 256 MiB")
                for row, accumulator, ref, out in zip(chunk, stats, expected, actual):
                    if ref.shape != out.shape or not np.isfinite(ref).all() or not np.isfinite(out).all():
                        raise ValueError("Matched activation has a different shape or non-finite values")
                    difference = out.astype(np.float64) - ref
                    accumulator["sum_abs"] += float(np.abs(difference).sum())
                    accumulator["sum_squared"] += float(np.square(difference).sum())
                    accumulator["reference_squared"] += float(np.square(ref.astype(np.float64)).sum())
                    accumulator["elements"] += ref.size
                    accumulator["max"] = max(accumulator["max"], float(np.abs(difference).max()))
                    accumulator["failures"] += int(not np.allclose(ref, out, atol=settings["atol"], rtol=settings["rtol"]))
                    row["expected_shape"] = list(ref.shape)
            for row, accumulator in zip(chunk, stats):
                count = accumulator["elements"]
                row.update(status="drift" if accumulator["failures"] else "pass", provenance="MEASURED", reason_code="measured_preserved_boundary", sample_count=len(samples),
                           mae=accumulator["sum_abs"] / count, max_abs=accumulator["max"],
                           nrmse=math.sqrt(accumulator["sum_squared"] / count) / max(math.sqrt(accumulator["reference_squared"] / count), 1e-12),
                           tolerance_failure_count=accumulator["failures"],
                           detail="Operation output before downstream QDQ, compared on calibration images with graph optimization disabled. Propagated drift is not proof of this operation being its cause.")
        except Exception as exc:
            for row in chunk:
                row.update(status="unmapped", provenance="UNAVAILABLE", reason_code="diagnostic_capture_failed", mae=None, max_abs=None, nrmse=None,
                           detail=f"Diagnostic capture unavailable: {type(exc).__name__}: {str(exc)[:250]}")
    measured = sum(row["provenance"] == "MEASURED" for row in rows)
    return {"status": "completed" if measured == len(rows) and measured else "partial" if measured else "unavailable",
            "sample_count": len(samples), "compared_operations": measured, "eligible_operations": len(rows), "rows": rows,
            "method": "Chunked preserved ONNX operation-output comparison; calibration only; unoptimized diagnostic sessions.",
            "limitations": "NRMSE includes upstream error propagation. Exclusion probes provide separate controlled validation evidence."}
