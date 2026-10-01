"""Build static QDQ candidates; evaluation stays in the existing benchmark engine."""
from collections import Counter
import inspect
import time

OP_TYPES = ["Conv", "Gemm", "MatMul"]


def require_fp32(graph):
    import onnx
    if any(n.op_type in {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear", "QLinearConv", "QLinearMatMul", "MatMulInteger", "ConvInteger"}
           for n in graph.graph.node):
        raise ValueError("Supply an FP32 ONNX baseline, not an already quantized model")
    floating = {onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE, onnx.TensorProto.BFLOAT16}
    if any(t.data_type in floating for t in graph.graph.initializer):
        raise ValueError("This experiment requires FP32 floating-point weights")
    if any(v.type.tensor_type.elem_type != onnx.TensorProto.FLOAT for v in list(graph.graph.input) + list(graph.graph.output)):
        raise ValueError("This experiment requires FP32 input and output tensors")
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS) for n in graph.graph.node for a in n.attribute):
        raise ValueError("Control-flow subgraphs are outside this initial quantization experiment")


def quantization_inventory(graph):
    """Count observed QDQ boundaries; do not infer execution precision from a label."""
    import onnx
    nodes = list(graph.graph.node)
    producers = {name: n for n in nodes for name in n.output}
    consumers = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    eligible = [n for n in nodes if n.op_type in OP_TYPES]
    surrounded = [n for n in eligible if any(producers.get(x) is not None and producers[x].op_type == "DequantizeLinear" for x in n.input)
                  and any(c.op_type == "QuantizeLinear" for x in n.output for c in consumers.get(x, []))]
    return {"operator_counts": dict(Counter(n.op_type for n in nodes)),
            "eligible_operation_count": len(eligible), "qdq_operation_count": len(surrounded),
            "qdq_operations": [{"name": n.name, "operation": n.op_type} for n in surrounded],
            "weight_dtypes": dict(Counter(onnx.TensorProto.DataType.Name(t.data_type) for t in graph.graph.initializer)),
            "interpretation": "QDQ graph coverage, not a kernel execution trace. Float32 operations may remain; this is not a fully integer deployment claim."}


def build_static_int8(baseline_path, output_path, calibration_inputs, config):
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    graph = onnx.load(str(baseline_path), load_external_data=False)
    require_fp32(graph)
    if not any(n.op_type in OP_TYPES for n in graph.graph.node):
        raise ValueError("No supported Conv, Gemm or MatMul operations to quantize")
    input_name = graph.graph.input[0].name

    class Images(CalibrationDataReader):
        def __init__(self):
            self.rewind()

        def get_next(self):
            value = next(self.values, None)
            return None if value is None else {input_name: value}

        def rewind(self):
            self.values = iter(calibration_inputs)

    started = time.perf_counter()
    prepared = output_path.with_name("quantization-prepared.onnx")
    # Shape inference only: retain graph topology for subsequent diagnostic work.
    onnx.save(onnx.shape_inference.infer_shapes(graph), prepared)
    cache_options = {}
    reader = Images()
    # Current pinned Windows ORT supports explicit calibration caches. Keep the
    # ranges next to the experiment for reproduction and avoid hidden temp files.
    if "calibration_cache_path" in inspect.signature(quantize_static).parameters:
        from onnxruntime.quantization.calibrate import create_calibrator, save_tensors_data
        calibrator = create_calibrator(prepared, OP_TYPES, augmented_model_path=str(output_path.with_name("calibration-augmented.onnx")),
                                       calibrate_method=getattr(CalibrationMethod, config["calibration_method"]), providers=["CPUExecutionProvider"])
        calibrator.collect_data(reader)
        ranges = calibrator.compute_data()
        cache = output_path.with_name("calibration-ranges.json")
        save_tensors_data(ranges, cache)
        del calibrator
        reader, cache_options = None, {"calibration_cache_path": cache}
    quantize_static(prepared, output_path, reader, quant_format=QuantFormat.QDQ,
                    op_types_to_quantize=OP_TYPES, per_channel=config["per_channel"],
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    calibrate_method=getattr(CalibrationMethod, config["calibration_method"]),
                    calibration_providers=["CPUExecutionProvider"], use_external_data_format=False,
                    extra_options={"WeightSymmetric": True}, **cache_options)
    seconds = time.perf_counter() - started
    converted = onnx.load(str(output_path), load_external_data=False)
    onnx.checker.check_model(converted)
    inventory = quantization_inventory(converted)
    if inventory["qdq_operation_count"] == 0:
        raise ValueError("Quantizer produced no observed QDQ operation boundaries")
    return seconds, inventory
