"""Small, synthetic TFLite fixture for live profiling; no converter quality claim.

Run with the desktop engine's Python. Generated binaries/images stay in .cache.
This hand-authored classifier separates dark/light pixels; it is not a trained
real-world vision model, and it must never substitute for a developer's model.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile


def build_model(size=32, quantized=True):
    import flatbuffers
    import numpy as np
    from ai_edge_litert import schema_py_generated as s
    features = size * size * 3
    model = s.ModelT(); model.version = 3
    model.description = 'EdgeLens synthetic dark/light fixture; not converter or accuracy evidence'
    opcode = s.OperatorCodeT(); opcode.builtinCode = s.BuiltinOperator.FULLY_CONNECTED
    opcode.deprecatedBuiltinCode = s.BuiltinOperator.FULLY_CONNECTED; opcode.version = 4 if quantized else 1
    model.operatorCodes = [opcode]
    buffers = [s.BufferT() for _ in range(3)]
    weights = np.array([-1] * features + [1] * features, dtype=np.int8)
    buffers[1].data = weights.view(np.uint8) if quantized else (weights.astype('<f4') / features).view(np.uint8)
    bias_scale = 1 / (255 * features)
    buffers[2].data = np.rint(np.array([.5, -.5]) / bias_scale).astype('<i4').view(np.uint8) if quantized else np.array([.5, -.5], dtype='<f4').view(np.uint8)
    model.buffers = buffers
    graph = s.SubGraphT(); graph.name = 'synthetic_dark_light'
    tensors = []
    for i, (name, shape, buffer) in enumerate([('image', [1, size, size, 3], 0), ('weights', [2, features], 1), ('bias', [2], 2), ('scores', [1, 2], 0)]):
        tensor = s.TensorT(); tensor.name = name; tensor.shape = np.array(shape, dtype=np.int32); tensor.buffer = buffer
        tensor.type = s.TensorType.FLOAT32
        if quantized:
            tensor.type = s.TensorType.INT32 if i == 2 else s.TensorType.INT8
            params = s.QuantizationParametersT(); params.scale = np.array([[1/255, 1/features, bias_scale, .01][i]], dtype=np.float32)
            params.zeroPoint = np.array([-128 if i == 0 else 0], dtype=np.int64); tensor.quantization = params
        tensors.append(tensor)
    graph.tensors = tensors; graph.inputs = np.array([0], dtype=np.int32); graph.outputs = np.array([3], dtype=np.int32)
    op = s.OperatorT(); op.opcodeIndex = 0; op.inputs = np.array([0, 1, 2], dtype=np.int32); op.outputs = np.array([3], dtype=np.int32)
    op.builtinOptionsType = s.BuiltinOptions.FullyConnectedOptions; op.builtinOptions = s.FullyConnectedOptionsT()
    graph.operators = [op]; model.subgraphs = [graph]
    builder = flatbuffers.Builder(8192); offset = model.Pack(builder); builder.Finish(offset, file_identifier=b'TFL3')
    return bytes(builder.Output())


def create_demo(directory, size=32, count=300):
    import numpy as np
    from PIL import Image
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    content = build_model(size)
    model_path = directory / 'synthetic-int8.tflite'; model_path.write_bytes(content)
    spec = {'name': 'Synthetic 32x32 INT8 dark/light — pipeline test only', 'format': 'tflite',
            'input_shape': [1, size, size, 3], 'layout': 'NHWC', 'class_count': 2,
            'scale': 1/255, 'mean': [0, 0, 0], 'std': [1, 1, 1], 'resize': 'stretch', 'trusted_source': True}
    (directory / 'spec.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    rng = np.random.default_rng(20261002)
    with zipfile.ZipFile(directory / 'test.zip', 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('labels.json', json.dumps({'dark': 0, 'light': 1}))
        for i in range(count):
            low, high = (15, 90) if i % 2 == 0 else (165, 245)
            image = Image.fromarray(rng.integers(low, high, (size, size, 3), dtype=np.uint8))
            stream = io.BytesIO(); image.save(stream, format='PNG')
            archive.writestr(f'{"dark" if i % 2 == 0 else "light"}/{i:04d}.png', stream.getvalue())
    metadata = {'model_sha256': hashlib.sha256(content).hexdigest(), 'size_bytes': len(content), 'sample_count': count,
                'purpose': 'Synthetic pipeline/profiling fixture only; not trained classifier quality, conversion superiority or physical-device evidence.'}
    (directory / 'fixture.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / '.cache' / 'edge-profile-demo')
    args = parser.parse_args()
    print(json.dumps(create_demo(args.output), indent=2))
    print(f'Fixture files: {args.output.resolve()}')
