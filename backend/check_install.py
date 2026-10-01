"""Check the Windows runtime with a tiny local export and actual inference."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from uuid import uuid4

CORE = ('torch', 'torchvision', 'numpy', 'Pillow', 'onnx', 'onnxscript',
        'onnxruntime', 'ai-edge-litert', 'backports.strenum', 'fastapi', 'uvicorn', 'pydantic', 'httpx', 'pyserial')


def check(frontend_dir, report_path):
    # Exporter progress uses Unicode. Redirected Windows consoles otherwise
    # default to legacy code pages and can fail an otherwise valid conversion.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    result = {'checked_at': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'platform': platform.platform(),
              'status': 'failed', 'checks': {}, 'versions': {}}
    scratch = None
    try:
        if sys.platform != 'win32' or sys.version_info[:2] != (3, 12) or sys.maxsize < 2**32:
            raise RuntimeError('This distribution requires Windows x64 and 64-bit Python 3.12.')
        requirements = Path(__file__).with_name('requirements-windows-tested.txt')
        expected = dict(line.strip().split('==', 1) for line in requirements.read_text().splitlines()
                        if '==' in line and not line.lstrip().startswith('#'))
        for name in CORE:
            actual = importlib.metadata.version(name)
            result['versions'][name] = actual
            if actual != expected[name.lower()]:
                raise RuntimeError(f'{name} is {actual}; this build expects {expected[name.lower()]}. Run setup-windows.cmd.')
        print('Checking native Python/ONNX/TFLite libraries...', flush=True)
        import numpy as np
        import torch
        import torchvision
        import onnx
        import onnxruntime as ort
        import serial
        from ai_edge_litert.interpreter import Interpreter
        from app.main import create_app
        result['checks']['native_imports'] = True
        if torch.version.cuda is not None:
            raise RuntimeError('This distribution expects CPU PyTorch, not a CUDA build.')
        # Write only synthetic test artifacts, outside stored developer models/reports.
        base = (report_path.parent if report_path else Path(tempfile.gettempdir()) / 'EdgeLens-check').resolve()
        base.mkdir(parents=True, exist_ok=True)
        scratch = base / ('runtime-check-' + uuid4().hex)
        scratch.mkdir()
        print('Checking a tiny PyTorch export and ONNX execution...', flush=True)
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 2, 1), torch.nn.ReLU(),
                                    torch.nn.AdaptiveAvgPool2d((1, 1)), torch.nn.Flatten()).eval()
        image = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4) / 48
        exported = torch.export.export(model, (image,))
        pt2 = scratch / 'check.pt2'
        torch.export.save(exported, pt2)
        restored = torch.export.load(pt2)
        loaded = restored.module()
        with torch.no_grad():
            reference = loaded(image).numpy()
        converted = scratch / 'check.onnx'
        torch.onnx.export(restored, (image,), str(converted), dynamo=True,
                          external_data=False, opset_version=18, input_names=['image'], output_names=['scores'])
        session = ort.InferenceSession(str(converted), providers=['CPUExecutionProvider'])
        actual = session.run(None, {'image': image.numpy()})[0]
        if not np.allclose(reference, actual, atol=1e-5, rtol=1e-4):
            raise RuntimeError('The tiny ONNX conversion failed numerical verification.')
        result['checks']['pt2_export_and_load'] = True
        result['checks']['onnx_inference'] = True
        result['checks']['onnx_max_abs_error'] = float(np.abs(reference - actual).max())
        result['checks']['tflite_library_import'] = True
        result['checks']['serial_library_import'] = True
        if frontend_dir:
            frontend = frontend_dir.resolve()
            for relative in ('index.html', 'src/app.js', 'src/api.js', 'src/styles.css'):
                if not (frontend / relative).is_file():
                    raise RuntimeError(f'Built interface file missing: {relative}. Run setup-windows.cmd.')
            result['checks']['built_interface'] = True
        result['status'] = 'passed'
        print('Runtime check passed: PT2 export, real ONNX inference, TFLite/USB imports and interface files.', flush=True)
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        print('Runtime check failed: ' + result['error'], flush=True)
    finally:
        if scratch and scratch.exists():
            if scratch.resolve().parent != base or scratch.is_symlink():
                raise RuntimeError('Refusing to clean an unexpected diagnostic directory.')
            shutil.rmtree(scratch)
        if report_path:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-dir', type=Path)
    parser.add_argument('--report', type=Path)
    options = parser.parse_args()
    raise SystemExit(check(options.frontend_dir, options.report))
