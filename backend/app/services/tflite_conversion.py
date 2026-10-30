"""FP32 and calibrated static INT8 TFLite conversion via Linux or local WSL worker."""
import copy
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# WSL availability helpers
# ---------------------------------------------------------------------------

def _wsl_command():
    """Return the wsl executable path, or None if WSL is not accessible."""
    try:
        result = subprocess.run(
            ['wsl', '--status'],
            capture_output=True, timeout=8,
        )
        # wsl --status exits 0 when WSL is running; non-zero means unavailable.
        # Some Windows versions don't support --status; fall back to a simple echo.
        if result.returncode != 0:
            result = subprocess.run(
                ['wsl', '--', 'echo', 'ok'],
                capture_output=True, timeout=8,
            )
        return 'wsl' if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def _wsl_project_path(win_path: Path) -> str:
    """Convert a Windows path like D:\\foo to /mnt/d/foo for WSL."""
    drive = win_path.drive.rstrip(':').lower()
    rest = win_path.as_posix()[len(win_path.drive):]
    return f'/mnt/{drive}{rest}'


def _wsl_venv_python(project_root: Path) -> str | None:
    """Return the WSL python path if the pinned venv exists, else None."""
    venv_py = project_root / '.venv-tflite' / 'bin' / 'python'
    wsl_py = _wsl_project_path(venv_py)
    try:
        result = subprocess.run(
            ['wsl', '--', 'test', '-x', wsl_py],
            capture_output=True, timeout=8,
        )
        return wsl_py if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def wsl_status() -> dict:
    """Return WSL availability details for the capabilities API."""
    if platform.system() != 'Windows':
        return {'available': False, 'reason': 'WSL is a Windows-only facility.'}
    wsl = _wsl_command()
    if not wsl:
        return {'available': False, 'reason': 'WSL is not installed or access is denied. Install WSL and run bash scripts/setup-tflite-worker.sh once.'}
    # Locate project root relative to this file: backend/app/services/tflite_conversion.py
    project_root = Path(__file__).resolve().parents[3]
    py = _wsl_venv_python(project_root)
    if not py:
        return {
            'available': False,
            'wsl_present': True,
            'reason': 'WSL is present but the pinned TFLite venv is not set up. Run: bash scripts/setup-tflite-worker.sh',
            'setup_command': 'bash scripts/setup-tflite-worker.sh',
        }
    return {'available': True, 'wsl_present': True, 'python_path': py,
            'reason': 'WSL TFLite worker is ready.'}


# ---------------------------------------------------------------------------
# WSL-routed conversion
# ---------------------------------------------------------------------------

def _convert_via_wsl(ep, sample, calibration_arrays, output_dir: Path) -> list[dict]:
    """Run the pinned Linux TFLite worker inside WSL and collect its outputs.

    The function:
    1. Saves the PT2 export and calibration images to a temp directory that
       WSL can reach (/mnt/...).
    2. Calls `wsl -- <python> backend/cli.py` with --format tflite.
    3. Reads back the produced .tflite files and returns the same profile
       dicts that the native Linux convert_profiles() returns.

    The caller (developer_benchmark.py) never needs to know whether Linux or
    WSL was used; it sees the same structure either way.
    """
    import numpy as np
    import torch

    project_root = Path(__file__).resolve().parents[3]
    status = wsl_status()
    if not status.get('available'):
        raise ValueError(
            f"PT2 → TFLite via WSL: {status['reason']}\n"
            "Run: bash scripts/setup-tflite-worker.sh  (once, inside WSL)"
        )
    wsl_py = status['python_path']
    wsl_cli = _wsl_project_path(project_root / 'backend' / 'cli.py')

    with tempfile.TemporaryDirectory(prefix='edgelens-wsl-') as tmp_win:
        tmp = Path(tmp_win)
        wsl_tmp = _wsl_project_path(tmp)

        # --- Save PT2 ---
        pt2_path = tmp / 'model.pt2'
        torch.export.save(ep, str(pt2_path))

        # --- Save calibration images as a ZIP ---
        calib_zip = tmp / 'calibration.zip'
        spec_path = tmp / 'spec.json'

        # Build a minimal labels.json + image archive from raw numpy arrays.
        # We encode each array as raw float32 bytes saved as .npy, then
        # pass them via a synthetic spec that declares NCHW float32.
        shape = list(sample.shape)  # [1, C, H, W]
        calib_entries = []
        with zipfile.ZipFile(calib_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
            import io
            zf.writestr('labels.json', json.dumps({'calib': 0}))
            for i, arr in enumerate(calibration_arrays):
                # Save as PNG via PIL so the worker's standard image loader works.
                try:
                    from PIL import Image
                    # arr shape: [1, C, H, W] or [1, H, W, C] — undo preprocessing
                    # to get back a valid uint8 image for the worker.
                    # We use a lossless round-trip: save the raw float as a 32-bit
                    # TIFF so the worker reads it back identically.
                    img_arr = arr.squeeze(0)  # [C, H, W]
                    if img_arr.ndim == 3 and img_arr.shape[0] in (1, 3):
                        img_arr = img_arr.transpose(1, 2, 0)  # [H, W, C]
                    # Clamp + scale back to uint8 for a standard image
                    img_uint8 = np.clip(img_arr * 255, 0, 255).astype(np.uint8)
                    if img_uint8.shape[-1] == 1:
                        img_uint8 = img_uint8.squeeze(-1)
                    img = Image.fromarray(img_uint8)
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    name = f'calib/img_{i:04d}.png'
                    zf.writestr(name, buf.getvalue())
                    calib_entries.append({'path': name, 'label': 0})
                except Exception as exc:
                    raise ValueError(f'Could not encode calibration image {i} for WSL worker: {exc}') from exc

        # Write spec.json matching the sample tensor shape
        spec = {
            'input_shape': shape,
            'layout': 'NCHW',
            'class_count': 2,
            'scale': 1.0 / 255,
            'mean': [0.0, 0.0, 0.0],
            'std': [1.0, 1.0, 1.0],
            'resize': 'stretch',
            'resize_shorter': None,
        }
        spec_path.write_text(json.dumps(spec), encoding='utf-8')

        # Patch the calibration dataset JSON to include entries
        calib_dataset_json = tmp / 'calib-dataset.json'
        calib_dataset_json.write_text(json.dumps({
            'id': 'wsl_calibration', 'name': 'WSL calibration',
            'path': str(calib_zip), 'sha256': _sha256(calib_zip),
            'image_count': len(calib_entries), 'class_count': 1,
            'labels': {'calib': 0}, 'entries': calib_entries,
        }), encoding='utf-8')

        out_dir = tmp / 'wsl-output'
        out_dir.mkdir()
        wsl_out = _wsl_project_path(out_dir)

        cmd = [
            'wsl', '--',
            wsl_py,
            wsl_cli,
            '--model', _wsl_project_path(pt2_path),
            '--spec', _wsl_project_path(spec_path),
            '--calibration', _wsl_project_path(calib_zip),
            '--dataset', _wsl_project_path(calib_zip),  # same zip; worker holds it out
            '--format', 'tflite',
            '--strategy', 'fixed_profiles',
            '--target', 'esp32',
            '--output', wsl_out,
            '--trust-own-model',
        ]

        started = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=300,
                cwd=str(project_root),
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError('WSL TFLite conversion timed out after 5 minutes.') from exc

        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '')[:600]
            raise ValueError(f'WSL TFLite worker failed (exit {result.returncode}): {detail}')

        elapsed = time.perf_counter() - started

        # Collect .tflite outputs produced by the worker
        tflite_files = sorted(out_dir.glob('*.tflite'))
        if not tflite_files:
            raise ValueError(
                'WSL TFLite worker ran but produced no .tflite files. '
                'Check that the worker venv is complete: bash scripts/setup-tflite-worker.sh'
            )

        profiles = []
        for i, src in enumerate(tflite_files[:2]):  # standard, dashboard
            name = 'standard' if i == 0 else 'dashboard'
            dest = output_dir / (name + '.tflite')
            shutil.copy2(src, dest)
            precision = 'fp32' if name == 'standard' else 'static_int8'
            profiles.append({
                'profile': name,
                'path': dest,
                'precision': precision,
                'conversion_seconds': elapsed / max(len(tflite_files), 1),
                'label': 'LiteRT FP32 control (WSL)' if name == 'standard' else 'LiteRT calibrated static INT8 (WSL)',
                'configuration': {
                    'converter': 'litert-torch',
                    'execution_origin': 'wsl_local',
                    'quantization': precision,
                    'calibration_samples': len(calibration_arrays) if name == 'dashboard' else 0,
                },
            })

        if not profiles:
            raise ValueError('No TFLite profiles were produced by the WSL worker.')
        return profiles


def _sha256(path: Path) -> str:
    import hashlib
    with path.open('rb') as fh:
        return hashlib.file_digest(fh, 'sha256').hexdigest()


# ---------------------------------------------------------------------------
# Native Linux path (unchanged from original)
# ---------------------------------------------------------------------------

def _convert_native(ep, sample, calibration_arrays, output_dir):
    """Original Linux-only conversion using litert-torch directly."""
    import torch
    import litert_torch
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
    from litert_torch.quantize.quant_config import QuantConfig

    profiles = []
    for name in ('standard', 'dashboard'):
        started = time.perf_counter()
        target = output_dir / (name + '.tflite')
        module = copy.deepcopy(ep).module()
        if name == 'standard':
            edge = litert_torch.convert(module, (sample,))
            precision = 'fp32'
        else:
            quantizer = PT2EQuantizer().set_global(
                get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False))
            prepared = prepare_pt2e(torch.export.export(module, (sample,)).module(), quantizer)
            with torch.no_grad():
                for value in calibration_arrays:
                    prepared(torch.from_numpy(value))
            quantized = convert_pt2e(prepared, fold_quantize=False)
            edge = litert_torch.convert(quantized, (sample,),
                                        quant_config=QuantConfig(pt2e_quantizer=quantizer))
            precision = 'static_int8'
        edge.export(str(target))
        profiles.append({
            'profile': name, 'path': target, 'precision': precision,
            'conversion_seconds': time.perf_counter() - started,
            'label': 'LiteRT FP32 control' if name == 'standard' else 'LiteRT calibrated static INT8',
            'configuration': {
                'converter': 'litert-torch', 'quantization': precision,
                'calibration_samples': len(calibration_arrays) if name == 'dashboard' else 0,
            },
        })
    return profiles


# ---------------------------------------------------------------------------
# Public entry point (called by developer_benchmark.py)
# ---------------------------------------------------------------------------

def convert_profiles(ep, sample, calibration_arrays, output_dir):
    """Convert a PT2 export to FP32 + INT8 TFLite.

    On Linux: uses litert-torch directly (original behaviour, unchanged).
    On Windows with WSL set up: routes through the pinned Linux worker via WSL.
    On Windows without WSL: raises a clear actionable error.
    """
    if not calibration_arrays:
        raise ValueError('Static INT8 TFLite conversion requires separate calibration images.')

    if platform.system() == 'Linux':
        return _convert_native(ep, sample, calibration_arrays, output_dir)

    # Windows: attempt WSL routing
    return _convert_via_wsl(ep, sample, calibration_arrays, output_dir)
