# EdgeLens — Model Conversion Studio 0.4

A Windows desktop developer tool for validating image-classifier conversion and preparing real ESP32 benchmarks. Electron starts a private local Python engine automatically; SQLite stores models, datasets, metrics, layer findings and reports. Model binaries stay on disk with SHA256 metadata in SQLite.

## Open it

Double-click `start-desktop.cmd`, or `release/win-unpacked/EdgeLens.exe`. Keep the whole portable folder together. It contains Python, CPU PyTorch/ONNX, LiteRT inference and serial support. Data is stored separately under `%APPDATA%/EdgeLens/data`.

**Start with [the developer guide](docs/DEVELOPER-GUIDE.md)** for upload settings, test datasets, conversion comparisons, firmware steps and troubleshooting.

## Features

- Upload your own `.pt2`, single-file `.onnx` or `.tflite` image classifier with preprocessing and class-count settings. Supported signatures are fixed batch-one, one image input and one class-score output. Only load trusted exports; this is not an untrusted-model sandbox.
- Compare a PT2 reference against standard ONNX export and an **EdgeLens fidelity-guided conversion strategy**, using separate calibration and held-out datasets. Retain every candidate and the selection rationale; equal/worse results are valid.
- Benchmark imported ONNX/TFLite models independently. Without PyTorch reference data, conversion loss is explicitly unavailable.
- Measure labelled top-1 accuracy, prediction agreement, output MAE/max error, conversion/search time, median/p95 host latency and serialized size.
- Inspect conservative per-operation mappings, an unoptimized diagnostic graph, per-image predictions, runtime versions, hashes and exact settings. Unmapped operations remain unverified.
- Generate an ESP-IDF package for a compatible small TFLite model; capture matching ESP32 USB reports or import their JSON. Physical device results are separate from host metrics. EdgeLens does not flash boards automatically.
- Optionally upload a selected TFLite artifact to Edge Impulse for resource profiling. Explicit user action is required; keys are not persisted and provider estimates stay separate from real-board measurements.
- Export complete HTML/CSV/JSON reports. Use `backend/cli.py` for a headless software-tool workflow.

TFLite **inference** works in this Windows engine, including per-tensor int8 inputs/outputs. PT2-to-TFLite **conversion** requires the optional Linux LiteRT Torch stack and is not runtime-verified here. Raspberry Pi remains a deployment goal; its hardware runner is future work. No universal converter superiority, exact simulated device timing, peak RAM or energy claim is made.

## Develop

Use Node.js 22+ and Python 3.11/3.12 with compatible CPU torch/torchvision packages:

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
backend/.venv/Scripts/python.exe -m pip install -r backend/requirements-ml.txt -r backend/requirements-dev.txt -r backend/requirements-edge.txt
npm install
npm run desktop
```

The standalone API disables custom model execution by default. The desktop's private local session and `scripts/start-backend.ps1` enable it locally. Do not enable developer uploads on a public untrusted server without a separate isolated worker design.

## Test and package

```powershell
npm run check
npm test
cd backend
$env:EDGELENS_TEST_CUSTOM='1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
cd ..
npm run desktop:package
```

The custom integration suite runs tiny real PT2/ONNX/FP32 and int8 TFLite classifiers using synthetic fixtures. It verifies behavior, not project accuracy. The optional full pretrained MobileNetV2 smoke test uses `EDGELENS_ML_SMOKE=1` and may download official weights.

Packaging copies Python sources because the exporter inspects them. This is an unsigned portable development build, not a signed installer. Native Electron startup still needs confirmation outside the restricted agent session; see [verification](docs/VERIFICATION.md).

See [architecture and frameworks](docs/ARCHITECTURE.md), [methodology](docs/BENCHMARKING.md), [API](docs/API.md) and [deployment](docs/DEPLOYMENT.md).
