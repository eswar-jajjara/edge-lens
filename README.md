# EdgeLens

EdgeLens is a Windows desktop developer tool for validating image-classifier models and comparing conversion results. It runs its Python engine on your computer and stores experiment history in a local SQLite database.

## Start here

### Requirements

- Windows 10 or 11, 64-bit
- Python 3.12, 64-bit
- Node.js 22.12 or later, 64-bit
- Internet access for the first setup, which downloads the pinned runtime packages

No administrator access, GPU, WSL, Raspberry Pi, or ESP32 is required for local software benchmarks.

### Run EdgeLens

1. Download the source ZIP from [GitHub](https://github.com/eswar-jajjara/edge-lens) and extract it, or clone it:

   ```powershell
   git clone https://github.com/eswar-jajjara/edge-lens.git
   cd edge-lens
   ```

2. Double-click `run-edgelens.cmd`. On first launch it prepares the local Python and Node dependencies, builds the desktop interface, checks the engine, and opens EdgeLens. Use the same launcher for later starts.

For separate setup, diagnostic, and troubleshooting steps, see [Windows teammate setup](docs/TEAM-SETUP.md). EdgeLens is shared as source; the old v0.4.1 portable ZIP is outdated and cannot open newer database schemas.

## Run your first benchmark

1. Open **Overview → Local benchmark**.
2. Select a trusted exported image classifier: `.pt2`, single-file `.onnx`, or `.tflite`. Weights-only `.pth` files need their architecture and must be exported first.
3. Set the model input shape, class count, layout, and preprocessing to match its training/export configuration.
4. Upload a labelled image ZIP. Keep class folder names and label indices aligned with the model output order.
5. Choose a dataset and a conversion format, then start with **Fixed profiles / benchmark existing model** and **No · benchmark on this laptop only**.
6. Read the saved report and layer diagnostics. Export HTML, CSV, or JSON from **Reports & history**.

Use the [developer guide](docs/DEVELOPER-GUIDE.md) for model and dataset details, [manual software tests](docs/MANUAL-SOFTWARE-TESTS.md) for a feature-by-feature walkthrough, and the [reviewer test walkthrough](docs/REVIEWER-TEST-WALKTHROUGH.md) for a reproducible presentation sequence.

## What the results mean

| Label | What it describes |
| --- | --- |
| **MEASURED · host** | Results from the current computer, runtime, and settings |
| **ESTIMATED · Edge Impulse** | Provider analysis for an uploaded model and selected target; not a physical-device benchmark |
| **MEASURED · device** | A matching report captured on real hardware |
| **UNAVAILABLE** | The requested value was not measured or cannot be compared from the available reference |

A Raspberry Pi or ESP32 deployment goal does not simulate that hardware. Physical-device latency and memory stay unavailable until a matching device report is imported or captured. Comparisons against published internet results must retain the source, hardware, dataset, preprocessing, runtime, and measurement conditions; EdgeLens does not claim a universal converter is better.

### Optional Edge Impulse estimates

Connect a project in the separate **Edge Impulse** section, load its supported targets, and select **Yes** when starting a test to authorize analysis of the evaluated ONNX or TFLite artifact. Dataset images are not uploaded. ONNX profiling may involve provider-side conversion; accuracy of that provider-converted model remains unavailable unless independently measured. See [connecting a project](docs/EDGE-IMPULSE-CONNECTION.md) and [ONNX profiling](docs/EDGE-IMPULSE-ONNX.md).

## What EdgeLens includes

- PyTorch export comparison and imported ONNX/TFLite benchmarking
- Accuracy, prediction agreement, output drift, conversion time, host latency summaries, and serialized model size
- Layer-level diagnostics, report history, and HTML/CSV/JSON exports
- FP32 versus static INT8 QDQ experiments and validation-guided deployment search
- Optional TFLite Linux worker reports and ESP32 report import
- Local SQLite management for saved models and datasets

Removing a saved model or dataset from the library removes its SQLite entry; it does not erase the original file or snapshots retained in reports. Deleting completed test history is a separate action.

## Data and security

Models, datasets, database files, and reports stay on the local computer under `%APPDATA%\EdgeLens\data`. GitHub shares the source code only. Teammates connect their own Edge Impulse project keys; never commit API keys, private datasets, model weights, or personal reports.

Only load model exports you trust. PT2 loading can execute serialized code, and the local engine is not a security sandbox. Developer uploads are disabled by default on the standalone API.

## Develop

Use the same Python and Node versions listed above. From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-desktop.ps1
npm run desktop
```

Useful existing checks:

```powershell
npm run check
npm test
cd backend
$env:EDGELENS_TEST_CUSTOM='1'
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

The custom backend suite uses synthetic fixtures; it checks software behavior, not model accuracy on your project. Optional conversion and packaging paths have separate platform requirements. See [architecture](docs/ARCHITECTURE.md), [benchmark methodology](docs/BENCHMARKING.md), [verification notes](docs/VERIFICATION.md), and [deployment](docs/DEPLOYMENT.md).

For contribution workflow and repository layout, read [CONTRIBUTING.md](CONTRIBUTING.md).