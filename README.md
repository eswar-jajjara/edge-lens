# EdgeLens — Deployment Validation Tool 0.10.0

A Windows desktop developer tool for validating image-classifier conversion and investigating edge readiness with separately labelled provider estimates. Electron starts a private local Python engine automatically; SQLite stores models, datasets, metrics, layer findings and reports. Model binaries stay on disk with SHA256 metadata in SQLite.

**New in 0.10.0:** readable offline reports with compact tables, aligned numbers,
accuracy/latency/size charts and expandable technical evidence. **Reports &
history → Compare saved results and published references** combines measured
PyTorch/ONNX/TFLite JSON exports with separately labelled internet references.
Follow [the complete testing and final-report pathway](docs/FINAL-COMPARISON-WORKFLOW.md)
and use [the checked reference starter](docs/reference-examples.json).
Published conditions are not assumed identical; unknown values stay unavailable.

**New in 0.9.1:** profile evaluated ONNX files through Edge Impulse’s BYOM workflow. Upload and profiling jobs are saved separately; provider-converted accuracy remains unavailable. See [the ONNX walkthrough](docs/EDGE-IMPULSE-ONNX.md).

## Open it

**From GitHub source:** install 64-bit Python 3.12 and 64-bit Node.js 22.12+, double-click `setup-windows.cmd` once, then `start-desktop.cmd`. Setup installs the tested dependency versions and checks a real tiny PT2/ONNX conversion. `check-windows.cmd` verifies an existing setup. See [the teammate setup guide](docs/TEAM-SETUP.md) for prerequisites, diagnostics and sharing experiments.

**New in 0.8.0:** a separate **Edge Impulse** section, browser sign-in shortcut,
verified project-key connections and project/target selection for each test.
Overview asks **Estimate this test in Edge Impulse? Yes / No**. Reports & history
can delete an individual finished test after confirmation. See
[the connection and testing walkthrough](docs/EDGE-IMPULSE-CONNECTION.md).
Browser sign-in alone does not authorize this tool; account-wide OAuth project
discovery is not configured. Project keys connect their own projects.

**For Phase 2:** use current source, select an uploaded PT2 or FP32 ONNX model, then choose **Optimize deployment · validation search**. Upload separate calibration, validation and test ZIPs. Set a selection objective and accuracy/size/host-latency limits. See [the deployment search guide](docs/DEPLOYMENT-SEARCH.md).

The older [Windows x64 portable ZIP v0.4.1](https://github.com/eswar-jajjara/edge-lens/releases/tag/v0.4.1) predates precision experiments and deployment search. It cannot open newer databases. The source launcher starts the current source even if that older EXE is present. Each laptop keeps its own data under `%APPDATA%/EdgeLens/data`.

**Start with [the developer guide](docs/DEVELOPER-GUIDE.md)** for upload settings, test datasets, conversion comparisons, firmware steps and troubleshooting.

For a feature-by-feature explanation and a complete test sequence without an ESP32, follow [manual software tests](docs/MANUAL-SOFTWARE-TESTS.md).

For your project review, follow [the reviewer test walkthrough](docs/REVIEWER-TEST-WALKTHROUGH.md): every software check, the prepared 500-image ImageNet-1K/MobileNetV2 V2 experiment on this laptop, published references and the evidence folder to present.

## Features

- Upload your own `.pt2`, single-file `.onnx` or `.tflite` image classifier with preprocessing and class-count settings. Supported signatures are fixed batch-one, one image input and one class-score output. Only load trusted exports; this is not an untrusted-model sandbox.
- Compare a PT2 reference against standard ONNX export and an **EdgeLens fidelity-guided conversion strategy**, using separate calibration and held-out datasets. Retain every candidate and the selection rationale; equal/worse results are valid.
- Compare **FP32 ONNX versus static INT8 QDQ** from PT2 or FP32 ONNX uploads. Keep calibration, validation and test datasets separate; save exact settings, observed quantization coverage, per-metric provenance and failed candidates. See [precision experiments and the reproducible demo](docs/PRECISION-EXPERIMENTS.md).
- Search calibration methods and weight granularities, measure preserved ONNX operation drift, test selective FP32 weight exclusions and choose a measured configuration under explicit validation constraints. Inspect Pareto trade-offs, controlled sensitivity results and rejection reasons. FP32 or no feasible candidate are valid outcomes; held-out test data never chooses the winner.
- Benchmark imported ONNX/TFLite models independently. Without PyTorch reference data, conversion loss is explicitly unavailable.
- Measure labelled top-1 accuracy, prediction agreement, output MAE/max error, conversion/search time, mean/median/p95 host latency and serialized size.
- Inspect conservative per-operation mappings, an unoptimized diagnostic graph, per-image predictions, runtime versions, hashes and exact settings. Unmapped operations remain unverified.
- Generate an ESP-IDF package for a compatible small TFLite model; capture matching ESP32 USB reports or import their JSON. Physical device results are separate from host metrics. EdgeLens does not flash boards automatically.
- Connect one or more Edge Impulse projects with session-only keys, load their targets and choose a destination per test. No requires no upload; Yes explicitly authorizes an evaluated ONNX or TFLite upload after a successful test. ONNX uses provider conversion and replaces the project’s BYOM model; converted-model accuracy remains UNAVAILABLE. See [ONNX profiling](docs/EDGE-IMPULSE-ONNX.md). Provider estimates stay separate from host and physical-board measurements. No hardware is needed for this optional step.
- Delete a completed or failed test's local report history and generated artifacts after confirmation. Original uploaded models and datasets stay; active benchmarks cannot be deleted.
- Export complete HTML/CSV/JSON reports. Use `backend/cli.py` for a headless software-tool workflow.

TFLite **inference** works in this Windows engine, including per-tensor int8 inputs/outputs. PT2-to-TFLite **conversion** is verified for the tiny classifier on the pinned Linux worker; local Windows/WSL conversion is not verified. Import its checked worker report into the desktop tool. See [Stages 1–5](docs/STAGES-1-5.md) for the exact stack and limitations. Raspberry Pi remains a deployment goal; its hardware runner is future work. Reports include sampled whole-process RSS; this is not model-only RAM or a guaranteed true peak. No universal converter superiority, exact simulated device timing or energy claim is made.

## Develop

Use 64-bit Node.js 22.12+ and 64-bit Python 3.12 for the tested Windows runtime:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-desktop.ps1
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

The custom integration suite runs tiny real PT2/ONNX, static INT8 QDQ, and FP32/int8 TFLite classifiers using synthetic fixtures. It verifies behavior, not project accuracy. The optional full pretrained MobileNetV2 smoke test uses `EDGELENS_ML_SMOKE=1` and may download official weights.

Packaging copies Python sources because the exporter inspects them. This is an unsigned portable development build, not a signed installer. The [Windows CI workflow](https://github.com/eswar-jajjara/edge-lens/actions/workflows/windows-desktop.yml) checks fresh setup, native desktop startup and a relocated portable runtime for each source commit; see [verification](docs/VERIFICATION.md) for executed results. Use the setup/runtime check on each teammate's laptop. Version 0.9.0 is a source update; no new public portable ZIP is included.

See [architecture and frameworks](docs/ARCHITECTURE.md), [methodology](docs/BENCHMARKING.md), [API](docs/API.md) and [deployment](docs/DEPLOYMENT.md).

## Earlier Phase 3 checkpoint (0.7.0)

See [edge profiling and evidence](docs/EDGE-PROFILING.md) for exact artifact matching, target discovery, candidate-scoped diagnostics and the small 32x32 profiling fixture. The first live TFLite profile succeeded for the ESP-EYE target, with a saved provider response and a VERIFIED link to the same evaluated model bytes. Reports separate host MEASURED, provider ESTIMATED and ESP32 UNAVAILABLE. This synthetic fixture verifies the integration; it is not conversion-superiority evidence. At that checkpoint, a QDQ ONNX probe and validated Linux TFLite INT8 conversion were pending. The Linux conversion has since been verified in 0.9.0; the QDQ provider probe remains pending. Existing saved history remains on SQLite schema 4.


## Review-II objective completion · version 0.9.0

New structural evidence, seven controlled fault checks, pinned Linux FP32/static
INT8 TFLite conversion, portable worker-report import, 100-sample timing defaults
and sampled process RSS are documented in [Stages 1–5](docs/STAGES-1-5.md).
Open **Layer diagnostics → Run seven checks** for a saved self-test, or start a
new supported ONNX benchmark to see mapping reasons and graph connections.
In **Reports & history**, import the inner `edgelens-worker-report.zip` produced
by the Linux worker; Linux timings remain labelled as worker measurements.
Use **Edge Impulse** to request estimates for an evaluated ONNX or TFLite artifact. ONNX provider conversion is tracked separately; its converted-model accuracy is UNAVAILABLE.
Local WSL has not been verified in this session. Physical-device measurements
and arbitrary cross-format intermediate mapping remain unavailable.
