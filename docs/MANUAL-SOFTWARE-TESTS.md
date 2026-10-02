# EdgeLens 0.9.1: understand and test the software

This guide covers the current desktop tool, including Stages 1–4 and the TFLite/ONNX Stage 5 integration. It excludes physical ESP32 measurements, serial capture and flashing.

## What “Linux Conversion Worker” means

Some PyTorch-to-TFLite conversion dependencies use a separately pinned Linux environment. The Windows app can benchmark TFLite, but its local PyTorch-to-TFLite converter is not verified on Windows.

The workflow is:

```text
Trusted PyTorch export + preprocessing settings + separate images
                              ↓
             Run the conversion/test commands on Linux
                              ↓
        FP32 TFLite + calibrated INT8 TFLite + measurements
                              ↓
                    Worker report ZIP
                              ↓
      Reports & history → Import a verified worker report
                              ↓
      View saved Linux evidence in the Windows desktop tool
```

The panel in Windows **imports that ZIP**. It does not install Linux, start WSL, send your model to a cloud worker, execute the imported PyTorch file or repeat those Linux measurements. It checks artifact hashes, sizes and held-out evaluation identity, copies the artifacts into the local workspace, and adds a saved experiment to SQLite. The trust checkbox acknowledges the worker: hash checks do not prove externally supplied timings are truthful.

Linux and laptop timings remain separate. Original dataset archives are not inside the portable report bundle. Keep them for reproduction. Nothing is uploaded to Edge Impulse merely by importing a worker report.

Local WSL setup remains unverified. The synthetic conversion workflow has been verified on GitHub-hosted Linux. The converter produces FP32 and calibrated INT8 candidates; float operations or I/O may remain, and MCU compatibility is not established by file size alone.

## What every section does

| Section or feature | Purpose | How to use it |
| --- | --- | --- |
| Illustrative demo | Shows fabricated example values without executing models. | Explore the layout; do not use these values as evidence. |
| Local benchmark | Executes models through the private Python engine on your laptop. | Use this mode for every test below. Reconnect if the engine is unavailable. |
| Bring your own classifier | Saves a trusted `.pt2`, single-file `.onnx` or `.tflite` model with its input contract. | Declare shape, layout, classes, preprocessing and trust, then **Save model**. A weights-only `.pth` is not supported directly. |
| Input shape and layout | Describe the tensor the model expects. | NCHW means batch/channels/height/width; NHWC means batch/height/width/channels. Current models need one fixed batch-one image input and one `[1, classes]` output. |
| Pixel scale, means, standard deviations | Reproduce training preprocessing: `(pixel × scale − mean) ÷ std`. | Use the creator's exact settings. The tiny kit uses RGB, `1/255`, means `0,0,0`, std `1,1,1`. The runtime handles supported TFLite quantized I/O; do not invent new preprocessing for INT8. |
| Resize | Matches the model's training image transformation. | Choose stretch or shorter-side resize plus center crop as specified by the model creator. |
| Model presets | Download/use supported ImageNet classifiers as references. | Use the correct ImageNet class IDs and preprocessing, not the two-class synthetic images. |
| Labelled dataset upload | Supplies images and true class indices for accuracy. | Upload class-folder ZIPs with root `labels.json`. Folder names must map to the model's output order. |
| Calibration dataset | Sets INT8 activation ranges or evaluates fidelity configurations. | Keep it disjoint from validation and held-out test images. |
| Validation dataset | Compares configurations and chooses a deployment-search candidate. | Keep it separate from calibration and final test images. |
| Held-out test dataset | Measures the final model's accuracy and output differences. | It must not choose the winner or set quantization ranges. |
| Conversion format | Selects ONNX or TFLite processing. | ONNX works locally. PT2-to-TFLite conversion requires the separate Linux worker; an existing TFLite file can be benchmarked locally. |
| Deployment goal | Records ESP32 or Raspberry Pi as the intended target. | It does not measure or simulate that board. |
| Fixed profiles | Runs the standard/configured comparison or benchmarks an imported existing model. | Use it first. Imported ONNX/TFLite alone has no original PyTorch conversion-loss baseline. |
| Fidelity-guided ONNX conversion | Tries exporter/runtime configurations and selects the closest numerical match on calibration images. | Requires PT2, ONNX output and separate calibration images. A tie can retain the standard configuration. |
| FP32 versus static INT8 | Compares a floating-point ONNX baseline and a calibrated QDQ candidate. | Supply calibration, validation and test ZIPs; select MinMax, Entropy or Percentile and per-channel/per-tensor weights. |
| Deployment validation search | Tries quantization settings and selected operation exclusions under declared constraints. | Choose objective, accuracy-loss/size/host-latency limits and a search budget. Selection uses validation; the final test follows selection. |
| Candidate outcomes | Retains measured, failed and rejected candidates. | Read validation and test results separately. Validation-only candidates can have no final test metric. |
| Pareto chart and constraints | Shows trade-offs and reasons a candidate is feasible or rejected. | A Pareto point is not automatically feasible, selected or superior. Keeping FP32 or finding no feasible candidate can be valid. |
| Operation exclusion experiments | Tests whether leaving selected operations out of quantization recovers validation accuracy or output fidelity. | Compare recovery against the named parent. Positive, zero and negative results are all valid. |
| Benchmark settings | Sets warmups, timed runs, threads and numerical tolerances. | Start with 10 warmups, 100 measured runs, 1 thread, absolute tolerance `0.0001`, relative tolerance `0.001`. |
| Timing spread and process memory | Records raw timing samples, median/p95/spread and sampled whole-process RSS. | Inspect sample count, CPU, threads and available power-plan metadata. RSS includes models, data, runtime and engine state, not model-only RAM or guaranteed peak. |
| Layer numerical evidence | Compares supported tensor boundaries and records MAE/max error. | Filter by candidate, evidence and status; search names; expand unavailable reasons. |
| Structural graph evidence | Inventories operators, shapes and connections and compares supported correspondences. | Read MATCHED, CHANGED and UNAVAILABLE separately from numerical pass/fail. |
| Run seven checks | Introduces known faults in disposable synthetic models to test the diagnostic engine. | Wait for active benchmarks to finish, then run it in **Layer diagnostics**. |
| Reports & history | Keeps experiments, per-image results, provenance and artifacts in the local SQLite workspace. | Open saved tests, inspect reproducibility, download model files, and export HTML/CSV/JSON. Model binaries stay on disk with database metadata. |
| Delete a saved test | Removes its local report/provider records and generated artifacts after confirmation. | Use a disposable run. Uploaded originals, dataset archives, other runs, exports and remote Edge Impulse jobs are unaffected. |
| Linux worker report import | Brings completed worker evidence and exact artifacts into the local tool. | Choose the inner worker ZIP, acknowledge trust and import. |
| Edge Impulse connection | Authorizes this engine to access a project using a session-only Read + Write key. | Browser login alone is insufficient. Add a project key, load its targets and choose a destination. Disconnect/restart removes the session connection. |
| Estimate Yes / No | Records whether a new test should upload its evaluated artifact for provider estimates. | No performs no upload. Yes requires a connected project/target; multiple qualifying artifacts require selection. |
| Edge Impulse profiling | Saves uploads, jobs and sanitized estimates separately from host metrics. | Choose an evaluated artifact, give upload consent, submit once and fetch when processing completes. |
| ONNX provider conversion | Uses the BYOM workflow for the evaluated ONNX file. | It replaces the project's BYOM model. Use a dedicated project and leave it unchanged until profiling completes. Converted-model accuracy stays UNAVAILABLE. QDQ acceptance remains live-unverified. |
| Edge hardware | Holds later physical-device preparation and measurement functions. | For this checklist, only confirm that physical ESP32 evidence remains UNAVAILABLE. Do not flash, capture or import an invented device result. |
| CLI | Runs the same benchmark engine without Electron or an HTTP server. | Use `backend/cli.py` from a source installation; it creates reports and SQLite evidence in the chosen output directory. |

## Files and model settings for the small test

On Eswar's laptop, the prepared kit is:

```text
D:\projects\edge-lens\.cache\manual-test-kit-0.9.1
```

It contains `windows-source.pt2`, its `windows-fp32.onnx` export, `calibration.zip` (100 images), `validation.zip` (100), `test.zip` (300), two Linux-produced TFLite artifacts, a verified Linux worker ZIP, and a deliberately invalid worker ZIP. All images are synthetic red/blue workflow checks; their accuracy is not evidence of real-world quality or converter superiority.

Use these settings for the kit's image classifiers:

| Field | Value |
| --- | --- |
| Classes | `2` |
| Shape | `1,3,8,8` |
| Layout | `NCHW` |
| Pixel scale | `0.00392156862745098` |
| Means | `0,0,0` |
| Standard deviations | `1,1,1` |
| Resize | Resize to input dimensions / stretch |
| Labels | `red: 0`, `blue: 1` |

The Linux TFLite artifacts were converted from their **separate Linux reference**, preserved in the worker report. They must not be presented as conversions of `windows-source.pt2`. The Windows source and ONNX export do share the same conversion lineage.

For teammates, `.cache` files are not in GitHub. From a current source setup, generate their own PT2 and disjoint dataset fixture:

```powershell
backend/.venv/Scripts/python.exe scripts/create-precision-demo.py --output .cache/manual-software-fixture
```

Use `colour-classifier.pt2` and that directory's `spec.json` and three ZIPs. Create an ONNX file by running Test 2 and downloading the standard artifact. Obtain TFLite files/report through Test 7 or 8. The generator refuses to overwrite an existing directory; choose a fresh name if needed.

## Test in this order

Mark each item PASS, FAIL or BLOCKED and keep the run ID/exported JSON. All tests use **Local benchmark**, not illustrative demo. Close an older window and open the current `release\win-unpacked\EdgeLens.exe` first.

### 1. Model and dataset library

- [ ] In **Overview → Bring your own classifier**, upload `windows-source.pt2`, give it a recognisable name, enter the settings above, acknowledge trust and **Save model**.
- [ ] Upload `calibration.zip`, `validation.zip` and `test.zip` separately using **Upload dataset**.
- [ ] Confirm the saved model is in the model selector and all three datasets are selectable with their intended counts.

The test checks upload/settings/storage. It does not benchmark until you click Run.

### 2. Fresh PT2 → ONNX comparison, structure and timing

- [ ] Select the uploaded PT2, ONNX, **Fixed profiles / benchmark existing model**, and `test.zip` in **Labelled image dataset**.
- [ ] Choose either deployment goal. Set **Estimate this test? No**.
- [ ] Set 10 warmups, 100 measured runs, 1 thread and the default tolerances. Click **Run local benchmark**.
- [ ] Verify completion, recorded test count 300, accuracy, output MAE/max error, conversion duration and model size. Equal accuracy is a valid result.
- [ ] Expand **Timing spread and process memory**. Check 100 raw samples per timed profile in JSON, cycling up to 16 shared inputs, median/p95/spread, CPU, configured threads, available Windows power plan and RSS status.
- [ ] Open **Layer diagnostics → Structural graph evidence**. Expand the inventory and comparison. Inspect shapes, operations and parents/connection fields. Match coverage depends on supported mapping rules; CHANGED or UNAVAILABLE is not automatically a failed classifier.
- [ ] Try numerical/inventory views, each candidate, status filters and layer-name search. Expand a “Not compared” explanation if present. Read the first observed divergence if a measured drift exists; no divergence is valid for a faithful model.

For a repeatability check, run the same settings again after closing busy applications. Compare raw spread before interpreting a small median difference. This does not establish statistical significance or device speed.

### 3. Fidelity-guided conversion

- [ ] Keep the PT2 model, ONNX and `test.zip`; select **EdgeLens fidelity-guided ONNX conversion** and `calibration.zip`.
- [ ] Keep estimate **No** and run.
- [ ] In **Reports & history → Selection and reproducibility**, inspect tried configurations, selected configuration, calibration rationale, held-out metrics and artifacts.

Pass means the selection rationale and final measured evidence are recorded. Selecting the standard configuration, equal accuracy or a worse held-out result is possible; the test must not require an invented improvement.

### 4. FP32 versus INT8 precision experiment

- [ ] Select **Compare FP32 vs static INT8 · ONNX**; choose calibration, validation and test ZIPs in their three fields.
- [ ] Choose MinMax and per-channel weights initially; estimate **No**, then run.
- [ ] Inspect **Candidate outcomes** for FP32 and INT8 statuses, validation/test accuracy, agreement, serialized size, timing, output error, quantization settings and QDQ inventory.
- [ ] View candidate-specific layer diagnostics. Q/DQ helpers in inventory are not proof of numerically compared boundaries or integer hardware execution.

A tiny INT8 file can be larger or slower. A failed INT8 candidate should retain its actual reason rather than fabricated measurements.

### 5. Deployment search and constraints

- [ ] Use the same three disjoint ZIPs and choose **Optimize deployment · validation search**.
- [ ] Start with **Smallest model**, maximum accuracy loss `1` percentage point, blank size/latency limits and default search budget. Estimate **No**.
- [ ] Verify candidate outcomes, constraint/rejection reasons, Pareto chart, operation-exclusion controls, selected candidate or explicit no-selection reason.
- [ ] Confirm validation evidence determines selection. Final held-out results must be separate; validation-only candidates may have no test metric.

Optional checks: rerun with **Show trade-offs · no selection**; it should not force a winner. A restrictive size limit can give “no feasible candidate,” which is a valid outcome.

### 6. Seven diagnostic control/fault checks

- [ ] Wait for all benchmarks to finish. Open **Layer diagnostics → Run seven checks**.
- [ ] Verify all seven expected/detected results pass: unchanged control, changed weights, replaced operation, rewired connection, wrong shape, removed operation and wrong preprocessing.
- [ ] Invalid graphs must be rejected; measurable altered models must show the expected divergence. The unchanged control should not become a false error.
- [ ] Open the saved self-test from history and export its report.

This proves the specified synthetic controls work. It is not a claim that every possible model fault is diagnosed uniquely.

### 7. Linux report import and integrity rejection

- [ ] **Reports & history → Import a verified worker report**: select `linux-worker-report.zip`, acknowledge the trusted worker and import.
- [ ] Verify a completed saved run appears with PyTorch, FP32 TFLite and INT8 TFLite evidence. Its location/timings remain labelled Linux worker evidence, not fresh laptop measurements.
- [ ] Inspect TFLite operator/tensor/quantization inventory. Arbitrary PyTorch/TFLite intermediate numerical layer alignment remains UNAVAILABLE.
- [ ] Download its artifacts and export HTML/JSON/CSV. The imported artifact hashes should match the recorded worker hashes.
- [ ] Select `deliberately-invalid-worker-report.zip` and attempt import. It should be rejected because one model's bytes changed. There should be no completed imported run from this invalid bundle.

This tests the importer. To test a **new conversion**, use Test 8.

### 8. Run a fresh Linux conversion without an ESP32

You can use GitHub's Linux workflow rather than configuring WSL today:

- [ ] Open this repository's **Actions → Verify Linux TFLite worker → Run workflow**, choose `main` and start the workflow.
- [ ] Wait for actual conversion/evaluation and hash verification to succeed. A failure is a failure; do not use its partial output as completed evidence.
- [ ] Download the `tflite-worker-evidence` artifact and extract the outer download.
- [ ] Import the **inner** `worker-results/edgelens-worker-report.zip` through the Windows worker panel.
- [ ] Verify this is a newly generated run with recorded Linux versions, exact artifacts, separate calibration/test evidence, real timing samples and model hashes.

This workflow only generates and tests the synthetic example. It does not accept your custom model through the desktop UI, benchmark a cloud ESP32 or automatically dispatch jobs from the import panel.

For a custom model, run the pinned Linux worker separately with its matched PT2 export, preprocessing spec and disjoint images. See [Stages 1–5](STAGES-1-5.md) for Linux setup/CLI commands. Do not install the Linux conversion stack inside the Windows engine.

### 9. Benchmark existing ONNX and TFLite files on Windows

- [ ] Upload `windows-fp32.onnx` with the kit's settings, choose ONNX, fixed profiles and `test.zip`; estimate **No**, then run.
- [ ] Upload `linux-fp32.tflite`, choose TFLite, fixed profiles and `test.zip`; run again.
- [ ] Repeat with `linux-int8.tflite`. Use the same declared float preprocessing; the supported quantized input/output adapter handles tensor scale/zero-point conversion.
- [ ] Each should produce a fresh **laptop** accuracy/runtime report and complete artifact/dataset identity. Original PyTorch conversion loss remains UNAVAILABLE for a standalone imported file.

Do not treat the Linux-worker timing and the fresh Windows timing as measurements from the same CPU.

### 10. Edge Impulse TFLite and ONNX profiles — no hardware needed

Internet, a project and a Read + Write API key are needed. Keep the key inside EdgeLens's dedicated field. These integrations must be marked BLOCKED/unverified if you cannot connect or the provider rejects the model.

- [ ] Open **Edge Impulse**, connect a dedicated project and **Load supported targets**. Browser sign-in alone should not count as connected.
- [ ] Select the imported Linux test, choose one evaluated TFLite artifact and a returned provider target, check consent and click **Upload & request estimate** once.
- [ ] Wait or use **Fetch completed estimate** later. Verify a real job/response is stored, with artifact hash, target, project, date and sanitized response. Resource numbers stay ESTIMATED; physical ESP32 evidence stays UNAVAILABLE. TFLite evaluation identity should match the exact uploaded artifact.
- [ ] Select the fresh standalone ONNX test from Test 9 and its evaluated FP32 artifact. Read the replacement notice: the BYOM upload replaces this project's model. Leave the project unchanged until profiling completes.
- [ ] Confirm upload, submit once and fetch later. Verify separate upload/profile job IDs, ONNX format, original artifact hash and provider response.
- [ ] In the report, the original ONNX evaluation link can be VERIFIED, but **provider-converted accuracy link must remain UNAVAILABLE**. Never attach original ONNX accuracy to the provider's different converted model.
- [ ] Optional: try an evaluated INT8 QDQ candidate from Test 4. Its acceptance is not yet live-verified; a descriptive provider rejection is an acceptable compatibility result, not evidence of a successful profile.
- [ ] Disconnect the project. Attempting another profile without reconnecting should fail. Reconnect the same project before fetching a saved pending job after restarting.

Do not resubmit automatically after a POST timeout. Check Studio first: a lost receipt does not prove the upload failed. Local job tracking and sanitized completed responses preserve evidence when possible.

### 11. New-test Yes/No behavior

- [ ] Test 2 used **No**; confirm that run has no provider job from that benchmark.
- [ ] With a project/target connected, rerun the standalone ONNX benchmark from Test 9 and choose **Yes**, selecting its destination.
- [ ] After local evaluation, a single qualifying artifact should be submitted once. If several distinct artifacts qualify, the tool should ask for a final artifact selection in **Edge Impulse**.
- [ ] Verify local results are saved even if provider processing fails or is still pending. A Yes preference alone is not a completed estimate.

### 12. Exports, artifacts, SQLite history and deletion

- [ ] Export HTML, JSON and CSV from a measured run. Open HTML and compare its run ID, sample count and metrics to the tool. JSON includes full provenance/raw samples; CSV is useful for analysis.
- [ ] Download a model artifact. Its recorded hash identifies the file you evaluated; keep reports with the artifact for reproduction.
- [ ] Close/reopen the current app and open the same run from history. Local results should remain; project credentials should require reconnecting.
- [ ] Generate a **second disposable self-test**. Export it, then use its history Delete action. Try **Keep test** first; it should remain. Reopen the dialog and confirm deletion of that disposable run only.
- [ ] Verify that run is gone but the first self-test, uploaded original models, datasets, exported files and other tests remain. Do not delete your main comparison evidence for this check.
- [ ] Inspect **Edge hardware** and report evidence: actual ESP32 timing/memory must stay UNAVAILABLE. Do not flash, capture, or invent a device observation.

## Optional command-line check

With the source Python environment installed, run from the repository root, using a fresh output directory:

```powershell
backend/.venv/Scripts/python.exe backend/cli.py --model .cache/manual-test-kit-0.9.1/windows-source.pt2 --spec .cache/manual-test-kit-0.9.1/spec.json --dataset .cache/manual-test-kit-0.9.1/test.zip --format onnx --strategy fixed_profiles --target raspberry_pi --warmup-runs 10 --measured-runs 100 --threads 1 --output .cache/my-manual-cli-check --trust-own-model
```

Expect a new SQLite experiment, model artifacts and `runs/<run-id>/report.html`, `.csv` and `.json`. Teammates should use their generated fixture paths instead. This does not open Electron or profile an edge device.

## Read the outcomes honestly

- **MEASURED:** actual local or worker execution on the recorded host.
- **ESTIMATED:** provider resource/timing analysis.
- **REFERENCE:** external creator figures, which need comparable datasets, preprocessing and hardware before direct comparison.
- **UNAVAILABLE:** missing measurements or unsupported correspondence; never substitute zero.
- **Within tolerance:** a compared tensor passed the configured absolute/relative check.
- **Needs review:** a measured difference exceeds tolerance; a final prediction can still agree.
- **Not compared:** no supported numerical comparison, with a recorded mapping/capture reason. It is not a count of incorrect layers.
- **MATCHED / CHANGED / UNAVAILABLE:** structural correspondence status, separate from numerical fidelity.

These tests verify the software workflow and evidence handling. Real ImageNet accuracy claims require your actual model, exact creator preprocessing/class order and representative held-out images. Neither a synthetic 100% score nor a small timing difference establishes converter superiority.
