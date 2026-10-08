# Test EdgeLens and prepare your reviewer evidence

Use EdgeLens 0.10.0 or newer. First check the software with the tiny synthetic
classifier, then measure a real pretrained classifier. Keep those two sets of
evidence separate. No ESP32 is required for the software checks or provider
estimates. Do not claim actual board latency without a physical measurement.

## 1. Open the current desktop tool

On Eswar's laptop, close older EdgeLens windows and double-click:

```text
D:\projects\edge-lens\start-desktop.cmd
```

The rebuilt local portable app is also at
`D:\projects\edge-lens\release\win-unpacked\EdgeLens.exe`.
Use the source launcher when updating code. The public v0.4.1 release is older
and does not contain these features.

For a teammate, download/extract current GitHub source, install Python 3.12 x64
and Node.js 22.12+ x64, run `setup-windows.cmd` once, then `start-desktop.cmd`.
See [Windows setup](TEAM-SETUP.md). Each laptop keeps its own SQLite data.

Select **Local benchmark**. A demo marked **DEMO / NOT MEASURED** cannot supply
reviewer measurements. Answer **Estimate this test in Edge Impulse? No** initially.
Run one benchmark at a time. Export completed evidence before trying deletion.

## 2. Check every software feature with the small kit

The prepared local kit is:

```text
D:\projects\edge-lens\.cache\manual-test-kit-0.9.1
```

Upload `windows-source.pt2` under **Overview → Bring your own classifier**:

| Setting | Value |
| --- | --- |
| Number of classes | `2` |
| Input shape | `1,3,8,8` |
| Layout | `NCHW` |
| Pixel scale | `0.00392156862745098` |
| Means | `0,0,0` |
| Standard deviations | `1,1,1` |
| Image resize | Resize to input dimensions |

Confirm the trusted source and save the model. Upload `calibration.zip` (100),
`validation.zip` (100) and `test.zip` (300) separately. Use the ZIPs as uploaded;
do not extract them for EdgeLens's file picker.

Teammates can generate their own kit from the repository in PowerShell:

```powershell
.\backend\.venv\Scripts\python.exe scripts/create-precision-demo.py --output .cache/reviewer-software-fixture
```

Use its `colour-classifier.pt2`, `spec.json` and three ZIPs. The output directory
must be new. Cache files are not shared by GitHub. Download the standard ONNX
after check 2; obtain Linux TFLite files using check 8.

Perform these checks in order. The [complete checklist](MANUAL-SOFTWARE-TESTS.md)
gives individual clicks, negative checks and expected results for each one.

| Check | What to do | Evidence to save / pass condition |
| --- | --- | --- |
| 1. Upload/library | Save the PT2 and three ZIPs with the settings above. | Model appears in selector; image counts are 100/100/300. |
| 2. Standard conversion | PT2 → ONNX, **Fixed profiles / benchmark existing model**, test ZIP, 10 warm-ups, 100 measured runs, 1 thread, estimate No. | Completed PyTorch/standard ONNX metrics, 300 test images, model hashes, conversion time, raw timings and exported artifact. |
| 3. Fidelity selection | Same PT2/test; choose **EdgeLens fidelity-guided ONNX conversion**, calibration ZIP. | Tried configurations, frozen selection rationale and final held-out measurements. Equal accuracy or the standard winner is valid. |
| 4. Precision | **Compare FP32 vs static INT8 · ONNX**, all three ZIPs, MinMax/per-channel initially. | Candidate status, validation/test accuracy, size, numerical error, QDQ inventory; an unsupported candidate retains its failure reason. |
| 5. Deployment search | **Optimize deployment · validation search**, all three ZIPs; Smallest model, maximum loss 1 pp, size/latency limits blank. | Validation-based choice, constraints/rejections, Pareto trade-offs, sensitivity evidence. FP32 retention or no feasible candidate can be correct. |
| 6. Diagnostics | Examine candidate-specific numerical and structural views, filters/search; then **Layer diagnostics → Run seven checks** while idle. | Matched/unmapped explanations, graph operations/shapes/connections, and all seven control/fault expectations. Save the self-test report. |
| 7. Worker import | In history import `linux-worker-report.zip`; then try `deliberately-invalid-worker-report.zip`. | Valid report saves with Linux origin; changed-artifact bundle is rejected. |
| 8. New Linux conversion | GitHub **Actions → Verify Linux TFLite worker → Run workflow → main**. Download `tflite-worker-evidence`, extract the outer ZIP, import the inner `worker-results/edgelens-worker-report.zip`. | New completed synthetic conversion and exact FP32/INT8 TFLite artifacts. This workflow does not convert your uploaded MobileNetV2. |
| 9. Existing formats | Upload `windows-fp32.onnx`, `linux-fp32.tflite`, `linux-int8.tflite` individually. Use their actual input shape/layout, the kit's scale/means/stds and test ZIP; fixed profiles. | Fresh Windows accuracy/timing/size and hashes for each. A standalone file has no original PyTorch conversion-loss claim. |
| 10. Edge Impulse | Separate **Edge Impulse** section; connect project, load targets, select an evaluated TFLite or FP32 ONNX, consent, submit once, fetch completed estimate. | Real sanitized response, target/project/date and uploaded hash. Mark blocked if rejected; estimates stay ESTIMATED. |
| 11. Estimate Yes/No | Confirm No created no provider job. With project/target connected, rerun the standalone ONNX check with Yes. | Local results survive provider failure; one qualifying artifact submits once; multiple artifacts require explicit final selection. |
| 12. Persistence/deletion | Export HTML/JSON/CSV/artifact; restart; reopen the run. Create a second disposable self-test; try Keep test, then confirm deletion of that second run. | Saved results persist; only the disposable test's history/artifacts are deleted; uploaded originals remain. |
| 13. Report/comparison | Compare measured JSON exports plus `docs/reference-examples.json`; prepare and save HTML/CSV/JSON. | Charts, units and numbers agree with original exports; internet figures remain REFERENCE; missing data remains UNAVAILABLE. |

**Within tolerance** means a mapped numerical comparison passed the specified
threshold. **Not compared** means no supported numerical mapping/capture was
available; read the reason. It does not mean the layer passed or failed.
Structural CHANGED can represent a legal conversion; inspect numerical evidence.
General PT2/TFLite intermediate numerical alignment remains unavailable.

The Linux and Windows synthetic PT2 files are separate source models. Preserve
that distinction when showing their converted artifacts. A synthetic 100% score
checks workflow execution; it is not evidence of real image-classification quality.

Optional offline hardware check: use **Edge hardware → Prepare & save ZIP** on
a compatible small TFLite artifact, inspect the instructions and matching hashes.
An unsupported operation should produce a clear rejection. Do not flash or
capture a USB report for this software-only review. Raspberry Pi hardware running
is future work; selecting a goal does not change the location of laptop timing.

Optional command-line check, from `D:\projects\edge-lens` in PowerShell:

```powershell
.\backend\.venv\Scripts\python.exe backend/cli.py --model .cache/manual-test-kit-0.9.1/windows-source.pt2 --spec .cache/manual-test-kit-0.9.1/spec.json --dataset .cache/manual-test-kit-0.9.1/test.zip --format onnx --strategy fixed_profiles --target raspberry_pi --warmup-runs 10 --measured-runs 100 --threads 1 --output .cache/my-reviewer-cli-check --trust-own-model
```

CLI storage is in the requested output directory, not automatically in desktop
history. Use a new output directory for a separate experiment. Follow the printed
path to its HTML/JSON/CSV. Mark each check PASS, FAIL or BLOCKED, record the date,
run ID and export filenames, and save an error description for failures.

## 3. Use the real dataset already prepared on this laptop

You do not need another download to start. The ready directory is:

```text
D:\projects\edge-lens\.cache\imagenet-mobilenet-v2
```

Its preparation manifest identifies **ILSVRC2012 / ImageNet-1K validation** as
the image source and **MobileNet_V2_Weights.IMAGENET1K_V2** as the weights.
The `v2` in this folder/model name describes the model weights; it does not make
this dataset ImageNetV2. Use manifest provenance rather than inferring a dataset
from a filename. It records source archive/checkpoint hashes and seed 20261002.

| File | Use |
| --- | --- |
| `mobilenet-v2-imagenet-v2.pt2` | Trusted 1,000-class PyTorch export |
| `calibration.zip` | 100 images: calibration only |
| `validation.zip` | 100 different images: selection only |
| `test.zip` | 500 different images: final held-out measurement |
| `spec.json` | Exact upload settings |
| `preparation-manifest.json` | Original sources, hashes, transforms and selected image records |

Upload the PT2 with classes `1000`, shape `1,3,224,224`, layout `NCHW`, scale
`0.00392156862745098`, means `0.485,0.456,0.406`, stds `0.229,0.224,0.225`.
Choose **Resize to input dimensions** for THESE prepared archives. They contain
lossless 224×224 PNGs already resized/cropped using the official V2 spatial
transform; applying another shorter-side crop would transform them twice.
Use the uploaded classifier path rather than the preset that automatically
applies transforms again.

For newly supplied original JPEGs, V2 instead requires shorter-side resize 232
and centre crop 224 before normalization. V1 uses 256 and has different weights.
The [official TorchVision table](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)
lists V2 top-1 **72.154%** and V1 **71.878%** on ImageNet-1K. Add V2 as your exact
creator reference. Those are full-evaluation figures, not target values to force
for this 500-image subset. One changed prediction here moves accuracy by 0.2 pp.

The [official ImageNet download page](https://www.image-net.org/download.php)
describes 50,000 validation images. Use validation images with their ground-truth
labels, not the unlabelled challenge test set. If your teammate prepares a new
subset, preserve class order 0–999, selection seed, transforms and three disjoint
splits. The tool requires class-folder ZIPs with one root `labels.json`; see the
[developer guide](DEVELOPER-GUIDE.md). Current limits are 1,000 images, 50 MiB ZIP
and 512 MiB prepared input per archive; at 224×224 RGB the memory limit also
rules out 1,000 images in one run. The provided 500-image test fits these limits.

[ImageNetV2](https://github.com/modestyachts/ImageNetV2) is a separate dataset.
It can be another generalization experiment, but it does not reproduce the
ImageNet-1K reference score. Do not mix these dataset names in your presentation.

## 4. Run the real experiment in four configurations

Use the ready MobileNetV2 PT2 and the SAME three ZIPs throughout:

| Experiment | Strategy / required data | Main question |
| --- | --- | --- |
| A. Standard | Fixed profiles, held-out test | Does standard ONNX preserve PyTorch outputs/accuracy? |
| B. Fidelity | Fidelity-guided ONNX, calibration + test | Which declared configuration has the best calibration fidelity, and how does it behave on held-out data? |
| C. Precision | FP32 vs static INT8, calibration + validation + test | How much size/timing changes, and what accuracy is lost or retained? |
| D. Deployment | Validation search, all three splits, predeclared loss/size/timing constraints | Can selection find an acceptable candidate or correctly retain/reject one? |

Start with 10 warm-ups, 100 measured invocations and 1 CPU thread. For your final
timing report, use 500 measured invocations and at least three independent
complete repetitions of each configuration if practical. Keep the laptop plugged
in, power plan and settings consistent, and other workloads quiet. Record the
CPU, OS, runtime versions, threads, median, p95, SD and between-run range.
The invocations are timing samples cycling shared inputs; they are not extra
labelled images or extra accuracy observations.

Name exports `A-standard-1.json`, `B-fidelity-1.json`, etc., and save matching
HTML/CSV and artifacts. Four strategies × three repetitions fit the comparison
panel's 12-report limit. Predeclare selection constraints before reviewing final
test results; do not keep changing them to obtain a favourable test score.

Collect correct/total and top-1 accuracy, agreement versus PyTorch, MAE/max error,
median/p95 latency, exact model bytes, conversion/search seconds, selected settings
and rejection reasons. RSS is sampled whole-engine process memory, not model-only
RAM or guaranteed peak RAM. Conversion/search time and inference latency measure
different things. Include equal/worse results and failed candidates.

## 5. Add a real TFLite comparison after its conversion succeeds

The local kit tests TFLite import/inference. It does NOT establish a successful
conversion of this ImageNet MobileNetV2. Custom PT2 → TFLite uses the separate
pinned Linux worker; local WSL operation and this larger model conversion still
require validation. See [Linux setup and commands](STAGES-1-5.md).

In a working Linux environment, run `bash scripts/setup-tflite-worker.sh` and
export the exact V2 architecture/checkpoint with that worker's matched PyTorch
version. The local ready PT2 was exported with a newer Windows PyTorch version
and may not load in the older pinned worker. Keep the same weight checkpoint
hash and record the new export hash/version. Copy the ready calibration and test
ZIPs to `reviewer-input`, and the matched PT2 plus its preprocessing spec there.
The spec still uses stretch for these already-cropped PNGs.

Then, from the Linux repository root:

```bash
.venv-tflite/bin/python backend/cli.py \
  --model reviewer-input/mobilenet-v2.pt2 --spec reviewer-input/spec.json \
  --calibration reviewer-input/calibration.zip --dataset reviewer-input/test.zip \
  --format tflite --strategy fixed_profiles --target raspberry_pi \
  --warmup-runs 10 --measured-runs 500 --threads 1 \
  --output reviewer-tflite-results --bundle-output reviewer-tflite-report.zip \
  --trust-own-model
```

If unsupported, save the actual failure and mark this comparison BLOCKED.
If successful, import the inner report ZIP in Windows and download its FP32 and
INT8 TFLite artifacts. Upload each back to Windows and benchmark on the same
500 test images and timing settings, using its actual input shape/layout.
Repeat three times. Linux timings remain separate. INT8 does not automatically
mean all operators/I/O are integer or that the model fits ESP32.

## 6. Add optional provider evidence without a board

Use the separate **Edge Impulse** section with project `1126810` or another
dedicated project and its Read + Write key. Browser sign-in alone does not connect
the tool or enumerate all account projects; connected project keys supply access.
Load targets, choose an already-evaluated artifact, confirm consent, submit once,
then fetch the completed estimate. Reconnect after restarting the app.

For TFLite, verify the uploaded hash matches that exact artifact's held-out
evaluation. For ONNX, provider conversion replaces that project's BYOM model;
do not alter it during processing. Its converted-model accuracy remains
UNAVAILABLE. QDQ acceptance is still a live experiment, not a promised feature.
Save real provider receipts or record BLOCKED; do not reuse historical estimates
for a different model. No new real upload is performed by following this guide
until you explicitly select it in the tool.

## 7. Assemble the report for your reviewer

Open **Reports & history → Compare saved results and published references**.
Choose up to 12 measured original report JSONs and `docs/reference-examples.json`.
Use a title such as **MobileNetV2 V2 — ImageNet-1K 500-image held-out comparison**.
Prepare comparison, then save HTML, JSON and CSV. Use separate comparison batches
if TFLite repetitions take the total beyond 12. Check original run IDs, matching
dataset/preprocessing identities, artifact hashes and Windows/Linux origins.
The table cannot infer that unrelated models with matching input contracts share
weights or conversion lineage.

Published ONNX Model Zoo FP32 **69.48%** and QDQ **67.40%** describe a different
MXNet-trained lineage; see its [primary model table](https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/README.md).
Use those as REFERENCE context, not as proof your TorchVision converter improves
accuracy. No comparable laptop median latency is supplied by those reference rows.
For a direct converter comparison, reproduce both methods locally from the same
source weights, labelled test set, preprocessing and host settings. A runtime
speed difference between PyTorch and ONNX is not by itself a better exporter.
The TFLite starter row has no numeric benchmark; supply an exact creator source
later or leave those fields unavailable.

Keep a private reviewer folder containing:

```text
EdgeLens-review-evidence/
  01-method-and-sources.txt
  02-feature-checks.csv
  03-run-reports/             HTML + JSON + CSV, including repetitions
  04-model-artifacts/         exact evaluated files + recorded hashes
  05-final-comparison/        comparison HTML + JSON + CSV
  06-provider-evidence/       sanitized receipt, or BLOCKED explanation
  07-screenshots/             completion, candidate choices, diagnostics, faults
```

Include `preparation-manifest.json` and the version/commit you tested. Keep
licensed source images/private artifacts outside the public repository.
Open the comparison HTML, expand useful evidence, then Print → Save as PDF →
Landscape; inspect the preview. This is browser printing, not a native PDF export.

| Claim | Evidence needed |
| --- | --- |
| The tool works | Completed real model run, hashes/true-label counts, saved artifacts, restart persistence, feature checks and fault-control evidence |
| A candidate preserved/improved accuracy | Same source weights, preprocessing and held-out images; correct/total and delta in percentage points |
| A candidate is smaller | Actual bytes of the measured artifacts; specify format/precision |
| A candidate runs faster here | Same host/settings, raw samples, median/p95/spread, independent repetitions |
| The tool explains conversion issues | Conservative mappings with reasons and observed fault/control outcomes |
| An edge target may fit | Exact-artifact provider receipt labelled ESTIMATED; suitability is not physical execution |
| Actual ESP32 speed/RAM | UNAVAILABLE until a real matching board measurement exists |

In the presentation, separate **MEASURED laptop**, **ESTIMATED provider**,
**REFERENCE published**, and **UNAVAILABLE** evidence. Describe improvements,
ties, regressions and gaps using actual results. A defensible conclusion can be:
"EdgeLens combines conversion validation, diagnostics, constraint-based candidate
selection and reproducible reports. On this model and dataset, we observed the
following accuracy, size and latency trade-offs." Fill in measured numbers;
do not promise universal superiority or change scores to match a reference.
