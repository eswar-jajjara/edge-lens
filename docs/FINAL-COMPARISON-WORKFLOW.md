# Test EdgeLens and prepare your final comparison report

First verify the software with the small synthetic test kit. Then measure a real
classifier with real held-out labels. Finally combine those exported results
with separately labelled published references. This guide covers PyTorch,
ONNX Runtime and LiteRT/TFLite; ONNX is a model format, not a training framework.

## 1. See the improved reports

Close an older EdgeLens window and start the current source with
`start-desktop.cmd`, or use the updated local portable folder. Open **Reports &
history**, select a measured saved run and choose **Save HTML report**. Open the
saved HTML in a browser; it works offline.

The report now has 16 px body text, 14 px tables, 24 px section headings, aligned
numeric columns, labelled units, a compact metric table and accuracy/median
latency/model-size charts. Detailed layers, predictions, raw timings, hashes and
provider responses are retained in expandable panels. No recorded measurement
is changed by the new layout. Missing values display UNAVAILABLE, including in
charts. A rounded zero at three decimals is not a proven exact zero; JSON retains
the original precision.

The default printable summary uses A4 landscape. Open panels you need, choose
browser **Print → Save as PDF → Landscape → 100% scale**, inspect the preview and
save. The tool generates HTML/CSV/JSON, not a native PDF. CSV and JSON retain the
full numerical evidence; HTML is the readable presentation.

## 2. Test every implemented feature first

Follow [Manual software tests](MANUAL-SOFTWARE-TESTS.md), Tests 1–12, in order. They
cover uploads, preprocessing, datasets, fixed conversion, fidelity search,
FP32/INT8, deployment search, structural/numerical diagnostics, seven controlled
faults, worker import/integrity rejection, a fresh Linux worker, standalone
ONNX/TFLite, Edge Impulse, Yes/No, exports, reopen/history and disposable deletion.
Keep the run ID, JSON and outcome PASS / FAIL / BLOCKED for every test. Use **Local
benchmark**, never illustrative demo, for evidence. Physical ESP32 flashing and
USB capture require real hardware and remain a later test.

On Eswar's laptop, the small kit is
`D:\projects\edge-lens\.cache\manual-test-kit-0.9.1`. Stage 5's prechecked report
bundles are in `.cache\stage5-live-ready`. Teammates generate their own fixture
as described in the manual guide; cache/model/dataset files are not in GitHub.

## 3. Choose one real model and freeze the experiment

Use your existing **TorchVision MobileNetV2** experiment first. Record the exact
weight enum, creator URL, model hash, export version and input contract. A
weights-only `.pth` needs the matching architecture before you export trusted
`.pt2`; do not upload it as if it were a complete classifier. The existing tool
accepts `.pt2`, single-file `.onnx` and `.tflite` with one fixed batch-one image
input and one class-score output.

Choose either V1 or V2 deliberately. The [official model page](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)
reports 71.878% for V1 and 72.154% for V2 on its ImageNet evaluation. V1 resizes the
shorter side to 256, V2 to 232; both centre-crop to 224. DEFAULT identifies V2.
These are **REFERENCE** scores, not predictions for your subset. RGB, class
indices and normalization must match the selected weights. For V1: shape
`1,3,224,224`, NCHW, scale `1/255`, means `0.485,0.456,0.406`, standard deviations
`0.229,0.224,0.225`, bilinear shorter-side resize 256 and centre crop 224.

Prepare disjoint **calibration**, **validation** and **test** ZIPs with class
folder names and root `labels.json` matching all 1,000 ImageNet output indices.
Do not map a 1,000-class checkpoint to only the few class folders present. Use
the dataset uploader's format and the [developer guide](DEVELOPER-GUIDE.md).
Keep the hashes, image counts, subset-selection seed and original archives.
ImageNet-V2 and ImageNet-1K validation are different datasets. No ImageNet images
or private models should be committed to the public repository.

The present uploader allows at most 1,000 images per archive, 50 MiB zipped and
512 MiB of prepared inputs. You cannot reproduce a published complete 50,000-
image score in one current run. A 500-image subset has accuracy steps of 0.2 pp;
1,000 images have 0.1 pp steps. A 0.001 pp top-1 improvement is not resolvable
with this prototype's single-run dataset limit. Use exact correct/total counts.

## 4. Measure PyTorch and standard ONNX on your laptop

1. Upload the trusted PT2 and its exact input contract. Upload the three datasets
   separately. Save their names so you can reuse the same files.
2. Select PT2 → ONNX, **Fixed profiles**, held-out test ZIP, **Estimate: No**.
3. Set **10 warm-ups**, **100 measured runs** initially, **1 CPU thread**. For
   the final timing experiment use **500–1,000 measured runs** if practical,
   and repeat the complete run at least three independent times. Keep your
   laptop plugged in and the same power plan; close unrelated busy programs.
4. Save HTML/JSON/CSV, run ID, standard ONNX artifact and its SHA-256. Check the
   host CPU, runtime versions, thread count, warm-ups and raw timing count.
5. Compare PyTorch and standard ONNX accuracy, prediction agreement, output
   error, median/p95 latency and actual bytes. PyTorch export bytes are not the
   creator's checkpoint size. Runtime speedup alone is not proof of an improved
   conversion algorithm.

## 5. Measure EdgeLens strategies against that standard baseline

Run **Fidelity-guided ONNX conversion** with the same PT2, held-out test and
separate calibration ZIP. Then run **FP32 versus static INT8** and **Deployment
validation search** with all three datasets. Start with MinMax/per-channel,
then test other supported settings. Predeclare the objective, accuracy-loss
limit and size/latency constraints before reviewing final test results.

Preserve the standard baseline, each successful/failed candidate, validation
rationale and selected final artifact. Do not tune settings using final test
images. Record each result, including equal/worse results, FP32 retention and
no-feasible-candidate outcomes. Excluding an operation can improve validation
accuracy while failing to improve held-out accuracy.

For each **same-run** candidate against standard, calculate:

- Accuracy difference in pp = candidate accuracy − baseline accuracy.
- Median latency reduction (%) = `(baseline_ms − candidate_ms) / baseline_ms × 100`.
- File-size reduction (%) = `(baseline_bytes − candidate_bytes) / baseline_bytes × 100`.
- Conversion/search cost separately, in seconds. It is not inference latency.

Read median, p95, SD and raw samples across repetitions. Tiny overlapping
differences do not establish a reliable speed advantage. Report mean/median
across complete runs and their range; do not choose only the fastest run.

## 6. Measure TFLite fairly

For your own PT2 → TFLite conversion, use the pinned Linux worker from
[Stages 1–5](STAGES-1-5.md). Its matched PyTorch export version matters. The
GitHub smoke workflow tests only the generated tiny classifier, not your
uploaded MobileNetV2. Local Windows/WSL conversion is not verified. Unsupported
models must retain their real failure.

Import the worker's inner report ZIP into Windows. Its timings remain **Linux
worker measurements**. Download its FP32 and INT8 TFLite files and upload each
as a standalone classifier to the **Windows** tool. Use the same held-out
images, scale/normalization, labels and each artifact's actual input layout and
shape. The runtime handles supported quantized I/O. Do not copy NCHW settings
blindly to an NHWC artifact.

Run each TFLite artifact at the same thread/timing settings and repeat it as in
Step 4. These fresh Windows runs permit a laptop runtime comparison with your
Windows PyTorch/ONNX runs. Keep the imported Linux evidence separately. A
standalone TFLite run has no original PyTorch conversion-loss comparison; its
worker report supplies that lineage. Float operations may remain in an INT8
candidate. Fully integer MCU execution and firmware fit are separate tests.

## 7. Add internet references without mixing experiments

The checked starter is [reference-examples.json](reference-examples.json), dated
8 October 2026. It includes primary TorchVision and ONNX Model Zoo figures and
a TFLite methodology row with unavailable numeric metrics. Unknown latency is
left blank; no hardware-dependent value is invented.

The [ONNX Model Zoo MobileNet table](https://github.com/onnx/models/blob/main/validated/vision/classification/mobilenet/README.md)
uses a different, MXNet-trained lineage. Its opset-12 FP32 and QDQ scores are
69.48% and 67.40%; those differences must not be attributed to an EdgeLens
conversion of TorchVision weights. Its rounded download sizes are contextual.
[LiteRT's quantization guide](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization)
explains the conversion method; it does not provide a benchmark of your model.

For every new reference, record exact model/weights, framework/runtime,
precision, source URL, checked date, evaluation dataset/preprocessing, input
shape/batch, device, thread count and latency method. Record file size as
published (MB/MiB/rounded) or measure exact downloaded bytes. Missing details
stay UNAVAILABLE. For a direct advantage claim, reproduce the standard model
**locally** under the same controlled settings rather than subtracting a
different website's hardware score.

## 8. Build the full cross-framework report in the tool

1. Export **JSON** from each measured run: PyTorch/standard ONNX, fidelity
   search, precision/search, Linux worker and fresh Windows TFLite FP32/INT8.
   Keep repeated runs separately. Exclude illustrative demo and fault-only
   reports from the performance comparison.
2. **Reports & history → Compare saved results and published references**.
3. Select 1–12 report JSON files and optionally your edited
   `docs\reference-examples.json`. Use a clear title including model/dataset.
4. Click **Prepare comparison**. Save its HTML, JSON and CSV with the three
   download buttons. Comparison files are not a new SQLite experiment.
5. Read the measured table and per-run charts. Different comparison-group
   IDs identify different dataset/preprocessing contracts; CPU/runtime origin
   and full settings still need checking for latency. Distinct model weights
   can share a dataset group and must not imply shared conversion lineage.
6. Published values appear in a separate **REFERENCE** section. No internet
   superiority percentage is calculated automatically. Same-run standard
   comparisons show observed deltas with their limitations.
   Older exports may name the runtime from their saved profile but lack an
   explicit metric-to-model/dataset hash link. Such links and accuracy deltas
   remain UNAVAILABLE; repeat the experiment with the current tool for complete
   evidence. Check `artifact_identity_status` in comparison JSON/provenance.
7. Open the HTML and print the summary to PDF. Keep JSON/CSV and original
   reports/artifacts as the appendix evidence.

CLI alternative, from the repository root in PowerShell:

```powershell
backend/.venv/Scripts/python.exe scripts/compare-reports.py --reports "C:/path/pytorch-onnx.json" "C:/path/windows-tflite.json" --references docs/reference-examples.json --title "MobileNetV2 held-out comparison" --output .cache/my-final-comparison
```

Replace the two paths with your actual exports. The output folder must be new.
Choose fewer/smaller files if you exceed 8 MiB per file or the tool's total
24 MiB request limit. The tool does not upload these report files to the internet.

## 9. Write what EdgeLens improves, field by field

Use this order for your final document:

| Section | Evidence to include | Claim that evidence can support |
| --- | --- | --- |
| Problem and scope | Developer classifier, input signature, target and limits | A local conversion-validation and reporting workflow |
| Baselines and method | Exact standard exporter/quantizer settings, hashes, disjoint datasets, host versions | A reproducible controlled comparison |
| Accuracy and fidelity | Correct/total, top-1, pp delta, agreement, MAE/max, tolerance | Measured improvement, equal result or loss for the tested artifact only |
| Runtime and storage | Median/p95/SD across repeats, exact bytes, conversion/search time | A measured trade-off under the recorded conditions |
| Diagnostics | Fault suite, mapped boundaries, first difference, unavailable reasons | Successful detection of the tested faults and actionable evidence |
| Constraint selection | Validation results, rejection reasons, frozen selection | Keeping an accurate baseline or selecting a feasible configuration |
| Developer workflow | Saved history, exports, worker provenance, reference comparisons | Integrated inspection and reproducible reporting |
| Edge readiness | Exact-artifact provider receipt | ESTIMATED target analysis; physical hardware stays UNAVAILABLE |
| Limitations and conclusion | Unsupported cases, dataset resolution, repeated-run uncertainty | A bounded conclusion with no universal superiority claim |

EdgeLens builds on PyTorch, ONNX Runtime and LiteRT converters. Standard tools
also have their own optimization/debugging capabilities. Compare a **defined
workflow and tested versions**, not an unsupported assertion that all other
tools lack diagnostics. EdgeLens's contribution is its combined validation,
selection, provenance and developer report workflow; numerical gains must be
measured independently.

Your earlier 500-image ImageNet-V2 run reported PyTorch/FP32 69.2% and initial
INT8 58.8%. Its validation constraints selected FP32. That supports accuracy
preservation through constraint handling, with the size trade-off stated. The
earlier 10-sample timing needs fresh repeats before a final latency claim. A
synthetic classifier's 100% accuracy verifies plumbing, not real-world quality.

Finish with a table of **where it improved, where it tied, where it lost and
where evidence is unavailable**. Use your actual numbers to decide those cells.
