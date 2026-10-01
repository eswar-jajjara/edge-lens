# FP32 and static INT8 experiments

EdgeLens 0.5.0 extends the existing desktop benchmark engine with a fixed FP32
versus static INT8 ONNX experiment. It records actual results without selecting a
winner. Sensitivity search, mixed precision and constraint selection are later
milestones.

## Run your own classifier

1. Update the source and run `setup-windows.cmd`, then `start-desktop.cmd`.
2. Save a trusted exported `.pt2` or single-file **FP32** `.onnx` classifier in
   **Bring your own classifier**. Declare its shape, class count and exact training
   preprocessing. The current contract is one fixed batch-one float32 image input
   and one `[1, classes]` output. Already quantized ONNX is not an FP32 baseline.
3. Upload three labelled image ZIPs. Each contains class folders and a root
   `labels.json`, using the same class-index mapping:
   - **Calibration:** determines activation ranges for static INT8.
   - **Validation:** evaluates the fixed configurations; reserved for future search.
   - **Held-out test:** final accuracy and output comparison, never quantizer input.
4. Choose **Compare FP32 vs static INT8 · ONNX**. Select calibration and validation
   above; select your held-out test ZIP in the main dataset field. Choose MinMax,
   Entropy or Percentile calibration and per-channel/per-tensor weights.
5. Run the experiment. Read **Candidate outcomes** and export JSON, CSV or HTML
   from Reports. Failed candidates retain their reason while successful candidates
   keep their measured evidence and downloadable artifacts.

ZIP limits are 50 MiB compressed, 200 MiB expanded and 1,000 images per archive.
The three preprocessed input sets together must fit 512 MiB. At 224×224 RGB this
allows fewer images than the archive limit. Use several hundred representative
test images; tiny sets are workflow checks. A change of one correct prediction
moves top-1 accuracy by `100 / N` percentage points. The bounded MVP does not support
claims of a 0.001 pp top-1 improvement.

## What the experiment measures

- True-label accuracy, correct/sample counts, agreement, output MAE and maximum
  absolute error on validation and test separately.
- Mean, median and p95 host CPU invocation latency, raw samples, warmups, threads
  and runtime settings. Preprocessing/loading/conversion/diagnostics are excluded.
- Serialized bytes and conversion time, including the reused FP32 export
  dependency; incremental and dependency build times are also retained.
- Model/artifact/archive/preprocessed-input hashes and software versions.
- Exact quantizer settings and observed QDQ operation/weight-type inventory.

PT2 uploads use their original PyTorch output as the numerical reference. ONNX
uploads use their FP32 baseline, so they establish quantization differences, not
original PyTorch conversion loss. The ONNX runtime settings are identical across
FP32 and INT8: CPUExecutionProvider, ORT_ENABLE_ALL, sequential execution and
declared threads. The quantizer uses signed INT8 QDQ, Conv/Gemm/MatMul operation
selection, shape inference without graph rewriting and calibration-only inputs.
The pinned Windows runtime also saves calibration ranges next to the experiment
and provides their hash/download for reproduction. Older compatible ORT runtimes
without calibration-cache support use their standard calibration API.

Every reported metric has `MEASURED` or `UNAVAILABLE` provenance. Peak RAM and
physical-device latency remain unavailable. QDQ coverage is not proof that every
operation uses an INT8 hardware kernel. Operator inventory is not a proven layer
correspondence or a root-cause/sensitivity diagnosis. Source ONNX with control-flow
subgraphs is outside this initial experiment.

SQLite migration 3 adds `run_candidates` and `run_datasets` without replacing old
reports. Candidate configuration, status/failure, validation/test metrics and
dataset roles are stored with the experiment. Model binaries stay on disk with
hash metadata, following the existing storage architecture.

ESP32 firmware/capture remains available for TFLite artifacts in its existing
workflow. ONNX precision candidates are not direct ESP32 firmware. Selecting
Raspberry Pi does not measure it; a future physical Pi runner will be separate.

## Reproducible synthetic workflow check

From the project root, run these PowerShell commands after Windows setup:

```powershell
& backend/.venv/Scripts/python.exe scripts/create-precision-demo.py --output .cache/precision-demo
& backend/.venv/Scripts/python.exe backend/cli.py --model .cache/precision-demo/colour-classifier.pt2 --spec .cache/precision-demo/spec.json --calibration .cache/precision-demo/calibration.zip --validation .cache/precision-demo/validation.zip --dataset .cache/precision-demo/test.zip --format onnx --strategy quantization_compare --target raspberry_pi --output .cache/precision-results --trust-own-model
```

The generator trains a tiny colour classifier on separate synthetic training
images, then exports PT2 and 100 calibration/100 validation/300 test images from
different seeds. It refuses to replace an existing output directory. In the GUI,
use shape `1,3,8,8`, two classes, NCHW and default scale/mean/std. Reports are written
under `.cache/precision-results/runs/<run-id>/` and persisted in that directory's
SQLite database. A small model can become larger/slower after QDQ; those actual
outcomes are retained.

**Synthetic demo accuracy only checks this workflow. Use a real model and
representative labelled images for your review's model-quality evidence.**

The v0.4.1 portable download predates this feature. Use current source setup or a
later portable release built from this source.
