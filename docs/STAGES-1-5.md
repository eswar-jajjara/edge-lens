# Completing the Review-II objectives

These stages extend the **Electron desktop developer tool**. The private local
Python engine remains the benchmark engine. Edge Impulse supplies optional
provider estimates; physical-device results are unavailable until measured.

## 1. Structural diagnostics

Start a **new** supported PT2 → ONNX experiment, then open **Layer diagnostics**.
Old saved reports are preserved; they do not acquire new measurements.

The report now contains separate graph inventories (operators, tensor shapes,
inputs/outputs and producer connections) and structural comparisons. Exact
exporter origins establish a correspondence. Two additional bounded rules cover
unique module scopes for batch-one pooling to 1×1 and flattening to two dimensions.
Ambiguous/reused scopes are not guessed. Final numerical checks remain separate.

- **MATCHED**: the declared operator/data-connection/output-shape checks match.
  It does not prove complete graph equivalence, parameter identity or accuracy.
- **CHANGED**: an observed difference; legal optimization or quantization can
  cause this. Inspect numerical fidelity and the recorded change fields.
- **UNAVAILABLE**: no supported correspondence was established. Reasons identify
  missing origins, decomposition, unsupported operators, capture limits and
  capture failures. Q/DQ inventory helpers are identified separately.

ONNX FP32/QDQ graph comparisons transparently traverse Q/DQ on data edges.
TFLite has a real flatbuffer operator/tensor/dtype/quantization inventory; arbitrary
PyTorch-to-TFLite intermediate numerical mapping remains **UNAVAILABLE**.

## 2. Controlled faults

Open **Layer diagnostics → Run seven checks**. Wait for other benchmarks to
finish first. The tool creates disposable synthetic ONNX models, records the
expected and detected outcomes and saves the report in SQLite history.

Cases: unchanged control, changed weights, replaced operator, rewired connection,
invalid shape, removed operator and incorrect preprocessing. Invalid graphs
must be rejected; measurable faults must drift. A structural change is also
required for the weights/operator/connection cases. A first observed divergence
is not necessarily the cause. Wrong preprocessing cannot be diagnosed uniquely
from output drift without a declared input contract.

CLI alternative:

```powershell
backend\.venv\Scripts\python.exe scripts\validate-faults.py --output .cache\my-fault-check
```

The output directory must be new. These tests do not alter uploaded models or
establish real-world classifier quality.

## 3. Pinned Linux TFLite worker

The verified worker uses **Linux x64, Python 3.11, torch 2.9.1+cpu,
torchvision 0.24.1+cpu, torchao 0.17.0+cpu, litert-torch 0.9.4 and LiteRT 2.2.0**.
All resolved packages are pinned in `backend/requirements-tflite-linux-tested.txt`.
The first torch 2.13 / torchao 0.17 experiment failed on an ATen overload; the
tested 2.9.1 environment resolved it. Do not upgrade these packages together
without repeating the smoke test. Do not install them in the Windows engine.

WSL service access was denied during this local session; Docker's Linux daemon
was not running. Therefore **local WSL operation is UNAVAILABLE**, not verified.
Actual conversion/evaluation has been verified on a GitHub-hosted Linux worker.

To use WSL, first confirm `wsl --list --verbose` works and your distribution has
Python 3.11 with venv support. From the repository in Linux/WSL:

```bash
bash scripts/setup-tflite-worker.sh
.venv-tflite/bin/python scripts/create-precision-demo.py --output worker-fixture
.venv-tflite/bin/python backend/cli.py \
  --model worker-fixture/colour-classifier.pt2 --spec worker-fixture/spec.json \
  --calibration worker-fixture/calibration.zip --dataset worker-fixture/test.zip \
  --format tflite --strategy fixed_profiles --target esp32 \
  --output worker-results --bundle-output my-worker-report.zip --trust-own-model
```

Supply your own trusted model/spec and disjoint calibration/test archives for
real work. Export PT2 with the worker's matched PyTorch version when possible;
newer PT2 serialization/operators may not load in the pinned older converter.
Unsupported models fail explicitly. The worker creates an FP32 control and a
calibrated static INT8 candidate; it is built on the standard LiteRT converter.
Its INT8 candidate can retain float operations and float I/O. It is not assumed
to be fully integer TFLite Micro firmware.

The **Verify Linux TFLite worker** GitHub workflow provides a reproducible smoke
test using only generated synthetic data. Download its `tflite-worker-evidence`
artifact and extract `worker-results/edgelens-worker-report.zip`. Do not commit
private developer models or datasets to this public repository.

In the Windows tool, open **Reports & history → Import a verified worker report**,
choose the **inner worker-report ZIP**, acknowledge the trusted worker and import.
The importer validates every artifact hash/size and held-out evaluation link,
never executes the imported PT2, and keeps Linux timings labelled as imported
worker measurements. Original datasets are not included in the portable report
bundle; keep their originals for reproduction. Artifact identity checking is not
cryptographic attestation that externally supplied measurements are truthful.

The tiny colour classifier has real FP32/INT8 conversion evidence and model files
below the configured **2 MiB model-file budget**. That budget is not a universal
ESP32 limit or proof of total firmware/RAM/operator compatibility.

## 4. Host timing and memory

New defaults: **10 warm-ups and 100 measured invocations**, cycling up to 16 shared
inputs and rotating profile order. The UI/CLI allows up to 1,000 samples. Reports
include all raw timings and input indices, min/max, p25/p50/p75/p95 and sample
standard deviation. Fewer than 100 samples carry a warning. Record independent
repetitions before interpreting small differences; this does not establish
statistical significance. No automatic speed-superiority claim is made.

CPU model, logical CPU count, configured threads, platform and runtime versions
are saved. Windows records the active power plan using `powercfg`; unsupported
or inaccessible values are **UNAVAILABLE**. Linux power governor is currently
**UNAVAILABLE**. The tool records the plan; it does not change it.

RSS is measured in a separate 30-inference pass, sampled every 10 ms and after
each invocation. The sampled maximum includes **all loaded models, datasets,
runtime and engine state**. It is neither model-only memory nor a guaranteed
true peak. Baseline and increase are retained; measurement failure stays
UNAVAILABLE. ESP32 arena usage, process RSS, serialized bytes and total firmware
flash are different quantities.

## 5. Exact-artifact provider profiling

Import the worker report, open **Edge Impulse**, connect a Read + Write project
key, choose the imported test and its evaluated INT8 TFLite artifact, load the
supported targets, choose one and request analysis with consent.

Before upload the engine re-hashes the bytes and requires the artifact hash,
held-out dataset hash, preprocessing, labels, sample count and accuracy to match
the saved metric. The sanitized provider response, uploaded hash, target, job ID,
date and provider provenance are stored separately. Changed bytes are rejected.
API keys remain transient and are excluded from SQLite and exported reports.

A fresh profile of the newly converted model needs a live connected project key.
Historical provider evidence cannot be reassigned to the new model. Failure,
unsupported targets and missing provider responses remain explicit.

## Shared candidate contract

`candidate_schema_version: 1` defines conversion candidate records with `id`,
`format`, `status`, `precision`, `artifact_sha256`, `configuration`,
`conversion_seconds` and `evidence_status`. Metrics carry the candidate profile,
artifact/dataset hashes, evaluation split, sample count and metric provenance.
Calibration metadata is separate from held-out test metadata. Existing Phase 2
candidate records retain their configuration/validation/test fields and are not
silently rewritten. Structural mapping and numeric capture status are distinct.

When updating the PPT in Stage 7, use verified saved reports. Every number must
be identified as **MEASURED**, **ESTIMATED**, **REFERENCE** or **UNAVAILABLE**.
No physical-device result should appear until an actual board test exists.
