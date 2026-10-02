# Phase 3 — evidence-linked edge profiling

Phase 3 is in progress. Source version 0.7.0 supplies the local profiling and
reporting checkpoint. The first small TFLite experiment now has a successful
live Edge Impulse response with a verified link to its held-out evaluation.
There are no new physical ESP32 measurements or published portable binaries.

## Responsibilities and evidence

EdgeLens evaluates the actual classifier, records host metrics, diagnoses ONNX
operation differences and searches deployment configurations. Edge Impulse is
an optional remote resource estimator. It is not an ESP32 simulator or a source
of ground-truth accuracy for the EdgeLens test set.

Reports and the interface separate:

| Section | Evidence status |
| --- | --- |
| Laptop / host | MEASURED when the model was actually evaluated |
| Edge Impulse | ESTIMATED when a profile response exists; otherwise UNAVAILABLE |
| ESP32 | UNAVAILABLE until a matching physical observation exists |

The existing firmware/package/import functions remain a later optional workflow.
A provider estimate does not mean a board was flashed or a model fits on a
generic ESP32. ESP-EYE and an ESP32 DevKit may have different memory/build options.

## Exact artifact identity

New evaluations record `artifact_sha256`, `dataset_sha256` and
`evaluation_split:test` in the metric itself. A profile submission must find a
complete held-out evaluation with the same artifact hash, dataset hash and sample
count. The engine rechecks the file and the exact bytes sent before submission.
The job stores the evaluation snapshot, including preprocessing, labels and the
preprocessed dataset hash. Responses retain their target, project/job IDs,
timestamp, provider URL, uploaded model hash and a SHA-256 of the sanitized raw
response. Credentials and request model bytes are removed from responses.

Accuracy links are VERIFIED only when the response's captured evaluation still
matches the saved run. Older reports without explicit evaluated hashes remain
readable, but require re-evaluation before a new profile can be attached. Older
provider records retain UNAVAILABLE accuracy links. No association is inferred
from a matching filename, candidate name or an ONNX/TFLite architecture label.

Provider resources describe the uploaded input artifact under the provider's
profiling/build methods. They do not validate the prediction accuracy of generated
firmware. Unsupported-model and profiling-error fields remain visible; never
replace them with invented fit or timing values.

## First small live experiment

1. Create a private project in [Edge Impulse Studio](https://studio.edgeimpulse.com/).
2. Find its project ID and project API key under Dashboard / Keys. Use the
   **API Keys** section and a key permitted to start profiling jobs; an HMAC key
   is a separate credential. Enter the key only
   into EdgeLens's Project API key field; never commit it or send it in chat.
3. Evaluate a small TFLite classifier in EdgeLens using its true input shape,
   layout, scale, normalization and labelled held-out images. Input dimensions
   are model-specific; 32x32, 96x96 and other supported fixed sizes are allowed.
4. Open the completed run, then Edge hardware. Choose the evaluated TFLite artifact.
5. Enter the project ID and key; click **Load supported targets**. This reads the
   project's `latencyDevices` list and uses each device's `mcu` identifier.
6. Choose a supported target. If using an ESP32-family reference, ESP-EYE is
   an estimate for that provider configuration; it is not evidence from your board.
7. Check the explicit upload-consent checkbox and click **Upload & request analysis**.
8. After Studio finishes the job, click **Fetch completed analysis**. Export the
   refreshed report. Check the model hash, evaluation link and ESTIMATED label.

Before step 8 succeeds, integration status remains **tested with mocks only**
for a run without live evidence. Network or account failures do not fabricate a
successful profile. Keep the original held-out results unchanged.

### Reproducible synthetic fixture

From the repository directory on Windows:

```powershell
backend\.venv\Scripts\python.exe scripts\create-edge-profile-demo.py
backend\.venv\Scripts\python.exe backend\cli.py --model .cache\edge-profile-demo\synthetic-int8.tflite --spec .cache\edge-profile-demo\spec.json --dataset .cache\edge-profile-demo\test.zip --format tflite --target esp32 --output .cache\edge-profile-results --trust-own-model
```

Alternatively upload the generated model and test ZIP in the desktop interface.
Use NHWC `[1,32,32,3]`, 2 classes, scale `1/255`, mean `[0,0,0]`, std `[1,1,1]`
and stretch resize. `spec.json` contains these settings. The file is a
hand-authored, 6,912-byte INT8 fully connected dark/light classifier, with 300
deterministic synthetic images. Its accuracy checks plumbing only. It is not a
trained vision model, a PyTorch conversion result, or evidence of superiority.
Provider estimates for the first run are recorded below. Actual timing and
memory on a physical MCU remain unverified.
Generated binaries/images are ignored by Git.

### Live result — 2 October 2026

The same 6,912-byte fixture was profiled for **Espressif ESP-EYE (ESP32 240MHz)**,
provider target `espressif-esp32`. The successful response reported INT8,
`isSupportedOnMcu: true`, `hasPerformance: true` and **1 ms estimated inference
time**. It returned the following resource estimates:

| Provider build | RAM (bytes) | ROM (bytes) | Arena (bytes) |
| --- | ---: | ---: | ---: |
| TFLite | 5,883 | 31,032 | 5,747 |
| EON | 3,912 | 15,656 | 3,168 |
| TFLite, CMSIS-NN disabled | 5,825 | 26,456 | 5,689 |
| EON, CMSIS-NN disabled | 3,880 | 15,032 | 3,136 |

These are the provider's fields, not measured total board memory. The response
contains one inference-time value; no separate timing per build is inferred.
Model SHA-256:
`6c6f1735e0fd930a79b05556c3d7afa7a594dd760761390ea29a5d89c9dbc407`.
The held-out snapshot matches that artifact, 300 synthetic images, dataset hash,
labels and preprocessing. Its report accuracy link is **VERIFIED**; its resource
section is **ESTIMATED** and its physical ESP32 section remains **UNAVAILABLE**.
The sanitized raw response, response checksum and dated receipt are stored in
the local SQLite workspace and exported report, without credentials.

This verifies a live integration for one small imported TFLite model. It does
not verify PyTorch-to-TFLite conversion, provider acceptance of QDQ ONNX, firmware
prediction equivalence, generic ESP32 DevKit fit, or improved classification
accuracy. EON/TFLite resource differences are provider build differences.

## Layer diagnostics

The overview uses the diagnosed candidate's eligible operation count, rather
than dividing all measured rows by every candidate's graph inventory. The
diagnostics page defaults to numerical comparisons, with separate candidate,
evidence and status filters. Inventory can be opened explicitly; large lists
are shown in bounded pages. HTML exports group each candidate, summarize
operation inventory and Q/DQ helpers, and retain the full entries in expandable
details. CSV/JSON preserve all rows.

Within tolerance means a recorded numerical comparison met the declared limits.
Needs review means observed drift or a failed comparison. Not compared means
that numerical evidence is unavailable. An inventory entry is not a failed
conversion, and Q/DQ helpers are not extra original PyTorch layers. Calibration
diagnostics on one candidate are not coverage of every alternative or the entire
held-out dataset. Controlled validation exclusions remain separate evidence.

## Remaining experiments, in order

1. **Live TFLite profile:** completed for the small synthetic fixture above.
2. **ONNX QDQ probe:** Edge Impulse BYOM accepts ONNX, but the current EdgeLens
   profile adapter is TFLite-only. Test one QDQ file in Studio before expanding
   the adapter. Record its original hash, target, date and provider outcome.
   If the provider converts it internally, resource results describe that
   provider build. Do not attach the original ORT accuracy as deployment accuracy.
3. **Validated TFLite conversion:** the existing optional LiteRT Torch FP32 path
   needs Linux/WSL and runtime verification. It does not implement a verified
   static-INT8 TFLite search. Keep the Linux environment separate from Windows
   requirements, record resolved versions and evaluate the final TFLite bytes
   with equivalent logical preprocessing on the same held-out images. Preserve
   calibration/test separation for an INT8 path. Conversion failures are valid
   outcomes. ONNX QDQ versus TFLite INT8 comparison remains pending.
4. **Physical devices:** optional future add-on. No ESP32 deployment, board
   measurements or Pi runner are claimed by this checkpoint.

Our present evidence is a measured accuracy/size/host-latency trade-off, not
universal conversion or accuracy improvement. A MobileNetV2 size alone does not
establish total firmware flash, runtime RAM, or fit on an MCU.

Official references: [TFLite profiling API](https://docs.edgeimpulse.com/apis/studio/jobs/profile-tflite-model),
[BYOM](https://docs.edgeimpulse.com/studio/projects/dashboard/byom),
[LiteRT Torch Linux requirements](https://github.com/google-ai-edge/litert-torch).
