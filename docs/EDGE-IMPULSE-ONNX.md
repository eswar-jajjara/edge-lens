# ONNX profiling in EdgeLens 0.9.1

EdgeLens now accepts an evaluated **ONNX or TFLite** artifact in its separate Edge Impulse section. This extends Stage 5. It does not add physical ESP32 measurements.

## Try an ONNX test

1. Close the old desktop window and open the current `EdgeLens.exe`, or use the [current source setup](TEAM-SETUP.md).
2. Run a local ONNX classifier benchmark with a labelled held-out dataset. A PyTorch-to-ONNX benchmark also works. Older reports without matching artifact/dataset hashes need to be evaluated again.
3. Open **Edge Impulse**. Connect a **dedicated project** using its Read + Write API key. Browser sign-in alone does not connect the tool. The key stays in engine memory and is not saved in reports or SQLite.
4. Click **Load supported targets** and choose a provider target.
5. Select the saved test and an **evaluated ONNX / TFLite artifact**. The selector includes the format, profile, size and hash. Unsupported formats and unevaluated diagnostic graphs are omitted.
6. Read the ONNX conversion notice, confirm the upload and choose **Upload & request estimate**.
7. EdgeLens saves the upload job, waits for conversion, starts one profiling job and records its result separately. If it takes longer than automatic polling, select the submitted job and use **Fetch completed estimate**. Reconnect the same project after restarting the tool.

For new tests, **Estimate this test in Edge Impulse? Yes** authorizes the upload after evaluation. If several candidates qualify, choose one explicitly in the Edge Impulse section. **No** performs no upload.

## What the report means

| Evidence | Label | Meaning |
| --- | --- | --- |
| Original ONNX accuracy and latency | MEASURED | EdgeLens evaluated the original file on the recorded host and test images. |
| Upload evaluation link | VERIFIED / UNAVAILABLE | The bytes uploaded match the evaluated ONNX SHA-256 and dataset identity. |
| Edge Impulse RAM, flash and inference time | ESTIMATED | Resource analysis of the provider-derived model for the selected target, not ONNX Runtime performance or a physical-device measurement. |
| Provider-converted model accuracy link | UNAVAILABLE | EdgeLens has not downloaded and evaluated the provider-derived model on the same held-out images. Original ONNX accuracy must not be substituted. |
| ESP32 measurements | UNAVAILABLE | Until an actual matching hardware report exists. |

ONNX upload uses Edge Impulse's **Bring Your Own Model** workflow and replaces the selected project's uploaded BYOM model. Keep that project unchanged until the job completes; EdgeLens rejects a changed filename, model metadata or conflicting profile-job ID. These association checks are not cryptographic proof of the provider's derived artifact. One unfinished ONNX workflow per project is permitted in the local engine. Completed responses are cached so later Studio changes cannot silently replace saved evidence.

The uploaded filename includes the ONNX SHA-256. The request supplies the same fixed batch-one image input shape used for evaluation. No dataset images, calibration images or representative-feature arrays are uploaded. Credentials and request model bytes are removed from saved responses. The prototype upload limit remains **20 MiB**.

## Unsupported models and interrupted requests

ONNX acceptance depends on Edge Impulse's supported operations, conversion tools, target and account access. **INT8 QDQ acceptance is not yet live-verified**. FP32 is a useful first probe. Provider failures do not become benchmark successes, and ONNX artifacts are still rejected by the TFLite-only firmware-package endpoint.

An upload/profile POST timeout may mean Studio accepted the job but EdgeLens lost its receipt. The tool records this ambiguity and never automatically resends that POST, including after restarting. Check Studio before explicitly uploading again. A Read-only or unavailable project key can return 403; use the project's Read + Write key inside the tool, never in a report or chat.

## Verification scope

The integration follows the official Python SDK and API bindings (`edgeimpulse-api` 1.95.16 inspected for this implementation): multipart `/pretrained-model/upload`, job `/jobs/{id}/status`, `/pretrained-model/profile` and `/pretrained-model`. No new SDK dependency is installed in EdgeLens.

Transport and workflow tests use mock provider responses, including separate upload/profile job IDs, changed project models, provider failures, timeouts, restart recovery, exact-byte hashes, fixed input shape, redaction, ONNX-derived accuracy isolation and unchanged TFLite transport. **A live ONNX upload is not yet verified.** Until you connect your project and successfully profile a model, that run must show no completed provider estimate.

Primary references: [Edge Impulse supported formats](https://docs.edgeimpulse.com/tools/libraries/sdks/studio/python), [official SDK workflow](https://github.com/edgeimpulse/python-sdk/blob/main/edgeimpulse/model/_functions/profile.py), [official upload helper](https://github.com/edgeimpulse/python-sdk/blob/main/edgeimpulse/util.py), [API bindings](https://pypi.org/project/edgeimpulse-api/1.95.16/).
