# Connect Edge Impulse in the desktop tool

EdgeLens 0.8.0 has a separate **Edge Impulse** sidebar section, available before
you run a benchmark. It estimates resources without a physical ESP32. **Edge
hardware** is the separate, later physical-device workflow.

## Sign in and connect a project

1. Start `start-desktop.cmd` from the current source. Close an older EdgeLens
   window first; the application permits only one desktop instance.
2. Select **Edge Impulse → Sign in in my browser**. The tool opens the official
   Studio in your default Internet browser. Sign in there, using whichever
   browser you normally use. EdgeLens never asks for your account password.
3. Open your project in Studio, then **Dashboard → Keys**. Copy a project API key
   with **Read + Write** permission. Enter it only in EdgeLens's password field
   and click **Connect project**. A 403 can mean that this permission or model
   profiling access is unavailable; connection success does not guarantee
   every target/model will be accepted.
4. The tool verifies the key with Studio and lists its accessible project name
   and ID. Select the project, click **Load supported targets**, then choose a
   target. Your existing project is **1126810**; teammates can use their own
   projects. Access to the GitHub source does not grant access to your private
   Edge Impulse project.
5. Add another project's key to connect another project. Use **Disconnect** to
   remove that connection. Keys stay only in the engine's memory for at most
   eight hours and disappear when the application closes. The password field
   is cleared after a connection attempt. Reconnect after restarting.

**Authentication limit:** opening Studio signs you into the browser; it does
not authorize EdgeLens to use your whole account. A project key lists that
project only. Automatic browser OAuth callback and account-wide project
discovery are **not configured or implemented in this version**. They require
a registered EdgeLens OAuth client and verified authorization-code integration.
No browser cookies, passwords, JWTs or client secrets are extracted by this tool.
The project-key workflow is the available connection method.

Official API contract: [list projects](https://docs.edgeimpulse.com/apis/studio/projects/list-projects)
and the [Studio OpenAPI specification](https://docs.edgeimpulse.com/.assets/openapi.yaml).
The specification distinguishes API-key project access from account JWT access.

## Estimate a new test: Yes or No

In **Overview**, every new benchmark has **Estimate this test in Edge Impulse?**

- **No:** local benchmarking only. No model upload or provider profiling job is
  triggered by that test. This works without an Edge Impulse account.
- **Yes:** choose a connected project and a loaded provider target. This choice
  explicitly authorizes uploading the evaluated TFLite model after the local
  test. The selected project ID/name and target are saved with the test request
  and exported reports; credentials and connection IDs are not saved there.

After a successful local test with one TFLite artifact, EdgeLens opens the
separate section and submits that exact file once. It checks for a completed
estimate every ten seconds for up to two minutes. If Studio takes longer, select
the saved test/job and click **Fetch completed estimate** later. Opening history
or restarting the tool does **not** upload again. If a submission times out,
check Studio before submitting again; it may already have accepted the job.

Multiple TFLite artifacts require you to select one and explicitly confirm its
upload. An ONNX-only test cannot get an estimate through this TFLite adapter:
its host report remains valid, and the estimate is marked unavailable. A Yes
preference is not proof of a completed provider estimate.

For an older test, select it directly in **Edge Impulse → Saved test**, choose
its evaluated TFLite artifact and project/target, check the upload consent and
click **Upload & request estimate**. Fetching an older job requires connecting
to its original project again.

Only the model artifact is uploaded; dataset images remain local. Model and
dataset hashes connect the saved estimate to the exact held-out evaluation.
Unsupported models, target limitations and provider errors are shown rather
than filled in with illustrative numbers.

## Read or share the result

The section displays project, job, target, timestamp, uploaded SHA-256 and
accuracy-link status. Expand the provider response for timing, RAM and flash
estimates. Export HTML, JSON or CSV from **Reports & history** after a completed
estimate. Evidence remains separate:

| Source | Report label | Meaning |
|---|---|---|
| This laptop | MEASURED | Accuracy and host timing on the selected images |
| Edge Impulse | ESTIMATED | Provider target/resource analysis of the uploaded file |
| Physical ESP32 | UNAVAILABLE until tested | Requires a matching hardware report |

The earlier live Phase 3 profile succeeded; see [edge profiling evidence](EDGE-PROFILING.md).
New project-connection and automatic orchestration paths are covered by local
tests, but a fresh live connection/Yes test should be verified with your own key.

## Delete report history

In **Reports & history**, click **Delete** beside a completed or failed test.
The confirmation names the test and explains the permanent local deletion.
Cancel keeps everything. Confirm removes its SQLite report, metrics, layer and
candidate rows, provider/device records and generated artifacts. Uploaded
models, original datasets, other tests and already exported files stay.

Queued/running tests cannot be deleted. A file-lock failure before deletion
keeps the report; failed final cleanup is reported separately. Deleting local
history does not delete a job or model from Edge Impulse Studio. Export reports
first if you need a copy. Do not manually edit the SQLite schema or delete the
whole data folder to clear one report.
