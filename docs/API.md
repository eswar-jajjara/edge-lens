# Local API — version 0.8

## Session connections, per-test estimates and history deletion

Desktop requests still require the private local session and same-origin writes.
Connection/deletion routes are enabled in the local developer workspace only.
Validation failures omit submitted values, including malformed credentials.

| Route | Behavior |
|---|---|
| GET `/edge/impulse/connections` | Safe connection/project IDs and names; no credentials |
| POST `/edge/impulse/connections` | `{api_key}`; verify project access and hold key in memory for up to eight hours |
| DELETE `/edge/impulse/connections/{id}` | Remove credential from memory; 204 |
| DELETE `/runs/{run_id}` | Delete one finished local test and generated artifacts; preserve uploaded inputs; active runs return 409 |

Target loading, profile submission and refresh accept either `connection_id` or
the legacy transient `api_key`, never both. A connection can access only projects
returned by Studio for that key. Account-wide OAuth is not implemented in 0.8.

`POST /runs` accepts `edge_estimate:{enabled:false}` (default), or
`edge_estimate:{enabled:true,project_id:1126810,project_name:"My project",device:"espressif-esp32"}`.
Yes requires a project and target; No must not include a destination. Credentials
and connection IDs are rejected from this persisted preference. Reports expose
it as `edge_estimate_request`. The desktop orchestrates the authorized upload
after evaluation. Creating a run through the API/CLI alone never uploads, and
opening saved history never automatically resubmits.

Deletion returns `deleted_run_id`, `artifacts_removed` and
`models_and_datasets_preserved`. Report rows and legacy edge records are removed
transactionally. A validated run-specific directory is staged first and restored
if the transaction fails; linked paths are rejected. Locked final cleanup may
leave a local `.deleted_` directory while history deletion succeeds. Remote
Studio jobs and previously exported files are unaffected.

Base: `/api/v1`. The desktop engine binds a random loopback port and requires its private session cookie/header. Use the CLI for ordinary scripting without needing desktop session credentials. The standalone development server binds `127.0.0.1:8000`.

| Method / path | Purpose |
|---|---|
| GET `/health`, `/capabilities` | Engine/runtime status, upload enablement, imported-TFLite availability |
| GET / POST `/models` | List exported models / upload raw binary with percent-encoded JSON `X-Model-Spec` header |
| GET / POST `/datasets` | List datasets / raw ZIP body, optional `X-Dataset-Name` header |
| POST `/runs` | Queue a measured experiment (202); maximum two active/queued jobs, one worker |
| GET `/runs`, `/runs/{id}` | History / detailed state and report |
| GET `/runs/{id}/report`, `.html`, `.csv` | Export full evidence, including attached edge observations |
| GET `/runs/{id}/artifacts/{index}` | Download a checksummed saved artifact confined to its run directory |
| GET `/edge/ports` | Enumerate local serial ports; no probing or flashing |
| GET `/runs/{id}/edge` | Packages, hardware observations and provider jobs/results for that run |
| POST `/runs/{id}/edge/packages` | Generate ESP-IDF ZIP using `artifact_index`, optional `arena_kib` (16–2048, default96) |
| GET `/edge/packages/{id}/download` | Download generated firmware package |
| POST `/edge/packages/{id}/capture` | Read matching serial report; body `{"port":"COM5"}`; 12-second timeout |
| POST `/edge/packages/{id}/import` | Import the protocol JSON; validates identity and values; marks import provenance |
| POST `/runs/{id}/edge/impulse` | Explicit provider upload/profile request |
| POST `/edge/impulse/{record_id}/refresh` | Retrieve completed provider result with transient API key |

Model spec fields: `name`, `format` (`pt2/onnx/tflite`), `input_shape`, `layout` (`NCHW/NHWC`), `class_count`, `scale`, `mean`, `std`, `resize` (`stretch/shortest_center_crop`), optional `resize_shorter`, `trusted_source:true`. Upload limit100MiB; stored metadata excludes filesystem paths from public responses. PT2 is a trusted executable serialization, not a safe interchange format for arbitrary parties.

Run request:

```json
{"model_id":"model_returned_by_upload","format":"onnx","target":"esp32","dataset_id":"ds_test","strategy":"fidelity_search","calibration_dataset_id":"ds_calibration","settings":{"atol":0.0001,"rtol":0.001,"warmup_runs":3,"measured_runs":10,"threads":1}}
```

`fixed_profiles` is the default. Fidelity search requires uploaded PT2 + ONNX and separate calibration data. Preset IDs `mobilenet_v2` and `resnet18` remain supported with fixed profiles. Imported ONNX/TFLite must use their existing format and do not manufacture a PyTorch reference. Limits: warm-ups1–50, timed runs3–200, threads1–8, finite tolerances0–1.

FP32/static-INT8 experiment request (uploaded PT2 or **FP32 ONNX**, ONNX output):

```json
{"model_id":"model_returned_by_upload","format":"onnx","target":"raspberry_pi","dataset_id":"ds_test","strategy":"quantization_compare","calibration_dataset_id":"ds_calibration","validation_dataset_id":"ds_validation","quantization":{"calibration_method":"MinMax","per_channel":true},"settings":{"atol":0.0001,"rtol":0.001,"warmup_runs":3,"measured_runs":10,"threads":1}}
```

All three IDs/archive hashes and preprocessed image sets must be separate.
Calibration methods: `MinMax`, `Entropy`, `Percentile`; `per_channel` is a strict
boolean. Unknown options are rejected. Reports include `candidates` (including
failures), `datasets` by role, validation/test metrics, raw timings and
`provenance`. No automatic candidate selection occurs. See
[precision methodology](PRECISION-EXPERIMENTS.md).

Deployment search uses the same upload and run endpoints:

```json
{"model_id":"model_returned_by_upload","format":"onnx","target":"raspberry_pi","dataset_id":"ds_test","strategy":"deployment_search","calibration_dataset_id":"ds_calibration","validation_dataset_id":"ds_validation","quantization":{"calibration_method":"MinMax","per_channel":true},"search":{"calibration_methods":["MinMax","Entropy","Percentile"],"try_per_tensor":true,"max_candidates":16,"sensitivity_probes":4,"diagnostic_samples":3,"max_seconds":600},"constraints":{"objective":"size","max_accuracy_loss_pp":1,"max_size_mib":null,"max_host_latency_ms":null}}
```

`search` rejects unknown options: 2–24 candidates including FP32 and the initial
INT8 control; 0–8 operation probes; 1–8 diagnostic calibration images; 10–1,800
seconds as a soft budget checked between operations. Calibration methods must
be unique; `try_per_tensor` is a strict boolean. Objectives are `tradeoffs`
(no automatic choice), `size`, `latency` and `accuracy`. Constraints accept finite
values or null; accuracy loss is 0–100 percentage points relative to **validation
FP32**, size is >0–100 MiB, host median latency is >0–60,000 ms. Unsupported device
latency/RAM constraints are rejected.

Schema-4 search reports include `selection`, candidate `eligibility` and
`pareto_efficient`, `diagnostics`, `sensitivity`, dataset hashes, validation
timings, and final test results only for the references, initial control and
selected configuration. No feasible candidate is an explicit result. See
[deployment search methodology](DEPLOYMENT-SEARCH.md).

Firmware observation protocol: `edgelens.esp32.v1`, `package_id`, model/input SHA256, chip, IDF version, CPU MHz, arena used/capacity bytes, warm-up count, `samples_us`, output score vector. Values must match the saved manifest and be finite. Unknown fields are rejected. Hardware records preserve raw samples and capture/import source. This is report validation, not hardware attestation.

Edge Impulse submission: `artifact_index`, positive `project_id`, exact supported MCU `device`, transient `api_key`, `consent_upload:true`. Refresh: `{"api_key":"..."}`. These actions use only `https://studio.edgeimpulse.com/v1/api`. Redirects are disabled; keys and request bodies are not stored in SQLite. Provider results have `measurement_scope:provider_analysis`.

SQLite migrations preserve old data: schema 2 adds `custom_models`/`edge_records`, schema 3 adds `run_candidates`/`run_datasets`, and schema 4 adds `sensitivity_results`/`deployment_selections`. Reports and normalized records are saved transactionally. API startup marks interrupted jobs failed. No retry synthesizes successful measurements. Model artifacts remain in the data directory; public report paths are sanitized.

### Phase 3 provider evidence

`POST /edge/impulse/targets` accepts `{project_id, api_key}` and returns only supported `{mcu,name}` pairs from the project information API. It does not upload a model or store the key. Profile submission now returns 409 before network access unless the selected TFLite SHA-256 has a complete, matching held-out metric. It also verifies the bytes actually sent. Job/result records carry `evaluated_artifact`, model hash, target, provider URL and timestamp. Results retain `raw_response` (credential/request-byte fields removed) and `response_sha256`. Public reports include derived `evidence_sections` and candidate-scoped `diagnostic_summary`; these additions require no SQLite migration. Old runs remain readable with unavailable accuracy links.
