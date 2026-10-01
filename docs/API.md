# Local API — version 0.4

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

Firmware observation protocol: `edgelens.esp32.v1`, `package_id`, model/input SHA256, chip, IDF version, CPU MHz, arena used/capacity bytes, warm-up count, `samples_us`, output score vector. Values must match the saved manifest and be finite. Unknown fields are rejected. Hardware records preserve raw samples and capture/import source. This is report validation, not hardware attestation.

Edge Impulse submission: `artifact_index`, positive `project_id`, exact supported MCU `device`, transient `api_key`, `consent_upload:true`. Refresh: `{"api_key":"..."}`. These actions use only `https://studio.edgeimpulse.com/v1/api`. Redirects are disabled; keys and request bodies are not stored in SQLite. Provider results have `measurement_scope:provider_analysis`.

SQLite schema2 adds `custom_models` and `edge_records` without replacing v1 datasets/runs/normalized rows. API startup marks interrupted jobs failed. No retry synthesizes successful measurements. Model artifacts remain in the data directory; public report paths are sanitized.
