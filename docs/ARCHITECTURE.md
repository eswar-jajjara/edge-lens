# Architecture — EdgeLens 0.6

EdgeLens is a desktop developer tool. Electron starts a private loopback Python engine and renders the bundled local interface. A CLI can invoke the same engine without a GUI or HTTP service.

```text
Developer export + preprocessing + labelled images
  → model/dataset validation and SHA256 storage
  → PT2 reference execution, or imported ONNX/TFLite execution
  → ONNX fixed profiles or calibration-guided fidelity search
  → optional INT8 calibration variants and controlled FP32 exclusions
  → validation constraints, explicit objective and Pareto evidence
  → held-out classification/numerical tests and host timings
  → operation diagnostics + per-image evidence
  → SQLite experiment record and HTML/CSV/JSON reports
       ├─ optional ESP-IDF package → actual ESP32 → USB/JSON report
       └─ optional explicit TFLite upload → Edge Impulse analysis
```

| Layer | Framework/library | Main algorithms/responsibilities |
|---|---|---|
| Desktop | Electron, Chromium | Private local session, engine lifecycle, native downloads |
| Interface | HTML/CSS/JavaScript | Model and dataset forms, history, evidence tables, separate device/provider results |
| API | FastAPI, Pydantic | Typed requests, bounded uploads, job admission, local-only feature switch |
| Source execution | PyTorch `torch.export` | Load trusted exported graph; batch-one reference inference |
| Conversion | PyTorch ONNX dynamo exporter; optional Linux LiteRT Torch | Standard graph export and graph-preserving variants |
| Selection | EdgeLens fidelity search | Lexicographic minimum: tolerance failures → maximum absolute error → MAE; baseline wins ties |
| Deployment search | EdgeLens, ONNX Runtime static QDQ quantizer | MinMax/Entropy/Percentile × per-channel/per-tensor weights; one-operation and cumulative exclusions; validation constraints, explicit lexicographic objective, Pareto dominance |
| Evaluation | NumPy, Pillow, ONNX Runtime, LiteRT | Labelled top-1, argmax agreement, absolute error, `allclose`, median/p95 timing |
| Diagnostics | Torch FX interpreter, ONNX metadata/shape inference | Conservative exact-node mapping; bounded tensor capture; explicit unmapped entries |
| Persistence | SQLite/WAL, filesystem | Transactional runs/metrics/layers/artifacts, SHA256 identity, immutable edge records |
| Physical hardware | ESP-IDF, Espressif TFLite Micro, ESP-NN, pyserial | Micro kernel registration, `Invoke()` timing with `esp_timer_get_time`, arena usage, output check |
| Provider analysis | Edge Impulse Studio REST API, httpx | Optional asynchronous resource profile; source-labelled estimates |

## Source layout

```text
desktop/main.cjs                        Native shell and local session
frontend/index.html                    Interface pages
frontend/src/{app,api,demo-data}.js     UI, transport, isolated illustrative data
backend/desktop_entry.py                Authenticated loopback engine + static UI
backend/cli.py                          Headless developer workflow
backend/app/api/routes/models.py        Own-model upload/library
backend/app/api/routes/runs.py          Jobs, artifacts, report exports
backend/app/api/routes/edge.py          Packages, serial capture, provider jobs
backend/app/services/models.py          Model/preprocessing contract and hashes
backend/app/services/developer_benchmark.py  Exported classifiers and fidelity search
backend/app/services/quantization.py      Shared static quantizer and coverage evidence
backend/app/services/deployment_search.py Validation-only bounded configuration search
backend/app/services/quantization_diagnostics.py Preserved ONNX activation comparisons
backend/app/services/benchmark.py       Registered pretrained models and shared metrics
backend/app/services/hardware.py        ESP-IDF source generator and report validator
backend/app/services/edge_impulse.py    Optional fixed-provider API client
backend/app/services/reports.py         Developer HTML/CSV reports
backend/app/repositories/sqlite.py      Persistence and additive migrations through v4
backend/tests/test_developer.py         Actual tiny PT2/ONNX/TFLite integration checks
```

## Boundaries

- The one-worker queue serializes CPU benchmarks because PyTorch thread settings are process-global. Custom execution is local and trusted; it is not a sandbox for third-party model code.
- Calibration and held-out datasets have separate IDs; archive equality and duplicate preprocessed inputs are rejected. Calibration selects numerical fidelity, never held-out accuracy.
- Deployment search uses three disjoint splits. Calibration controls INT8 ranges and activation diagnostics; validation controls exclusions, constraints and selection; test evaluation follows selection freeze. Diagnostics match unique preserved ONNX operation boundaries and report unavailable mappings explicitly.
- Model and image binaries live on disk, with hashes, paths and metadata in SQLite. Public API reports omit internal filesystem paths. Exported ONNX files can still contain source metadata from the original exporter.
- Completed host reports stay intact. Hardware packages, observations and Edge Impulse jobs/results are stored separately and joined into report exports. Concurrent additions cannot replace the host result.
- USB/device JSON is validated against an exact prepared package. The system does not cryptographically attest hardware or infer full dataset accuracy from one image.
- API keys are transient. Edge Impulse uploads occur only on the explicit UI action. Provider estimates never become host or physical-device timings.
- Raspberry Pi is currently a deployment goal. A Pi device runner remains future work.
