# Verification — EdgeLens

## Review-II objective work — source version 0.9.0

- **MEASURED:** seven synthetic known-fault/control cases pass, including a
  rewired edge, altered weights/operator, invalid shape, missing operator and
  preprocessing drift. The native-tool Diagnostics action saves results in
  SQLite and supports HTML/CSV/JSON exports. First observed divergence is not
  causal proof. Existing uploaded models are never modified.
- **MEASURED:** a fresh Windows tiny-classifier PT2/ONNX experiment uses 300
  synthetic held-out images, 10 warm-ups, 100 timed invocations and 16 shared
  inputs. Both candidate graphs have 5/5 supported structural boundaries and
  the unoptimized diagnostic profile has all five numerical boundaries plus
  final output. Other models may have unavailable mappings. Input renaming and
  the declared pooling/flattening decomposition rules are covered by tests.
- **MEASURED:** Linux workflow 37006609016 converted and evaluated actual PT2,
  FP32 TFLite and calibrated static INT8 TFLite on the same 300 synthetic images,
  with 100 separate calibration images. Accuracy was 100% for each; this is
  plumbing evidence, not real-world model quality or converter superiority.
  FP32/INT8 artifacts were 3,376 / 4,040 bytes; maximum output differences versus
  PyTorch were 0.0000019073486328125 / 0.06374835968017578. INT8 contains seven
  INT8 tensors with calibrated integer weights and activations, while float
  tensors/I/O remain. Neither fully-integer firmware nor MCU compatibility is
  established. The configured 2 MiB limit is a model-file workflow budget.
- **MEASURED:** torch 2.9.1+cpu / torchvision 0.24.1+cpu / torchao 0.17.0+cpu /
  litert-torch 0.9.4 / LiteRT 2.2.0 work together on Linux x64 Python 3.11. The
  complete dependency snapshot is pinned. An earlier torch 2.13 trial failed on
  an ATen overload and is not the supported stack. Local WSL access was denied
  and Docker's Linux daemon was absent: **UNAVAILABLE locally**.
- **MEASURED:** reports retain min/max, quartiles, median, p95, SD and raw timing
  samples/input indices. Process RSS is sampled in a separate inference pass;
  it includes all loaded models/runtime/dataset state and is not guaranteed peak
  or model-only RAM. CPU, threads and active Windows power plan are recorded;
  unavailable power metadata is labelled. No tiny-difference significance claim.
- Worker bundles verify artifact hashes, sizes and held-out evaluation links
  before entering SQLite. Unexpected paths and changed bytes are rejected;
  import never executes PT2. Measurements retain worker origin rather than
  being presented as re-measured on the laptop. Trusted-worker measurement
  claims are not cryptographically attested.
- **ESTIMATED:** the existing live Edge Impulse job 54379928 was rechecked against
  its original evaluated TFLite hash and sanitized-response hash. It is not
  reassigned to the new converted model. A **new live profile is UNAVAILABLE**
  until the user connects their project key in the tool. Existing hash guards,
  consent, redaction and provenance tests remain in place. Actual ESP32 results
  remain **UNAVAILABLE**.
- Local backend checks: **86 total, 85 passed, one optional pretrained-download
  smoke skipped**, with full custom-model execution enabled. Frontend checks:
  **17 passed**, syntax/assets/build passed. Final GitHub/Windows packaging
  checks are recorded separately in the delivery result.


## 0.8.0 project connections and history deletion — 2 October 2026

- Backend: **76 tests, 75 passed, one optional pretrained download skipped** with
  `EDGELENS_TEST_CUSTOM=1`. New checks cover scoped project connections,
  expiration/disconnect, provider failures, malformed-key redaction, exact
  connected artifact profiling, project provenance, credential-free exports,
  Yes/No destination validation and transactional report deletion/rollback.
- Frontend: **17 tests passed**; syntax, assets, native main/preload syntax and
  build passed. Decision tests require a measured TFLite artifact, reject an
  ONNX-only estimate and require selection for multiple TFLite artifacts.
- Browser verification of the embedded desktop renderer uses an isolated local
  dataset/history. The separate provider page is visible before a run, can open
  the previous verified live receipt, and Yes is blocked without a connected
  project/target. A real No-choice TFLite benchmark evaluates the synthetic
  held-out fixture without submitting a provider job. Normal desktop history
  is not deleted or changed by these tests.
- Connections and orchestration were tested locally with mocked provider calls.
  The earlier live profile below is real; a fresh live key connection and Yes
  test in this update still need user verification. No browser-account OAuth
  client is registered or implemented, so automatic all-account project listing
  is unavailable. Studio sign-in plus explicit project-key access is supported.
- Native Electron startup remains blocked in this agent's Windows sandbox by
  IPC permissions. The Windows workflow now verifies preload/browser action,
  separate section, Yes/No controls and project selectors in source/portable
  startup. Inspect the workflow for the commit you use; local browser preview
  does not establish native startup success. No new public portable ZIP is
  released by this source update. Physical hardware remains UNAVAILABLE.

See [the Edge Impulse walkthrough](EDGE-IMPULSE-CONNECTION.md) and
[Windows teammate setup](TEAM-SETUP.md).

## Phase 3 local checkpoint — 2 October 2026, source version 0.7.0

- Full backend suite: 65 tests, 64 passed, one optional download test skipped.
  New contracts cover exact held-out model/dataset identity, missing/legacy
  provenance, pre-upload checksum rejection, explicit consent, target discovery,
  sanitized raw receipts and independent host/provider/device evidence.
- All 13 frontend transport tests, JavaScript/native-entry syntax and the
  interface build passed. Browser verification of the desktop interface used an
  isolated local workspace; it did not change the user's desktop history.
- A real 6,912-byte, hand-authored 32x32 INT8 TFLite fixture was evaluated on 300
  synthetic labelled images using the actual LiteRT interpreter and CLI. Its
  labelled accuracy was 100% in this plumbing test. SHA-256:
  `6c6f1735e0fd930a79b05556c3d7afa7a594dd760761390ea29a5d89c9dbc407`.
  This is not a PyTorch conversion result, trained model quality claim or a
  physical hardware measurement. The saved metric hash matches the evaluated bytes.
- A read-only rendering of the previous MobileNetV2 report shows the diagnosed
  candidate's 53/53 eligible operations on three calibration images. The UI
  candidate filter shows exactly those 53 comparisons; inventory shows 100 of
  351 entries initially with a load-more control. HTML groups measured evidence,
  failed/unavailable diagnostics and inventory separately. The original measured
  metrics and source artifacts were not rerun or changed.
- SQLite remains schema 4; saved metric/evaluation fields and provider receipts
  are additive JSON. Old history remains readable, with unavailable provider
  accuracy links unless explicit evaluated-artifact provenance exists.
- One live Edge Impulse TFLite job completed for ESP-EYE (ESP32 240MHz), with
  `success:true`, INT8, the matching 6,912-byte model, supported-on-MCU status and
  1 ms estimated inference time. Its sanitized raw response and checksum are
  saved in SQLite. Exported HTML/CSV/JSON retain the receipt and a VERIFIED
  300-image evaluation link. The saved model bytes and response checksum were
  independently rechecked. Provider memory details and limits appear in
  [the live result](EDGE-PROFILING.md#live-result--2-october-2026).
- Both [project validation](https://github.com/eswar-jajjara/edge-lens/actions/runs/36988341916)
  and [Windows teammate setup](https://github.com/eswar-jajjara/edge-lens/actions/runs/36988341891)
  passed for the Phase 3 source checkpoint. Windows CI includes native startup
  and the relocated portable runtime. An interactive launch from the local
  agent sandbox hit a Windows IPC access-denied error; that launch is not
  counted as successful desktop verification.
- A QDQ ONNX provider probe, validated Linux TFLite INT8 conversion and physical
  ESP32 benchmarking remain pending; see [Phase 3 scope](EDGE-PROFILING.md).

## Phase 2 — 2 October 2026, source version 0.6.0

- The full 55-test backend suite passed (54 passed, one optional pretrained
  download test skipped). The Phase 2 checks exercise actual tiny PT2 export,
  static QDQ, operation diagnostics, single-operation exclusions, validation-only
  selection, no-feasible-candidate behavior, failed-candidate exclusion, Pareto
  membership and transactional SQLite 3 → 4 migration preserving history.
  A follow-up diagnostic-only test covers a two-control budget with zero probes.
- The real private desktop engine started with an isolated copy of the older
  schema-3 database, migrated it to schema 4 and preserved all six saved runs.
  Its unauthenticated API rejected access. The original desktop database was
  unchanged, and the test-owned engine process was stopped afterward.
- All 12 frontend transport tests, JavaScript/native entry syntax, asset checks
  and the version-0.6.0 interface build passed. Native Windows startup and a
  relocated portable runtime are checked by the existing Windows workflow;
  inspect the Actions result for the source commit being used.
- A real MobileNetV2 ImageNet V2 PT2 experiment used the original 100 calibration
  and 100 validation images. A new 500-image test subset (seed 20261003) excluded
  all 700 images from the earlier experiment, including duplicate prepared pixels.
  The original official checkpoint and declared TorchVision spatial/normalization
  preprocessing were retained. No ImageNet images or model binaries are published.
- The search built 13 successful candidates: FP32, six static INT8 configurations,
  four single-operation exclusions and two cumulative exclusions. It compared all
  53 eligible ONNX operation outputs on three calibration images. Search after
  initial controls took 90.32 s; total strategy work took 158.32 s in this run.
- Validation: FP32 74%, initial MinMax/per-channel INT8 67%, Percentile/per-channel
  INT8 71%, and the best selective candidate 72%. Excluding `node_Conv_909`
  recovered 1 pp against its 71% parent. Other controls include zero/negative
  numerical changes. These are validation observations, not held-out improvement
  or statistical-significance claims.
- With objective **size** and a maximum validation accuracy loss of **1 pp**, only
  FP32 was eligible, so the tool retained FP32. Selection was frozen before test
  inference. The unselected alternatives have validation metrics and no test
  metrics; the original INT8 control remains a final comparison reference.
- Fresh held-out top-1: PyTorch **69.2%**, FP32 **69.2%**, initial INT8 **58.8%**.
  Median host latency was respectively **27.570 / 12.625 / 16.422 ms**; serialized
  bytes **15,411,324 / 14,202,387 / 4,105,648**. ONNX FP32 agreed with all PyTorch
  predictions. Timing uses one image, three warm-ups and ten samples at one thread;
  it is not a device measurement or a reliable tiny-difference speed claim.
- The FP32 and original INT8 control binaries were byte-identical to the user's
  Phase 1 artifacts. All 27 saved artifact hashes were verified. SQLite stored
  13 candidates, three dataset-role records, four sensitivity controls and one
  deployment selection. HTML/CSV/JSON exports completed. This experiment used an
  isolated CLI workspace and did not change the user's desktop history database.

This demonstrates actual configuration search, conservative diagnostics and
constraint handling. It does not establish universal converter superiority.
There is no new public portable release from this source update; the older
v0.4.1 ZIP predates Phase 1/2 and cannot read upgraded databases. Use current
source or rebuild the portable app. Phase 3 physical-device work has not started.

## Phase 1 — source version 0.5.0

- Backend: 50 tests, 49 passed, 1 optional pretrained smoke test skipped. Real
  PT2/ONNX static QDQ, MinMax/Entropy/Percentile calibration, per-channel/per-tensor
  weights, overlap rejection, failed-candidate retention and SQLite migration 3
  are covered. Existing TFLite/ESP32 report behavior still passes.
- Frontend: 12 API tests passed; JavaScript/assets and desktop entry syntax passed.
  In this restricted workspace, Node tests used `--test-isolation=none` because
  child-process test spawning was denied. Windows CI uses ordinary `npm test`.
- An actual trained synthetic colour classifier was exported and evaluated with
  100 calibration, 100 validation and 300 test images using the CLI. Both candidates
  completed, exported reports and were stored with separate dataset hashes and
  calibration statistics. Accuracy was equal; INT8 was larger and slightly slower
  in this one host run. This is workflow evidence, not real-world model-quality
  evidence or a performance/significance claim.
- The desktop interface's precision controls and separate dataset roles were
  checked in a temporary local preview. Native Electron startup in this agent's
  sandbox was blocked by Windows IPC permissions. The existing
  [Windows desktop workflow](https://github.com/eswar-jajjara/edge-lens/actions/workflows/windows-desktop.yml)
  now checks the native precision controls, runs real QDQ tests, and checks the
  relocated portable runtime on a Windows runner. Inspect its result for the
  source commit you use.

At the Phase 1 checkpoint, only fixed FP32/static-INT8 experiments were implemented.
Layer sensitivity, selective quantization search and constraint selection were
added in Phase 2 above. Physical Pi measurements remain unimplemented. No new
portable release was published by the Phase 1 source update.

## 1 October 2026 update — version 0.4.1

- Fresh Windows source installation from a Git archive in a path containing spaces passed on a hosted Windows runner with Python 3.12 x64 and Node.js 22. The setup creates its own virtual environment, installs the pinned CPU packages and downloads the locked Electron executable.
- A real tiny PT2 export/load and ONNX conversion/inference check passed. The check verifies native TFLite/USB imports and built interface assets, and writes a diagnostic JSON file. Synthetic outputs do not establish project accuracy.
- Native Electron startup, private Python engine readiness, real runtime capabilities and interface loading passed in [Windows CI](https://github.com/eswar-jajjara/edge-lens/actions/runs/36851255667). The earlier native-startup limitation below describes the restricted local agent session, not this hosted Windows result.
- The local suite passed 42 backend tests plus 12 frontend tests, with the pretrained download test skipped. FP32/int8 TFLite checks also passed after selecting the installable Windows dependency snapshot.
- Each laptop starts with its own SQLite workspace. Latency is specific to the machine; sharing source does not synchronize model files or saved history.

The Windows 10/11 x64 distribution still needs a teammate's check on their particular laptop. The earlier ESP32, live Edge Impulse, Linux PT2-to-TFLite conversion and model-quality limitations remain applicable.

## Executed successfully

- Actual custom PT2 upload and ONNX export with all three calibration candidates; held-out classification, timing, selection evidence, FX-node diagnostics and saved report artifacts.
- Actual imported ONNX inference without a PyTorch reference; unavailable conversion-loss metrics remain null.
- Actual FP32 **and int8** TFLite inference on Windows with LiteRT 2.2.0. Int8 input quantization and generated byte counts verified.
- ESP-IDF source/ZIP generation from a real TFLite fixture, operator registration, embedded input/model hashes and manifest contents. This checks package generation, not compilation or execution on a board.
- Device-report mismatch rejection, import provenance, timing statistics and single-image output comparison. Test device observations are explicitly synthetic fixtures and are not user hardware evidence.
- Mocked Edge Impulse job submission/result storage and separation from physical-device metrics; keys are not persisted.
- Additive SQLite v1 → v2 migration preserving existing reports and normalized metric rows.
- 42 backend tests pass; one optional pretrained-model smoke test is skipped in this suite. It was executed separately in the previous version. The custom ML tests run with `EDGELENS_TEST_CUSTOM=1`.
- 12 frontend transport tests, JavaScript/resource checks and UI build pass.
- Desktop-mode browser preview: own PT2 upload, test and calibration ZIP uploads, conversion run, candidate evidence, saved history/reopen, precision display, hardware workflow and USB empty state. No console errors observed. Data is isolated under `.cache/ui-v04`, outside the user's desktop history.
- CLI executes the same actual PT2 calibration workflow and writes HTML/CSV/JSON plus SQLite and artifacts. Its synthetic fixture is not accuracy evidence.

## Not yet verified

- **ESP32 hardware:** serial enumeration returned no COM ports. No firmware was flashed, compiled with ESP-IDF, or executed on the user's board. No physical device latency has been measured. Install/check the USB data cable and appropriate USB-UART driver, confirm the exact chip, build/flash the generated project and capture its report.
- **Edge Impulse:** no project credentials were provided. The adapter follows the official profile request/result endpoints, but a live account call has not been made.
- **PT2 → TFLite conversion:** optional Linux LiteRT Torch path is implemented but not runtime-tested here. Windows TFLite import/inference is verified; Windows PyTorch-to-TFLite conversion is unavailable.
- **Native Electron window:** the preceding native startup check was blocked by this agent session's Windows IPC/child-process restrictions. The app must be launched normally outside the restricted test session. Interface verification does not prove native lifecycle behavior.
- **Raspberry Pi:** selectable deployment goal only; no Pi hardware runner or physical test yet.
- **Quality claim:** no developer's real validation dataset was supplied. Tiny synthetic fixtures establish functionality, not classifier quality or superiority. The observed fixture comparison reported equal accuracy.
- No host peak RAM, energy consumption, full-dataset edge accuracy, signed installer, public service, or cloud hardware simulator is provided.

The portable folder includes the updated engine, UI, LiteRT inference and serial dependencies. Keep the entire folder together. Firmware preparation does not guarantee model fit, and imported device JSON is not cryptographic hardware attestation.
