# Verification — EdgeLens

## Phase 2 — 2 October 2026, source version 0.6.0

- The full 55-test backend suite passed (54 passed, one optional pretrained
  download test skipped). The Phase 2 checks exercise actual tiny PT2 export,
  static QDQ, operation diagnostics, single-operation exclusions, validation-only
  selection, no-feasible-candidate behavior, failed-candidate exclusion, Pareto
  membership and transactional SQLite 3 → 4 migration preserving history.
  A follow-up diagnostic-only test covers a two-control budget with zero probes.
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
