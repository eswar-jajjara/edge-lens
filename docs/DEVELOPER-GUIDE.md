# EdgeLens 0.4 — test your own classifier

Open `start-desktop.cmd` from your project folder after running `setup-windows.cmd`, or open `EdgeLens.exe` from the fully extracted Windows portable release. The project can live on any drive. See [teammate setup](TEAM-SETUP.md). The app is a desktop program with a local engine and SQLite storage. A browser preview is only used during interface development.

## What you can upload

| Input | What EdgeLens can establish |
|---|---|
| `.pt2` from `torch.export.save` | Original PyTorch predictions, ONNX conversions, accuracy change, output fidelity, host timing and mapped operator diagnostics |
| Single-file `.onnx`, embedded weights | Uploaded model accuracy, host timing, size and operator inventory; no PyTorch conversion-loss claim |
| `.tflite` | Uploaded model accuracy, host timing, tensor inventory, ESP32 firmware preparation and optional Edge Impulse analysis |

This is architecture-independent within a supported contract, not a promise to execute every possible classifier: one fixed batch-one image input, 1 grayscale or 3 RGB channels, NCHW or NHWC, one finite `[1, class_count]` output, 2–10,000 classes. ONNX/PT2 inputs are float32; imported TFLite also supports per-tensor int8/uint8 quantization. Multi-input/output models, detection, variable input shapes, external ONNX weight files and arbitrary preprocessing code are not supported yet. Model file limit: 100 MiB.

Only load your own trusted exports. `torch.export.load` can execute serialized code; ONNX/TFLite runtimes are also not a security sandbox. The desktop enables developer execution inside its private local session. The standalone server disables it by default; do not expose the developer worker as a public untrusted-upload service.

To export your existing model, run this in **your model's Python environment**:

```python
import torch
model.eval().cpu()  # your actual architecture and already-loaded weights
example = torch.zeros(1, 3, 224, 224)  # replace with your actual input signature
program = torch.export.export(model, (example,))
torch.export.save(program, "my-classifier.pt2")
```

A weights-only `.pth` file cannot reconstruct its architecture automatically. Export using the same compatible PyTorch version as the worker where practical; export formats are version-sensitive.

## Run an experiment yourself

1. Open **Bring your own classifier**. Select your export. Enter its class count, input shape, layout, pixel scale, channel means/stds and resize rule. These must match training. Save the model.
2. Prepare 2–200 held-out JPG/PNG images in a ZIP (up to 50 MiB). At the ZIP root, include `labels.json`, for example `{"cat":0,"dog":1}`. Put images in `cat/` and `dog/`. IDs must match the model's actual output class order. The built-in ImageNet presets still use ImageNet indices, not this example mapping.
3. Upload the test ZIP. For a PT2 conversion, choose ONNX. For an imported ONNX/TFLite model, the original format is selected automatically.
4. For **EdgeLens fidelity-guided ONNX conversion**, upload an additional ZIP of different calibration images and select it in the calibration field. Keep the held-out ZIP selected as the labelled test dataset. Identical archives and duplicate preprocessed images across the two sets are rejected.
5. Select the deployment goal and run the benchmark. Real measurements appear only after execution succeeds. Unsupported models show a saved failure, not demo results.
6. Inspect **Layer diagnostics** and **Reports & history**. Export HTML for reading, CSV for analysis, JSON for complete machine-readable evidence. Close/reopen the app and open the saved run to verify SQLite persistence.

## What our converter adds

The new algorithm is **calibration-guided fidelity selection**, built on PyTorch's ONNX exporter and ONNX Runtime. It evaluates:

1. Standard optimized export, runtime optimization enabled.
2. Export with graph optimization disabled, runtime optimization enabled.
3. Export with graph optimization disabled, runtime optimization disabled.

It chooses the fewest calibration samples outside tolerance, then the lowest maximum absolute output error, then the lowest MAE. Exact ties keep the standard configuration. The selected configuration is frozen before measuring the held-out set. The report includes all candidates, tolerances, selected settings and total search cost. Downloaded ONNX artifacts retain their required runtime optimization setting in report metadata; use that setting when reproducing results.

This is a new selection strategy, not a new low-level compiler. It can improve fidelity on some models, but an actual held-out run may be equal or worse. Accuracy, numerical drift and timing are separate claims. No test result is adjusted to force a positive difference.

A difference of **0.001 percentage points in top-1 accuracy** requires at least 100,000 test images for even a one-prediction step. This prototype accepts up to 200, so it cannot resolve that step. A **0.001 logit error** or **0.001 ms latency** is a different quantity; tiny latency changes can be noise. Use real validation data and independent repetitions before making review claims.

## ESP32: perform the real hardware test

1. Run an uploaded TFLite model first. For a classic ESP32 start with a small MCU-compatible classifier, typically quantized. A desktop ImageNet network may not fit. ONNX cannot directly run in TFLite Micro.
2. Open **Edge hardware** on that completed run. Select the TFLite artifact and a tensor-arena budget, then **Prepare & save ZIP**. The package contains the exact model, one preprocessed input, hashes, supported-kernel registrations, ESP-IDF source and instructions.
3. Install the official ESP-IDF 5.3–5.5 toolchain. Confirm the chip printed on your board or use Espressif's chip identification tooling. `ESP32`, `ESP32-S3` and `ESP32-C3` are different targets. A COM port alone does not identify the chip.
4. In an ESP-IDF terminal in the extracted package, for a confirmed classic ESP32:

```text
idf.py set-target esp32
idf.py build
idf.py -p COM_PORT flash
```

Replace `COM_PORT` with the detected port. **Flash replaces the current board firmware.** EdgeLens never flashes automatically. Build/AllocateTensors/Invoke errors mean compatibility is not established; reduce the model or adjust memory within the board's actual budget.

5. Close other serial monitors. In EdgeLens choose the saved package, **Scan USB ports**, select the port and **Capture USB report**. Capture waits 12 seconds; firmware repeats reports. Opening a serial port can reset some boards. The application checks package/model/input hashes and expected counts before saving the observation.
6. Compare host and board results in the report. Board timing covers `Invoke()` only; the output check covers one image. It is not a camera pipeline or full-dataset edge accuracy measurement. Tensor-arena usage is not total RAM or peak memory. Imported JSON is explicitly marked as imported. There is no cryptographic hardware attestation.

**Current connection check:** the USB serial scan returned no COM ports in this session. Check that the cable carries data and that the board's USB-UART driver is installed. No board was flashed and no physical latency has been measured yet.

Raspberry Pi remains a selectable deployment goal; a Pi hardware runner is future work.

## Optional Edge Impulse

On the same hardware page, enter your Edge Impulse project ID, supported target MCU identifier and project API key. The identifier is the `mcu` value from your project's `latencyDevices` API data; do not assume a generic `esp32` string is accepted. Confirm the explicit upload checkbox and submit the selected TFLite model. Fetch the completed job after processing. Network access, project permissions and supported targets depend on your account.

Only that explicit action sends the model to `studio.edgeimpulse.com`. The key stays in the open UI session/request memory and is not saved in SQLite or reports. Results retain provider provenance and do not substitute for measurements on your board. Integration behavior is tested with mocked provider replies; a live account call is still unverified.

## Use it as a command-line developer tool

The same engine can run without Electron or an HTTP server:

```powershell
backend\.venv\Scripts\python.exe backend\cli.py --model my-classifier.pt2 --spec model-spec.json --dataset test-images.zip --calibration calibration-images.zip --format onnx --target esp32 --output local-experiments --trust-own-model
```

Example `model-spec.json` (replace values with your actual training settings):

```json
{"name":"My classifier","format":"pt2","input_shape":[1,3,224,224],"layout":"NCHW","class_count":2,"scale":0.003921568627451,"mean":[0,0,0],"std":[1,1,1],"resize":"stretch"}
```

The CLI writes a SQLite experiment record, artifacts and HTML/CSV/JSON reports into the output directory. Omit `--calibration` for fixed profiles or imported-model benchmarking.

## Primary references

- [PyTorch export serialization and trust warning](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export/api_reference.html)
- [Espressif TFLite Micro 1.4.0](https://components.espressif.com/components/espressif/esp-tflite-micro/versions/1.4.0/readme)
- [Official ESP-IDF setup](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/)
- [Edge Impulse profile request](https://docs.edgeimpulse.com/apis/studio/jobs/profile-tflite-model)
- [Edge Impulse profile result](https://docs.edgeimpulse.com/apis/studio/jobs/get-tflite-profile-result-get)
