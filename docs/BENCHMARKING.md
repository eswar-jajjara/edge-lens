# Benchmark methodology — version 0.4

Developer classifiers use the exact uploaded export, declared image preprocessing and model class order. Common torchvision presets remain optional. A PT2 source supplies a PyTorch reference. ONNX/TFLite-only uploads support standalone inference metrics; missing reference comparisons are null.

## Fidelity-guided ONNX conversion

The standard baseline uses the PyTorch dynamo exporter with `optimize=True` and ONNX Runtime CPU graph optimizations enabled. Alternatives use `optimize=False` with runtime graph optimization enabled/disabled. ONNX opsets and operator lowering follow the installed, recorded PyTorch exporter; the strategy does not invent unsupported graph transformations.

Separate calibration images rank the candidates by `(number of images outside tolerance, maximum absolute output error, mean absolute error)`. Exact ties choose standard. The winner's artifact and runtime settings are frozen before held-out evaluation. The standard candidate remains a valid outcome. Accuracy improvements are never assumed; output error, accuracy and latency have different meanings.

Total strategy time includes two exports, session creation and calibration execution. Candidate export times are retained separately. A single conversion run is affected by initialization/cache order and is not a rigorous speed comparison of compiler implementations.

## Held-out measurements

- Top-1 accuracy = correct predictions / labelled images. Delta is in percentage points. Agreement compares argmax with the PyTorch reference, not true labels.
- Numerical differences compare raw class-score vectors without adding a softmax. Tolerance uses `abs(candidate - reference) <= atol + rtol * abs(reference)` as implemented by NumPy's comparison call.
- CPU timing warms each profile, rotates profile order between rounds and records raw values, median and p95. One fixed preprocessed image is used. Loading, preprocessing, conversion and diagnostic extraction are outside the timing loop; runtime calls/output copy/signature checks are included for uploaded classifiers.
- All profiles use configured CPU thread counts, CPU execution and the same test images. Concurrent external workloads and OS scheduling can affect results. Independent repetitions are needed for tiny timing claims.
- One changed label prediction shifts accuracy by `100/N` percentage points. Current dataset limit is 200 images; a 0.001 pp accuracy step is outside this prototype's resolution.
- Serialized model bytes are not peak process RAM. No host peak-memory or energy claim is made.

## Per-operation evidence

Registered models have module-level leaf diagnostics. Custom PT2 exports have FX-operation entries, including functional operations. Optimized/fused nodes without a proven one-to-one mapping remain unmapped. An additional unoptimized diagnostic profile enables conservative exact-node matching regardless of the selected deployment artifact.

Captures are limited to 64 MiB and one test image. Intermediate ONNX diagnostic outputs disable runtime graph optimization and can affect fusion. They are not interchangeable with optimized execution. Final-output fidelity and per-image predictions are obtained from the actual benchmark profiles. A drift at a boundary is an observation, not an automatic root-cause diagnosis. TFLite intermediate-to-PyTorch alignment remains unavailable; imported TFLite shows tensor inventory and quantization metadata.

## Device and provider evidence

ESP32 firmware times only TFLite Micro `Invoke()` and records raw microsecond samples, configured warm-ups, observed CPU clock, IDF version, chip target and arena usage. The exact model and one input are embedded. Results must match package/model/input identity and sample counts before they are attached to the run. Captured and imported reports retain distinct provenance. Hash matching is not cryptographic proof of physical hardware execution.

One-image board output can be compared with the host TFLite result. It does not establish full-dataset edge accuracy. Arena usage is not total/peak RAM, flash partition occupancy or energy consumption. Package generation does not prove compatibility or fit.

Edge Impulse profiling is optional and asynchronous. Its resource/timing analysis is shown as provider analysis, never as a measurement of the user's connected ESP32. No fake hardware latency is produced when a board or provider is unavailable.

TFLite conversion remains an optional Linux LiteRT Torch path with identical FP32 controls, not a distinct optimized TFLite converter. Imported FP32/int8 TFLite inference works in the current Windows engine. Quantization training/calibration and Raspberry Pi hardware execution are future work.
