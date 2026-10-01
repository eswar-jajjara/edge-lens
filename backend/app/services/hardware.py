"""Generate reproducible ESP-IDF jobs and validate device-reported observations."""
import hashlib
import io
import json
from pathlib import Path
import time
from uuid import uuid4
import zipfile
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from app.services.metrics import percentile

OPS = {"CONV_2D": "Conv2D", "DEPTHWISE_CONV_2D": "DepthwiseConv2D", "FULLY_CONNECTED": "FullyConnected",
       "AVERAGE_POOL_2D": "AveragePool2D", "MAX_POOL_2D": "MaxPool2D", "SOFTMAX": "Softmax", "RESHAPE": "Reshape",
       "ADD": "Add", "MUL": "Mul", "RELU": "Relu", "RELU6": "Relu6", "QUANTIZE": "Quantize", "DEQUANTIZE": "Dequantize",
       "MEAN": "Mean", "PAD": "Pad", "CONCATENATION": "Concatenation", "TRANSPOSE": "Transpose", "LOGISTIC": "Logistic",
       "RESIZE_NEAREST_NEIGHBOR": "ResizeNearestNeighbor", "RESIZE_BILINEAR": "ResizeBilinear"}


class DeviceObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    protocol: Literal["edgelens.esp32.v1"]
    package_id: str = Field(pattern=r"^pkg_[a-f0-9]{32}$")
    model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chip: str = Field(pattern=r"^esp32[a-z0-9-]*$", max_length=32)
    idf_version: str = Field(min_length=1, max_length=120)
    cpu_freq_mhz: int = Field(ge=1, le=600)
    arena_used_bytes: int = Field(ge=1, le=16 * 1024 * 1024)
    arena_capacity_bytes: int = Field(ge=1, le=16 * 1024 * 1024)
    warmup_runs: int = Field(ge=1, le=50)
    samples_us: list[float] = Field(min_length=3, max_length=200)
    output: list[float] = Field(min_length=2, max_length=1000)


def _array(name, content):
    return f"alignas(16) const unsigned char {name}[] = {{" + ",".join(str(x) for x in content) + "};\n"


def create_package(run, artifact_index, dataset, arena_kib=96):
    from app.services.developer_benchmark import load_images, tflite_session
    from app.services.benchmark import _hash_file
    from ai_edge_litert import schema_py_generated as schema
    report = run["report"]
    artifacts = report["artifacts"]
    if not 0 <= artifact_index < len(artifacts) or artifacts[artifact_index]["format"] != "tflite":
        raise ValueError("Choose a TFLite artifact from the completed run")
    artifact = artifacts[artifact_index]
    path = Path(artifact["path"])
    if _hash_file(path) != artifact["sha256"]:
        raise ValueError("Artifact checksum changed")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("This ESP32 firmware workflow accepts models up to 2 MiB. A smaller model may still be needed for your board.")
    spec = report["model"]
    if "layout" not in spec:
        raise ValueError("Upload a TFLite classifier with explicit image preprocessing before preparing ESP32 firmware")
    if spec["class_count"] > 1000:
        raise ValueError("The serial benchmark protocol currently supports up to 1,000 classes")
    # Only one image is required; avoid loading the complete validation set.
    first_dataset = dict(dataset, entries=dataset["entries"][:1])
    inputs, _ = load_images(first_dataset, spec)
    infer, interpreter, quantize = tflite_session(path, 1)
    expected = infer(inputs[0]).reshape(-1).tolist()
    raw_input = quantize(inputs[0]).tobytes()
    if len(raw_input) > 200 * 1024:
        raise ValueError("Input tensor exceeds the ESP32 package's 200 KiB limit")
    model_bytes = path.read_bytes()
    graph = schema.Model.GetRootAsModel(model_bytes, 0)
    if graph.SubgraphsLength() != 1:
        raise ValueError("This ESP32 generator supports single-subgraph classifiers only")
    codes = {value: name for name, value in vars(schema.BuiltinOperator).items() if isinstance(value, int)}
    used_codes = [graph.OperatorCodes(graph.Subgraphs(0).Operators(i).OpcodeIndex()) for i in range(graph.Subgraphs(0).OperatorsLength())]
    # Older FlatBuffers can contain only the deprecated byte field.
    used = sorted({codes.get(max(op.BuiltinCode(), op.DeprecatedBuiltinCode()), "UNKNOWN") for op in used_codes})
    unsupported = sorted(set(used) - OPS.keys())
    if unsupported:
        raise ValueError("ESP32 firmware generator has no registered Micro kernel for: " + ", ".join(unsupported))
    package_id = "pkg_" + uuid4().hex
    settings = report["settings"]
    metadata = {"id": package_id, "run_id": run["id"], "artifact_index": artifact_index, "profile": artifact["profile"],
                "model_sha256": artifact["sha256"], "input_sha256": hashlib.sha256(raw_input).hexdigest(),
                "model_size_bytes": len(model_bytes), "input_size_bytes": len(raw_input), "operators": used,
                "expected_output": expected, "warmup_runs": settings["warmup_runs"], "measured_runs": settings["measured_runs"],
                "arena_capacity_bytes": arena_kib * 1024, "atol": settings["atol"], "rtol": settings["rtol"],
                "image": dataset["entries"][0]["path"], "input_shape": spec["input_shape"],
                "notes": "Build and run on your exact chip. Package generation is not a hardware measurement."}
    resolver = "\n".join(f"  if (resolver.Add{OPS[op]}() != kTfLiteOk) {{ fail(\"register {op}\"); return; }}" for op in used)
    cpp = FIRMWARE.replace("@RESOLVER@", resolver).replace("@OP_COUNT@", str(max(1, len(used))))
    for key, value in {"PACKAGE": package_id, "MODEL_SHA": metadata["model_sha256"], "INPUT_SHA": metadata["input_sha256"],
                       "ARENA": arena_kib * 1024, "WARMUP": settings["warmup_runs"], "RUNS": settings["measured_runs"]}.items():
        cpp = cpp.replace("@" + key + "@", str(value))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("CMakeLists.txt", 'cmake_minimum_required(VERSION 3.16)\ninclude($ENV{IDF_PATH}/tools/cmake/project.cmake)\nproject(edgelens_benchmark)\n')
        archive.writestr("main/CMakeLists.txt", 'idf_component_register(SRCS "main.cc" INCLUDE_DIRS "." REQUIRES esp_timer esp_system esp_hw_support freertos)\n')
        archive.writestr("main/idf_component.yml", 'dependencies:\n  idf: ">=5.3,<6.0"\n  espressif/esp-tflite-micro: "1.4.0"\n')
        archive.writestr("main/main.cc", cpp)
        archive.writestr("main/data.h", "#pragma once\n" + _array("model_data", model_bytes) + _array("input_data", raw_input))
        archive.writestr("manifest.json", json.dumps(metadata, indent=2))
        archive.writestr("sdkconfig.defaults", "CONFIG_ESP_MAIN_TASK_STACK_SIZE=8192\nCONFIG_PARTITION_TABLE_SINGLE_APP_LARGE=y\n")
        archive.writestr("README.md", README)
    return metadata, stream.getvalue()


def validate_observation(observation, package, source):
    import numpy as np
    obs = DeviceObservation.model_validate(observation).model_dump()
    for key in ("model_sha256", "input_sha256", "warmup_runs", "arena_capacity_bytes"):
        if obs[key] != package[key]:
            raise ValueError(f"Device {key} does not match the prepared package")
    if obs["package_id"] != package["id"] or len(obs["samples_us"]) != package["measured_runs"]:
        raise ValueError("Device package ID or sample count does not match")
    if obs["arena_used_bytes"] > obs["arena_capacity_bytes"] or any(not 0 < x < 600_000_000 for x in obs["samples_us"]):
        raise ValueError("Device reported invalid arena usage or timing samples")
    expected, actual = np.asarray(package["expected_output"]), np.asarray(obs["output"])
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("Device output shape or values are invalid")
    milliseconds = [x / 1000 for x in obs["samples_us"]]
    return {"source": source, "measurement_scope": "physical_device_report", "device_report": obs,
            "profile": package["profile"], "artifact_index": package["artifact_index"], "model_sha256": package["model_sha256"],
            "latency_p50_ms": percentile(milliseconds, 50), "latency_p95_ms": percentile(milliseconds, 95),
            "latency_samples_ms": milliseconds, "arena_used_bytes": obs["arena_used_bytes"], "model_size_bytes": package["model_size_bytes"],
            "output_mae": float(np.abs(expected - actual).mean()), "output_max_abs": float(np.abs(expected - actual).max()),
            "within_tolerance": bool(np.allclose(actual, expected, atol=package["atol"], rtol=package["rtol"])),
            "single_image_prediction_agreement": bool(expected.argmax() == actual.argmax()),
            "notes": ["Device-reported Invoke timings; no cryptographic hardware attestation.",
                      "One fixed preprocessed image, excluding camera capture and preprocessing. This does not measure dataset accuracy.",
                      "Tensor arena usage is not total RAM or peak memory. Host and device timings use different execution environments."]}


def serial_ports():
    from serial.tools import list_ports
    return [{"port": p.device, "description": p.description, "vid": p.vid, "pid": p.pid} for p in list_ports.comports()]


def capture_serial(port, package_id, timeout=12):
    import serial
    if port not in [p["port"] for p in serial_ports()]:
        raise ValueError("Port is not currently enumerated. Check the USB data cable and board USB driver.")
    device = serial.Serial(port=None, baudrate=115200, timeout=.2)
    device.dtr, device.rts = False, False
    device.port = port
    buffer = b""
    deadline = time.monotonic() + timeout
    with device:
        while time.monotonic() < deadline:
            buffer += device.read(min(device.in_waiting or 1, 4096))
            if len(buffer) > 200000:
                raise ValueError("Serial message exceeds the protocol limit")
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.startswith(b"EDGELENS_ERROR:"):
                    raise ValueError(line.decode("utf-8", errors="replace")[:300])
                if line.startswith(b"EDGELENS_RESULT:"):
                    observation = json.loads(line[len(b"EDGELENS_RESULT:"):])
                    if observation.get("package_id") == package_id:
                        return observation
    raise ValueError("No matching benchmark report arrived. Flash this package, close other serial monitors, and retry capture.")


README = """# EdgeLens ESP32 benchmark firmware

This package contains your exact TFLite artifact and one preprocessed test image.
It is a timing and output-fidelity probe, not a camera application or full accuracy test.

1. Install the official ESP-IDF 5.3–5.5 toolchain and open its terminal.
2. Confirm your chip variant. For a classic ESP32 DevKit/WROOM, run `idf.py set-target esp32`.
   For an S3/C3 use the corresponding actual target. USB connection alone does not identify it.
3. Run `idf.py build`. The managed component downloads Espressif TFLite Micro 1.4.0.
4. Review the build size. Run `idf.py -p COM_PORT flash` only when ready to replace the board's current firmware.
5. Close the serial monitor. In EdgeLens select the matching saved package and port, then Capture USB report.
   Firmware repeats reports. Opening a serial port can reset some boards despite disabled DTR/RTS.

No board is flashed automatically. If no COM port appears, check the USB data cable and board's USB-UART driver.
If AllocateTensors fails, the configured tensor arena is too small or memory is unavailable.
Increase the arena only within the actual board's memory budget. A package or supported-op list does not prove that the model fits.
Use a small, quantized classifier for a classic ESP32; do not assume a desktop ImageNet model will fit.

The manifest records model/input hashes, arena size, warm-up and run counts. EdgeLens rejects mismatches.
Timing covers Invoke only, excludes serial output, input copy and preprocessing, and yields to the scheduler between calls.
Arena usage is not total or peak RAM. Results include a single output vector for checking against host TFLite.
There is no cryptographic hardware attestation: imported JSON is marked separately from USB capture.

Sources: https://github.com/espressif/esp-tflite-micro and https://docs.espressif.com/projects/esp-idf/en/stable/esp32/get-started/
"""

FIRMWARE = r'''#include <cstdio>
#include <cstring>
#include <cmath>
#include "sdkconfig.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "esp_private/esp_clk.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "data.h"
alignas(16) static uint8_t arena[@ARENA@];
static void fail(const char* reason) { printf("EDGELENS_ERROR:%s\n", reason); }
extern "C" void app_main() {
  const auto* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) { fail("TFLite schema mismatch"); return; }
  static tflite::MicroMutableOpResolver<@OP_COUNT@> resolver;
@RESOLVER@
  static tflite::MicroInterpreter interpreter(model, resolver, arena, sizeof(arena));
  if (interpreter.AllocateTensors() != kTfLiteOk) { fail("AllocateTensors failed: check memory and supported kernels"); return; }
  auto* input = interpreter.input(0);
  auto* output = interpreter.output(0);
  if (input->bytes != sizeof(input_data)) { fail("Input byte count mismatch"); return; }
  if (output->type != kTfLiteInt8 && output->type != kTfLiteUInt8 && output->type != kTfLiteFloat32) { fail("Unsupported output dtype"); return; }
  int classes = 1;
  for (int i = 0; i < output->dims->size; ++i) classes *= output->dims->data[i];
  if (classes < 2 || classes > 1000) { fail("Unsupported class count"); return; }
  while (true) {
    int64_t samples[@RUNS@];
    for (int i = -@WARMUP@; i < @RUNS@; ++i) {
      memcpy(input->data.raw, input_data, sizeof(input_data));
      int64_t start = esp_timer_get_time();
      TfLiteStatus status = interpreter.Invoke();
      int64_t elapsed = esp_timer_get_time() - start;
      if (status != kTfLiteOk) { fail("Invoke failed"); return; }
      if (i >= 0) samples[i] = elapsed;
      vTaskDelay(1);
    }
    printf("EDGELENS_RESULT:{\"protocol\":\"edgelens.esp32.v1\",\"package_id\":\"@PACKAGE@\",\"model_sha256\":\"@MODEL_SHA@\",\"input_sha256\":\"@INPUT_SHA@\",\"chip\":\"%s\",\"idf_version\":\"%s\",\"cpu_freq_mhz\":%d,\"arena_used_bytes\":%u,\"arena_capacity_bytes\":%u,\"warmup_runs\":@WARMUP@,\"samples_us\":[", CONFIG_IDF_TARGET, esp_get_idf_version(), esp_clk_cpu_freq()/1000000, (unsigned)interpreter.arena_used_bytes(), (unsigned)sizeof(arena));
    for (int i = 0; i < @RUNS@; ++i) printf("%s%lld", i ? "," : "", (long long)samples[i]);
    printf("],\"output\":[");
    for (int i = 0; i < classes; ++i) {
      float value = output->type == kTfLiteFloat32 ? output->data.f[i] :
          ((output->type == kTfLiteInt8 ? output->data.int8[i] : output->data.uint8[i]) - output->params.zero_point) * output->params.scale;
      if (std::isfinite(value)) printf("%s%.9g", i ? "," : "", (double)value);
      else printf("%snull", i ? "," : "");
    }
    printf("]}\n"); fflush(stdout);
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}
'''
