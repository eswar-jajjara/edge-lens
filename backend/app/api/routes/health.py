from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "edgelens-api", "mode": "benchmark"}


@router.get("/capabilities")
def capabilities(request: Request) -> dict:
    runtimes = request.app.state.capability_provider()
    return {
        "model_execution": any(item.get("available", False) for item in runtimes.values()),
        "artifact_uploads": True, "persistent_runs": True, "runtimes": runtimes,
        "models": [{"id": "mobilenet_v2", "name": "MobileNetV2"}, {"id": "resnet18", "name": "ResNet18"}],
        "targets": [{"id": "raspberry_pi", "name": "Raspberry Pi"}, {"id": "esp32", "name": "ESP32"}],
        "benchmark_location": "server_cpu",
        "custom_model_uploads": request.app.state.config.allow_custom_models,
        "imported_tflite_execution": __import__("importlib.util", fromlist=["find_spec"]).find_spec("ai_edge_litert") is not None,
        "hardware_benchmarks": True,
        "supported_uploads": ["pt2", "onnx", "tflite"],
    }
