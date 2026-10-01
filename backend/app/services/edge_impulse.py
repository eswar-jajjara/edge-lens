"""Optional explicit model upload to the official Edge Impulse profiling API."""
import base64
from pathlib import Path
import httpx

BASE = "https://studio.edgeimpulse.com/v1/api"


def call_api(method, path, key, body=None):
    try:
        with httpx.Client(timeout=45, follow_redirects=False) as client:
            response = client.request(method, BASE + path, headers={"x-api-key": key}, json=body)
        if response.status_code != 200:
            raise ValueError(f"Edge Impulse returned HTTP {response.status_code}. Check the project, key and profiling access.")
        if len(response.content) > 2 * 1024 * 1024:
            raise ValueError("Edge Impulse response exceeds the size limit")
        result = response.json()
        if not isinstance(result, dict) or result.get("success") is not True:
            # Never return arbitrary remote error strings that could echo credentials.
            raise ValueError("Edge Impulse did not return a successful result. The job may still be running; check Studio and retry.")
        return result
    except httpx.HTTPError as exc:
        raise ValueError("Could not reach Edge Impulse securely. Check network access and try again.") from exc


def start_profile(path: Path, project_id: int, device: str, key: str):
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Edge Impulse upload limit in this prototype is 20 MiB")
    result = call_api("POST", f"/{project_id}/jobs/profile-tflite", key,
                      {"tfliteFileBase64": base64.b64encode(path.read_bytes()).decode("ascii"), "device": device})
    if type(result.get("id")) is not int:
        raise ValueError("Edge Impulse returned no job ID")
    return result["id"]


def profile_result(project_id, job_id, key):
    result = call_api("GET", f"/{project_id}/jobs/profile-tflite/{job_id}/result", key)
    # Store documented profile fields only; never store credentials or request bodies.
    keys = {"variant", "device", "tfliteFileSizeBytes", "isSupportedOnMcu", "hasPerformance", "memory", "timePerInferenceMs", "mcuSupportError", "profilingError"}
    return {k: v for k, v in result.items() if k in keys}
