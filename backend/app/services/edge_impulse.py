"""Optional explicit model upload to the official Edge Impulse profiling API."""
import base64
import hashlib
import re
import math
from pathlib import Path
import httpx

BASE = "https://studio.edgeimpulse.com/v1/api"


def redact_response(value, key):
    """Preserve provider response structure with credentials/request bytes removed."""
    forbidden = {"apikey", "xapikey", "authorization", "password", "token", "secret", "tflitefilebase64", "modelbytes"}
    if isinstance(value, dict):
        return {k: redact_response(v, key) for k, v in value.items()
                if str(k).lower().replace("-", "").replace("_", "") not in forbidden}
    if isinstance(value, list):
        return [redact_response(v, key) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Edge Impulse returned a non-finite numeric value")
    return value.replace(key, "[REDACTED]") if isinstance(value, str) and key else value


def call_api(method, path, key, body=None):
    try:
        with httpx.Client(timeout=45, follow_redirects=False) as client:
            response = client.request(method, BASE + path, headers={"x-api-key": key}, json=body)
        if response.status_code != 200:
            raise ValueError(f"Edge Impulse returned HTTP {response.status_code}. Check the project, key and profiling access.")
        if len(response.content) > 2 * 1024 * 1024:
            raise ValueError("Edge Impulse response exceeds the size limit")
        try:
            result = response.json()
        except ValueError as exc:
            raise ValueError("Edge Impulse returned an invalid JSON response") from exc
        if not isinstance(result, dict) or result.get("success") is not True:
            # Never return arbitrary remote error strings that could echo credentials.
            raise ValueError("Edge Impulse did not return a successful result. The job may still be running; check Studio and retry.")
        return redact_response(result, key)
    except httpx.HTTPError as exc:
        raise ValueError("Could not reach Edge Impulse securely. Check network access and try again.") from exc


def start_profile(path: Path, project_id: int, device: str, key: str, *, expected_sha256=None):
    if path.stat().st_size > 20 * 1024 * 1024:
        raise ValueError("Edge Impulse upload limit in this prototype is 20 MiB")
    content = path.read_bytes()
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Edge Impulse upload limit in this prototype is 20 MiB")
    if expected_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Upload bytes no longer match the evaluated artifact hash")
    result = call_api("POST", f"/{project_id}/jobs/profile-tflite", key,
                      {"tfliteFileBase64": base64.b64encode(content).decode("ascii"), "device": device})
    if type(result.get("id")) is not int:
        raise ValueError("Edge Impulse returned no job ID")
    return result["id"]


def list_targets(project_id, key):
    result = call_api("GET", f"/{project_id}", key)
    rows = result.get("latencyDevices")
    if not isinstance(rows, list):
        raise ValueError("Edge Impulse returned no profiling target list")
    return [{"mcu": r["mcu"], "name": r["name"]} for r in rows
            if isinstance(r, dict) and isinstance(r.get("mcu"), str) and isinstance(r.get("name"), str)
            and re.fullmatch(r"[a-zA-Z0-9_.+ -]{1,120}", r["mcu"]) and len(r["name"]) <= 200]


def list_projects(key):
    result = call_api("GET", "/projects", key)
    rows = result.get("projects")
    if not isinstance(rows, list):
        raise ValueError("Edge Impulse returned no accessible project list")
    projects = {}
    for row in rows:
        if (isinstance(row, dict) and type(row.get("id")) is int and row["id"] > 0
                and isinstance(row.get("name"), str) and 0 < len(row["name"]) <= 200):
            projects[row["id"]] = {"id": row["id"], "name": row["name"]}
    return list(projects.values())


def profile_result(project_id, job_id, key):
    result = call_api("GET", f"/{project_id}/jobs/profile-tflite/{job_id}/result", key)
    return result
