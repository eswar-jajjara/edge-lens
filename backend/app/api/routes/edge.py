from pathlib import Path
import hashlib
import json
from typing import Literal
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from app.services import hardware, edge_impulse
from app.services.benchmark import _hash_file
from app.services.evidence import artifact_evaluation

router = APIRouter()


class PackageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_index: int = Field(ge=0)
    arena_kib: int = Field(default=96, ge=16, le=2048)


class CaptureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    port: str = Field(min_length=1, max_length=80)


class ProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_index: int = Field(ge=0)
    project_id: int = Field(ge=1, strict=True)
    device: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_.+ -]+$")
    api_key: SecretStr = Field(min_length=8, max_length=300)
    consent_upload: Literal[True]


class ProfileRefresh(BaseModel):
    model_config = ConfigDict(extra="forbid")
    api_key: SecretStr = Field(min_length=8, max_length=300)


class TargetRequest(ProfileRefresh):
    project_id: int = Field(ge=1, strict=True)


@router.post("/edge/impulse/targets")
def profile_targets(payload: TargetRequest, request: Request):
    require_local(request)
    try:
        return edge_impulse.list_targets(payload.project_id, payload.api_key.get_secret_value())
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc


def require_local(request):
    if not request.app.state.config.allow_custom_models:
        raise HTTPException(403, "Edge device access is enabled in the local developer workspace only")


def completed_run(request, run_id):
    run = request.app.state.repository.get_run(run_id)
    if not run or run["status"] != "completed" or not run["report"]:
        raise HTTPException(409, "Select a completed benchmark run")
    return run


def record(request, record_id, kind):
    result = request.app.state.repository.get_edge_record(record_id)
    if result is None or result["kind"] != kind:
        raise HTTPException(404, "Edge record not found")
    return result


def artifact_path(request, run, index):
    artifacts = run["report"].get("artifacts", [])
    if not 0 <= index < len(artifacts) or artifacts[index]["format"] != "tflite":
        raise HTTPException(422, "Select a TFLite artifact")
    artifact = artifacts[index]
    path = Path(artifact["path"]).resolve()
    root = (request.app.state.config.data_dir / "runs" / run["id"]).resolve()
    if not path.is_relative_to(root) or not path.is_file() or _hash_file(path) != artifact["sha256"]:
        raise HTTPException(409, "Artifact is missing or its checksum changed")
    return artifact, path


@router.get("/edge/ports")
def list_ports(request: Request):
    require_local(request)
    try:
        return hardware.serial_ports()
    except ImportError as exc:
        raise HTTPException(503, "Install pyserial for USB capture") from exc


@router.get("/runs/{run_id}/edge")
def list_edge_records(run_id: str, request: Request):
    completed_run(request, run_id)
    return [{k: v for k, v in item.items() if k not in {"path", "expected_output"}} for item in request.app.state.repository.edge_records(run_id)]


@router.post("/runs/{run_id}/edge/packages", status_code=201)
def prepare_package(run_id: str, payload: PackageRequest, request: Request):
    require_local(request)
    run = completed_run(request, run_id)
    artifact_path(request, run, payload.artifact_index)
    dataset = request.app.state.repository.get_dataset(run["request"]["dataset_id"])
    try:
        metadata, content = hardware.create_package(run, payload.artifact_index, dataset, payload.arena_kib)
    except ImportError as exc:
        raise HTTPException(503, "Install ai-edge-litert to prepare a firmware package") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    directory = request.app.state.config.data_dir / "runs" / run_id
    path = directory / f"{metadata['id']}.zip"
    path.write_bytes(content)
    metadata["path"] = str(path)
    value = request.app.state.repository.save_edge_record(run_id, "package", metadata, metadata["id"])
    return {k: v for k, v in value.items() if k not in {"path", "expected_output"}}


@router.get("/edge/packages/{package_id}/download")
def download_package(package_id: str, request: Request):
    item = record(request, package_id, "package")
    path = Path(item["path"]).resolve()
    root = (request.app.state.config.data_dir / "runs" / item["run_id"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(404, "Package file not found")
    return Response(path.read_bytes(), media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="edgelens-{package_id}.zip"'})


def save_observation(request, package, value, source):
    try:
        report = hardware.validate_observation(value, package, source)
    except ValueError as exc:
        raise HTTPException(422, str(exc)[:1200]) from exc
    return request.app.state.repository.save_edge_record(package["run_id"], "hardware", report)


@router.post("/edge/packages/{package_id}/capture", status_code=201)
def capture(package_id: str, payload: CaptureRequest, request: Request):
    require_local(request)
    package = record(request, package_id, "package")
    try:
        observation = hardware.capture_serial(payload.port, package_id)
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)[:500]) from exc
    except ImportError as exc:
        raise HTTPException(503, "Install pyserial for USB capture") from exc
    return save_observation(request, package, observation, "usb_serial_capture")


@router.post("/edge/packages/{package_id}/import", status_code=201)
def import_observation(package_id: str, payload: hardware.DeviceObservation, request: Request):
    require_local(request)
    return save_observation(request, record(request, package_id, "package"), payload.model_dump(), "imported_device_report")


@router.post("/runs/{run_id}/edge/impulse", status_code=201)
def submit_profile(run_id: str, payload: ProfileRequest, request: Request):
    require_local(request)
    run = completed_run(request, run_id)
    artifact, path = artifact_path(request, run, payload.artifact_index)
    evaluation = artifact_evaluation(run["report"], artifact)
    if evaluation["status"] != "MEASURED":
        raise HTTPException(409, evaluation["reason"])
    try:
        job_id = edge_impulse.start_profile(path, payload.project_id, payload.device, payload.api_key.get_secret_value(), expected_sha256=artifact["sha256"])
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc
    return request.app.state.repository.save_edge_record(run_id, "edge_impulse_job", {"project_id": payload.project_id, "job_id": job_id, "device": payload.device,
                  "artifact_index": payload.artifact_index, "profile": artifact["profile"], "model_sha256": artifact["sha256"],
                  "evaluated_artifact": evaluation, "source": "edge_impulse", "measurement_scope": "provider_analysis",
                  "evidence_status": "ESTIMATED", "provider_url": edge_impulse.BASE + f"/{payload.project_id}/jobs/profile-tflite"})


@router.post("/edge/impulse/{record_id}/refresh")
def refresh_profile(record_id: str, payload: ProfileRefresh, request: Request):
    require_local(request)
    job = record(request, record_id, "edge_impulse_job")
    run = completed_run(request, job["run_id"])
    artifact, _ = artifact_path(request, run, job["artifact_index"])
    if artifact["sha256"] != job["model_sha256"]:
        raise HTTPException(409, "Profile job artifact hash does not match this run")
    try:
        result = edge_impulse.profile_result(job["project_id"], job["job_id"], payload.api_key.get_secret_value())
    except ValueError as exc:
        raise HTTPException(502, str(exc)) from exc
    result = edge_impulse.redact_response(result, payload.api_key.get_secret_value())
    result_record = {k: job[k] for k in ("project_id", "job_id", "device", "artifact_index", "profile", "model_sha256")}
    result_record.update(evaluated_artifact=job.get("evaluated_artifact"), evidence_status="ESTIMATED",
                         provider_url=edge_impulse.BASE + f"/{job['project_id']}/jobs/profile-tflite/{job['job_id']}/result",
                         raw_response=result,
                         response_sha256=hashlib.sha256(json.dumps(result, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                         response_redaction="Credentials and request model bytes removed; otherwise provider response structure preserved.")
    result_record.update(source="edge_impulse", measurement_scope="provider_analysis", provider_result=result,
                         notes=["Edge Impulse resource/profile analysis. This is not a measurement from your USB-connected ESP32.", "Provider timings and memory estimates are not substituted for local host or physical-device metrics."])
    return request.app.state.repository.save_edge_record(job["run_id"], "edge_impulse_result", result_record)
