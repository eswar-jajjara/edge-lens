"""Recoverable BYOM upload/profile workflow; no claims about converted accuracy."""
import hashlib
import json
from threading import Lock
from app.services import edge_impulse

# One transition per project at a time in this single-engine desktop prototype.
_guard = Lock()
_project_locks = {}


class ProfilePending(ValueError):
    pass


def _identity(response, digest):
    model = response.get("model")
    if not isinstance(model, dict) or model.get("fileName") != edge_impulse.onnx_filename(digest):
        raise ValueError("The project's uploaded model changed. This estimate cannot be attached to the selected ONNX artifact.")
    # A filename is a association check, not a cryptographic identity guarantee.
    data = {"fileName": model["fileName"], "modelInfo": response.get("modelInfo")}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def advance(repository, job, key):
    with _guard:
        lock = _project_locks.setdefault(job["project_id"], Lock())
    if not lock.acquire(blocking=False):
        raise ProfilePending("The job may still be running; another refresh is already in progress.")
    try:
        job = repository.get_edge_record(job["id"])
        phase = job.get("phase")
        if phase == "complete":
            return job["completed_response"], job
        if phase in {"upload_unknown", "profile_unknown", "upload_submitting", "profile_submitting"}:
            raise ValueError("A previous ONNX submission has no confirmed receipt. Check Studio before explicitly starting another upload; EdgeLens will not retry it automatically.")
        if phase == "failed":
            raise ValueError("This ONNX provider job failed. Check Studio before trying another model.")
        if phase == "upload_pending":
            try:
                finished = edge_impulse.job_finished(job["project_id"], job["job_id"], key)
            except ValueError as exc:
                if "could not process" in str(exc):
                    repository.update_edge_record(job["id"], {"phase": "failed"})
                raise
            if not finished:
                raise ProfilePending("The job may still be running; ONNX upload/conversion is in progress.")
            uploaded = edge_impulse.pretrained_model(job["project_id"], key)
            try:
                identity = _identity(uploaded, job["model_sha256"])
            except ValueError:
                repository.update_edge_record(job["id"], {"phase": "failed"})
                raise
            repository.update_edge_record(job["id"], {"phase": "profile_submitting", "provider_identity_sha256": identity})
            try:
                profile_id = edge_impulse.start_pretrained_profile(job["project_id"], key)
            except ValueError:
                repository.update_edge_record(job["id"], {"phase": "profile_unknown"})
                raise
            job = repository.update_edge_record(job["id"], {"phase": "profiling", "profile_job_id": profile_id})
            raise ProfilePending("The job may still be running; the ONNX-derived profiling job was submitted. Fetch its estimate after processing finishes.")
        try:
            finished = edge_impulse.job_finished(job["project_id"], job["profile_job_id"], key)
        except ValueError as exc:
            if "could not process" in str(exc):
                repository.update_edge_record(job["id"], {"phase": "failed"})
            raise
        if not finished:
            raise ProfilePending("The job may still be running; ONNX-derived model profiling is in progress.")
        result = edge_impulse.pretrained_model(job["project_id"], key)
        model = result.get("model", {})
        try:
            if _identity(result, job["model_sha256"]) != job["provider_identity_sha256"]:
                raise ValueError("The provider model metadata changed during profiling. Fetching this estimate would mix different models.")
            if model.get("profileJobFailed") is True:
                raise ValueError("Edge Impulse could not process this ONNX profiling job. Check Studio.")
            other_job = model.get("profileJobId")
            if other_job is not None and other_job != job["profile_job_id"]:
                raise ValueError("The project is profiling a different model job. This response is not linked to the selected upload.")
        except ValueError:
            repository.update_edge_record(job["id"], {"phase": "failed"})
            raise
        if not model.get("profileInfo"):
            raise ProfilePending("The job may still be running; no provider resource estimates are available yet.")
        # Cached sanitized response is immutable even if Studio replaces its model later.
        result = edge_impulse.redact_response(result, key)
        job = repository.update_edge_record(job["id"], {"phase": "complete", "completed_response": result})
        return result, job
    finally:
        lock.release()
