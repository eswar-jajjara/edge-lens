from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, FileResponse
from app.schemas.runs import CreateRunRequest
from app.services.validation import QueueFull
from app.services.reports import report_csv, report_html
from app.services.evidence import diagnostic_groups, evidence_sections

router = APIRouter()


def required_run(request, run_id):
    run = request.app.state.repository.get_run(run_id)
    if run is None:
        raise HTTPException(404, "Run not found.")
    if run.get("report"):
        run["report"]["edge_results"] = [x for x in request.app.state.repository.edge_records(run_id) if x["kind"] in ("hardware", "edge_impulse_result")]
    return run


def public_report(report, run_id):
    value = dict(report)
    value["evidence_sections"] = evidence_sections(report)
    value["diagnostic_summary"] = [{k: v for k, v in g.items() if k not in {"measured", "inventory", "unavailable"}} for g in diagnostic_groups(report)]
    value["artifacts"] = [
        {**{key: item for key, item in artifact.items() if key != "path"},
         "download_url": f"/api/v1/runs/{run_id}/artifacts/{index}"}
        for index, artifact in enumerate(report.get("artifacts", []))
    ]
    if isinstance(value.get("dataset"), dict):
        value["dataset"] = {k: v for k, v in value["dataset"].items() if k not in {"path", "entries"}}
    return value


def public_run(run):
    return {**run, "report": public_report(run["report"], run["id"]) if run.get("report") else None}


def required_report(request, run_id):
    run = required_run(request, run_id)
    if run["status"] != "completed" or not run.get("report"):
        raise HTTPException(409, "A report is available only after the benchmark completes.")
    return public_report(run["report"], run_id)


@router.post("/runs", status_code=202)
def submit_run(payload: CreateRunRequest, request: Request):
    dataset = request.app.state.repository.get_dataset(payload.dataset_id)
    if dataset is None:
        raise HTTPException(404, "Upload a labelled dataset before starting the benchmark.")
    repository = request.app.state.repository
    custom = repository.get_model(payload.model_id)
    if custom is None and payload.model_id not in ("mobilenet_v2", "resnet18"):
        raise HTTPException(404, "Model not found. Upload an exported classifier first.")
    if custom and not request.app.state.config.allow_custom_models:
        raise HTTPException(403, "Developer model execution is disabled on this server")
    if custom and custom["format"] != "pt2" and custom["format"] != payload.format:
        raise HTTPException(422, "ONNX/TFLite uploads are benchmarked in their original format. Upload PT2 for conversion.")
    if payload.strategy == "fidelity_search":
        if not custom or custom["format"] != "pt2" or payload.format != "onnx":
            raise HTTPException(422, "Fidelity search currently requires an uploaded PT2 classifier and ONNX output")
        calibration = repository.get_dataset(payload.calibration_dataset_id) if payload.calibration_dataset_id else None
        if calibration is None or calibration["sha256"] == dataset["sha256"]:
            raise HTTPException(422, "Select a separate calibration dataset. Test data cannot select a converter.")
    if payload.strategy in {"quantization_compare", "deployment_search"}:
        if not custom or custom["format"] not in {"pt2", "onnx"}:
            raise HTTPException(422, "FP32/INT8 experiments require an uploaded PT2 or FP32 ONNX classifier")
        split_data = [dataset, repository.get_dataset(payload.calibration_dataset_id), repository.get_dataset(payload.validation_dataset_id)]
        if any(value is None for value in split_data):
            raise HTTPException(422, "Upload and select calibration, validation and test datasets")
        if len({value["sha256"] for value in split_data}) != 3:
            raise HTTPException(422, "Calibration, validation and test archives must be separate")
    runtime = request.app.state.capability_provider().get(payload.format, {})
    if custom and custom["format"] == "tflite":
        from app.services.benchmark import _present
        runtime = {"available": _present("ai_edge_litert"), "reason": "Install ai-edge-litert to benchmark imported TFLite files"}
    if not runtime.get("available", False):
        raise HTTPException(503, {"code": "RUNTIME_UNAVAILABLE", "message": runtime.get("reason", "Install the required model runtime on the server.")})
    try:
        return public_run(request.app.state.validation.submit(payload.model_dump(), dataset))
    except QueueFull as error:
        raise HTTPException(429, str(error)) from error


@router.get("/runs")
def list_runs(request: Request):
    return [{**public_run(run), "report": None} for run in request.app.state.repository.list_runs()]


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    return public_run(required_run(request, run_id))


@router.get("/runs/{run_id}/report")
def get_report(run_id: str, request: Request):
    return required_report(request, run_id)


@router.get("/runs/{run_id}/report.csv")
def get_csv(run_id: str, request: Request):
    content = report_csv(required_report(request, run_id))
    return Response(content, media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{run_id}-report.csv"'})


@router.get("/runs/{run_id}/report.html")
def get_html(run_id: str, request: Request):
    content = report_html(required_report(request, run_id))
    return Response(content, media_type="text/html", headers={"Content-Disposition": f'attachment; filename="{run_id}-report.html"'})


@router.get("/runs/{run_id}/artifacts/{artifact_index}")
def download_artifact(run_id: str, artifact_index: int, request: Request):
    run = required_run(request, run_id)
    artifacts = (run.get("report") or {}).get("artifacts", [])
    if run["status"] != "completed" or not 0 <= artifact_index < len(artifacts):
        raise HTTPException(404, "Artifact not found.")
    artifact = artifacts[artifact_index]
    file_path = Path(artifact["path"]).resolve()
    allowed = (request.app.state.config.data_dir / "runs" / run_id).resolve()
    if not file_path.is_relative_to(allowed) or not file_path.is_file():
        raise HTTPException(404, "Artifact is not available on this server.")
    return FileResponse(file_path, filename=file_path.name, media_type="application/octet-stream")
