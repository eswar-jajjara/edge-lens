from pathlib import Path
import json
from zipfile import BadZipFile
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, FileResponse
from app.schemas.runs import CreateRunRequest
from app.services.validation import QueueFull
from app.services.reports import report_csv, report_html
from app.services.evidence import diagnostic_groups, evidence_sections

router = APIRouter()


@router.post('/reports/comparison')
async def create_comparison(request: Request):
    if not request.app.state.config.allow_custom_models:
        raise HTTPException(403, 'Report comparison is available in the local developer tool.')
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 24 * 1024 * 1024:
            raise HTTPException(413, 'Comparison input exceeds 24 MiB. Select fewer exported reports.')
        chunks.append(chunk)
    from app.services.report_comparison import build_comparison, comparison_html, comparison_csv
    try:
        payload = json.loads(b''.join(chunks), parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Non-finite input')))
        if not isinstance(payload, dict) or set(payload) - {'reports', 'references', 'title'}:
            raise ValueError('Use reports, optional references and an optional title.')
        result = build_comparison(payload.get('reports'), payload.get('references'), payload.get('title', 'EdgeLens conversion comparison'))
        return {'report': result, 'html': comparison_html(result), 'csv': comparison_csv(result)}
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (KeyError, TypeError, AttributeError):
        raise HTTPException(422, 'Use original measured EdgeLens report exports and the documented reference JSON format.') from None


@router.post('/diagnostics/self-test', status_code=201)
def diagnostic_self_test(request: Request):
    if not request.app.state.config.allow_custom_models:
        raise HTTPException(403, 'Diagnostic self-tests are available in the local developer workspace only')
    repository = request.app.state.repository
    if any(run['status'] in {'queued', 'running'} for run in repository.list_runs()):
        raise HTTPException(409, 'Wait for active benchmarks to finish before running the diagnostic self-test')
    from app.services.fault_validation import run_fault_suite
    from app.services.benchmark import _artifact
    run = repository.create_run({'model_id': 'controlled_fault_self_test', 'format': 'onnx', 'target': 'esp32', 'dataset_id': None,
                                 'strategy': 'controlled_faults', 'settings': {'threads': 1}, 'edge_estimate': {'enabled': False}})
    directory = repository.data_dir / 'runs' / run['id']
    try:
        repository.update_run(run['id'], 'running')
        faults = run_fault_suite(directory)
        report = {'schema_version': 5, 'source': 'measured', 'run_id': run['id'], 'created_at': run['created_at'],
                  'model': {'name': 'Controlled diagnostic self-test · synthetic classifier'},
                  'summary': {'conclusion': f"{sum(c['passed'] for c in faults['cases'])}/{len(faults['cases'])} known-fault/control checks passed."},
                  'fault_validation': faults, 'metrics': [], 'predictions': [],
                  'structural': [{'profile': case['fault'], 'comparison': case['structural'], 'comparison_status': 'MEASURED',
                                  'reason': 'Known synthetic fault/control against the unchanged reference.'} for case in faults['cases'] if case.get('structural')],
                  'layers': [dict(row, profile=case['fault'], reason_code='controlled_fault_capture', first_observed_divergence=row['name'] == case['first_observed_divergence'])
                             for case in faults['cases'] for row in case.get('numerical', [])],
                  'artifacts': [_artifact(p.stem, p.suffix.removeprefix('.'), p) for p in directory.iterdir() if p.suffix in {'.onnx', '.json'}],
                  'edge_results': [], 'edge_estimate_request': {'enabled': False},
                  'environment': {'benchmark_scope': 'synthetic_diagnostic_validation', 'settings': {'threads': 1}},
                  'methodology': ['Deliberate faults in a disposable synthetic classifier; original uploaded models are never modified.'],
                  'limitations': [faults['scope'], faults['limitation'], 'No dataset-accuracy, latency, physical-device or provider-resource claim from this self-test.']}
        repository.update_run(run['id'], 'completed', report=report)
        return public_run(repository.get_run(run['id']))
    except Exception as exc:
        repository.update_run(run['id'], 'failed', error=f'Diagnostic self-test failed: {type(exc).__name__}: {str(exc)[:250]}')
        raise HTTPException(503, 'Diagnostic self-test could not complete. Check that ONNX and ONNX Runtime are installed.') from exc


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
    if custom and custom['format'] == 'pt2' and payload.format == 'tflite':
        calibration = repository.get_dataset(payload.calibration_dataset_id) if payload.calibration_dataset_id else None
        if calibration is None or calibration['sha256'] == dataset['sha256']:
            raise HTTPException(422, 'TFLite INT8 conversion requires separate calibration images in the Linux worker')
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


@router.post('/worker-reports', status_code=201)
async def import_worker_report(request: Request):
    if not request.app.state.config.allow_custom_models:
        raise HTTPException(403, 'Worker imports are available in the local developer workspace only')
    from app.services.worker_bundle import import_bundle, MAX_BYTES
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_BYTES: raise HTTPException(413, 'Worker bundle exceeds 100 MiB')
    try:
        return public_run(import_bundle(bytes(content), request.app.state.repository))
    except (ValueError, KeyError, TypeError, OSError, BadZipFile) as exc:
        raise HTTPException(422, 'Worker bundle is invalid or its evaluated artifact hashes do not match.') from exc


@router.get("/runs")
def list_runs(request: Request):
    return [{**public_run(run), "report": None} for run in request.app.state.repository.list_runs()]


@router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    return public_run(required_run(request, run_id))


@router.delete("/runs/{run_id}")
def delete_run(run_id: str, request: Request):
    if not request.app.state.config.allow_custom_models:
        raise HTTPException(403, "History deletion is enabled in the local developer workspace only")
    try:
        return request.app.state.repository.delete_run(run_id)
    except KeyError as exc:
        raise HTTPException(404, "Run not found.") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(409, "Could not remove this test's files. Close open artifact files and try again.") from exc


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
