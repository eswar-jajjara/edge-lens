"""Run the same developer benchmark engine without a GUI or HTTP server."""
import argparse
import json
from pathlib import Path
from app.repositories import Repository
from app.services.models import ModelSpec, ingest_model
from app.services.datasets import ingest_dataset
from app.services.developer_benchmark import run_developer_benchmark
from app.schemas.runs import CreateRunRequest
from app.services.reports import report_csv, report_html
from app.api.routes.runs import public_report


def main():
    parser = argparse.ArgumentParser(description="Benchmark your trusted exported image classifier locally.")
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True, help='JSON preprocessing/class signature, matching ModelSpec')
    parser.add_argument('--dataset', type=Path, required=True, help='Held-out labelled image ZIP')
    parser.add_argument('--calibration', type=Path, help='Separate calibration ZIP enables ONNX fidelity search')
    parser.add_argument('--format', choices=['onnx', 'tflite'], required=True)
    parser.add_argument('--target', choices=['esp32', 'raspberry_pi'], default='esp32')
    parser.add_argument('--output', type=Path, required=True, help='Local experiment storage directory')
    parser.add_argument('--trust-own-model', action='store_true', help='Acknowledge this is your own trusted model; PT2 loading can execute serialized code')
    args = parser.parse_args()
    if not args.trust_own_model:
        parser.error('Use --trust-own-model only for an export you trust.')
    spec = ModelSpec.model_validate(json.loads(args.spec.read_text(encoding='utf-8')) | {'trusted_source': True})
    if spec.format != 'pt2' and (args.calibration or args.format != spec.format):
        parser.error('Imported ONNX/TFLite models must use their existing format without calibration search.')
    if args.calibration and args.format != 'onnx':
        parser.error('Calibration search currently supports PT2 → ONNX only.')
    for path, limit in ((args.model, 100 * 1024 * 1024), (args.dataset, 50 * 1024 * 1024), (args.calibration, 50 * 1024 * 1024)):
        if path and path.stat().st_size > limit:
            parser.error(f'{path.name} exceeds its upload size limit')
    data = args.output.resolve(); repo = Repository(data)
    model = ingest_model(args.model.read_bytes(), spec, data); repo.save_model(model)
    dataset = ingest_dataset(args.dataset.read_bytes(), args.dataset.stem, data); repo.save_dataset(dataset)
    calibration = None
    if args.calibration:
        calibration = ingest_dataset(args.calibration.read_bytes(), args.calibration.stem, data); repo.save_dataset(calibration)
    request = CreateRunRequest(model_id=model['id'], format=args.format, target=args.target, dataset_id=dataset['id'],
                               strategy='fidelity_search' if calibration else 'fixed_profiles', calibration_dataset_id=calibration['id'] if calibration else None).model_dump()
    run = repo.create_run(request); directory = data/'runs'/run['id']; directory.mkdir(parents=True)
    try:
        repo.update_run(run['id'], 'running')
        report = run_developer_benchmark(request | {'_model': model, '_calibration': calibration}, dataset, directory)
        report.update(run_id=run['id'], created_at=run['created_at'], settings=request['settings'])
        repo.update_run(run['id'], 'completed', report=report)
        exported = public_report(report, run['id'])
        (directory/'report.json').write_text(json.dumps(exported, indent=2), encoding='utf-8')
        (directory/'report.html').write_text(report_html(exported), encoding='utf-8')
        (directory/'report.csv').write_text(report_csv(exported), encoding='utf-8')
        print(report['summary']['conclusion'])
        print(f'Report: {directory / "report.html"}')
    except Exception as exc:
        repo.update_run(run['id'], 'failed', error=f'{type(exc).__name__}: {str(exc)[:1200]}')
        raise


if __name__ == '__main__': main()
