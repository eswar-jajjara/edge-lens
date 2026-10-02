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
    parser.add_argument('--validation', type=Path, help='Separate validation ZIP for precision experiments or deployment search')
    parser.add_argument('--strategy', choices=['fixed_profiles', 'fidelity_search', 'quantization_compare', 'deployment_search'])
    parser.add_argument('--objective', choices=['tradeoffs', 'latency', 'size', 'accuracy'], default='tradeoffs')
    parser.add_argument('--max-accuracy-loss-pp', type=float)
    parser.add_argument('--max-size-mib', type=float)
    parser.add_argument('--max-host-latency-ms', type=float)
    parser.add_argument('--max-candidates', type=int, default=16)
    parser.add_argument('--sensitivity-probes', type=int, default=4)
    parser.add_argument('--diagnostic-samples', type=int, default=3)
    parser.add_argument('--search-seconds', type=int, default=600)
    parser.add_argument('--calibration-method', choices=['MinMax', 'Entropy', 'Percentile'], default='MinMax')
    parser.add_argument('--per-tensor', action='store_true', help='Use per-tensor instead of per-channel INT8 weight ranges')
    parser.add_argument('--bundle-output', type=Path, help='New portable worker ZIP to import in the Windows desktop tool')
    parser.add_argument('--warmup-runs', type=int, default=10)
    parser.add_argument('--measured-runs', type=int, default=100)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--format', choices=['onnx', 'tflite'], required=True)
    parser.add_argument('--target', choices=['esp32', 'raspberry_pi'], default='esp32')
    parser.add_argument('--output', type=Path, required=True, help='Local experiment storage directory')
    parser.add_argument('--trust-own-model', action='store_true', help='Acknowledge this is your own trusted model; PT2 loading can execute serialized code')
    args = parser.parse_args()
    if not args.trust_own_model:
        parser.error('Use --trust-own-model only for an export you trust.')
    spec = ModelSpec.model_validate(json.loads(args.spec.read_text(encoding='utf-8')) | {'trusted_source': True})
    strategy = args.strategy or ('quantization_compare' if args.validation else 'fidelity_search' if args.calibration and args.format == 'onnx' else 'fixed_profiles')
    if strategy in {'quantization_compare', 'deployment_search'} and (spec.format not in {'pt2', 'onnx'} or args.format != 'onnx' or not args.calibration or not args.validation):
        parser.error('FP32/INT8 requires PT2 or FP32 ONNX, --format onnx, --calibration and --validation; --dataset is held-out test.')
    if spec.format != 'pt2' and (args.format != spec.format or (args.calibration and strategy not in {'quantization_compare', 'deployment_search'})):
        parser.error('Imported ONNX/TFLite models must use their existing format without calibration search.')
    if args.format == 'tflite' and spec.format == 'pt2' and not args.calibration:
        parser.error('PT2 → INT8 TFLite requires separate --calibration images in the Linux worker.')
    for path, limit in ((args.model, 100 * 1024 * 1024), (args.dataset, 50 * 1024 * 1024), (args.calibration, 50 * 1024 * 1024), (args.validation, 50 * 1024 * 1024)):
        if path and path.stat().st_size > limit:
            parser.error(f'{path.name} exceeds its upload size limit')
    data = args.output.resolve(); repo = Repository(data)
    model = ingest_model(args.model.read_bytes(), spec, data); repo.save_model(model)
    dataset = ingest_dataset(args.dataset.read_bytes(), args.dataset.stem, data); repo.save_dataset(dataset)
    calibration = None
    if args.calibration:
        calibration = ingest_dataset(args.calibration.read_bytes(), args.calibration.stem, data); repo.save_dataset(calibration)
    validation = None
    if args.validation:
        validation = ingest_dataset(args.validation.read_bytes(), args.validation.stem, data); repo.save_dataset(validation)
    request = CreateRunRequest(model_id=model['id'], format=args.format, target=args.target, dataset_id=dataset['id'],
                               settings={'warmup_runs': args.warmup_runs, 'measured_runs': args.measured_runs, 'threads': args.threads},
                               strategy=strategy, calibration_dataset_id=calibration['id'] if calibration else None,
                               validation_dataset_id=validation['id'] if validation else None,
                               quantization={'calibration_method': args.calibration_method, 'per_channel': not args.per_tensor},
                               search={'max_candidates': args.max_candidates, 'sensitivity_probes': args.sensitivity_probes,
                                       'diagnostic_samples': args.diagnostic_samples, 'max_seconds': args.search_seconds},
                               constraints={'objective': args.objective, 'max_accuracy_loss_pp': args.max_accuracy_loss_pp,
                                            'max_size_mib': args.max_size_mib, 'max_host_latency_ms': args.max_host_latency_ms}).model_dump()
    run = repo.create_run(request); directory = data/'runs'/run['id']; directory.mkdir(parents=True)
    try:
        repo.update_run(run['id'], 'running')
        report = run_developer_benchmark(request | {'_model': model, '_calibration': calibration, '_validation': validation}, dataset, directory)
        report.update(run_id=run['id'], created_at=run['created_at'], settings=request['settings'])
        repo.update_run(run['id'], 'completed', report=report)
        exported = public_report(report, run['id'])
        (directory/'report.json').write_text(json.dumps(exported, indent=2), encoding='utf-8')
        (directory/'report.html').write_text(report_html(exported), encoding='utf-8')
        (directory/'report.csv').write_text(report_csv(exported), encoding='utf-8')
        if args.bundle_output:
            from app.services.worker_bundle import export_bundle
            export_bundle(directory/'report.json', args.bundle_output)
        print(report['summary']['conclusion'])
        print(f'Report: {directory / "report.html"}')
    except Exception as exc:
        repo.update_run(run['id'], 'failed', error=f'{type(exc).__name__}: {str(exc)[:1200]}')
        raise


if __name__ == '__main__': main()
