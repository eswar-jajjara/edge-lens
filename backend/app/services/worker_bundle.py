"""Portable measured-worker reports. Imports never execute serialized models."""
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

MAX_BYTES = 100 * 1024 * 1024


def export_bundle(report_path, output):
    report_path = Path(report_path)
    report = json.loads(report_path.read_text(encoding='utf-8'))
    manifest = []
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for index, artifact in enumerate(report['artifacts']):
            extension = artifact['format']
            if extension not in {'pt2', 'onnx', 'tflite', 'json'}: raise ValueError('Unsupported worker artifact')
            name = 'artifacts/' + str(index) + '.' + extension
            # CLI exports omit local paths. The reference uses the reference profile.
            source = report_path.parent / (artifact['profile'] + '.' + extension)
            if not source.is_file():
                candidates = [p for p in report_path.parent.glob('*.' + extension) if p.is_file() and p.stat().st_size == artifact['size_bytes']
                              and hashlib.sha256(p.read_bytes()).hexdigest() == artifact['sha256']]
                if not candidates: raise ValueError('Evaluated artifact is missing from the worker directory')
                source = candidates[0]
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != artifact['sha256']: raise ValueError('Artifact changed after evaluation')
            archive.writestr(name, content); manifest.append({'index': index, 'file': name})
        report['edge_results'] = []
        archive.writestr('report.json', json.dumps(report, allow_nan=False))
        archive.writestr('manifest.json', json.dumps({'schema_version': 1, 'artifacts': manifest}))


def import_bundle(content, repository):
    if len(content) > MAX_BYTES: raise ValueError('Worker bundle exceeds 100 MiB')
    archive = zipfile.ZipFile(io.BytesIO(content))
    infos = archive.infolist()
    if len(infos) > 100 or sum(i.file_size for i in infos) > MAX_BYTES: raise ValueError('Expanded worker bundle exceeds limits')
    if len({i.filename for i in infos}) != len(infos): raise ValueError('Duplicate ZIP entries')
    if any(i.file_size > MAX_BYTES or i.flag_bits & 1 for i in infos): raise ValueError('Unsupported worker ZIP entry')
    manifest = json.loads(archive.read('manifest.json')); report = json.loads(archive.read('report.json'))
    # Reject NaN/Infinity and any attempt to smuggle fields into reports.
    json.dumps(report, allow_nan=False)
    if manifest.get('schema_version') != 1 or report.get('source') != 'measured': raise ValueError('Expected a measured schema-1 worker bundle')
    if not isinstance(report.get('metrics'), list) or not report['metrics']: raise ValueError('Worker report has no measurements')
    artifacts = report.get('artifacts', [])
    if len(manifest.get('artifacts', [])) != len(artifacts): raise ValueError('Artifact manifest does not cover this report')
    files = {}
    for item in manifest['artifacts']:
        index, name = item['index'], item['file']
        if type(index) is not int or index in files or not 0 <= index < len(artifacts) or not re.fullmatch(r'artifacts/\d+\.(pt2|onnx|tflite|json)', name):
            raise ValueError('Invalid artifact manifest')
        artifact = artifacts[index]; raw = archive.read(name)
        if hashlib.sha256(raw).hexdigest() != artifact.get('sha256') or len(raw) != artifact.get('size_bytes'): raise ValueError('Worker artifact hash or size mismatch')
        files[index] = (name, raw)
    expected_entries = {'manifest.json', 'report.json'} | {name for name, _ in files.values()}
    if set(archive.namelist()) != expected_entries: raise ValueError('Unexpected files in worker bundle')
    from app.services.evidence import artifact_evaluation
    for artifact in artifacts:
        if artifact['format'] in {'onnx', 'tflite'} and artifact_evaluation(report, artifact)['status'] != 'MEASURED':
            raise ValueError('Converted artifact lacks a matching complete held-out evaluation')
    original_run = report.get('run_id')
    converted_formats = {a['format'] for a in artifacts if a['format'] in {'onnx', 'tflite'}}
    if len(converted_formats) != 1: raise ValueError('Expected one converted model format in the worker report')
    request = {'model_id': 'worker_import', 'format': next(iter(converted_formats)), 'target': report.get('target', {}).get('id', 'esp32'),
               'dataset_id': report.get('dataset', {}).get('id'), 'settings': report.get('settings', {}),
               'strategy': 'fixed_profiles', 'edge_estimate': {'enabled': False}}
    run = repository.create_run(request)
    directory = repository.data_dir / 'runs' / run['id']; directory.mkdir(parents=True)
    try:
        for index, (name, raw) in files.items():
            path = directory / Path(name).name; path.write_bytes(raw)
            artifacts[index]['path'] = str(path); artifacts[index].pop('download_url', None)
        report.update(run_id=run['id'], edge_results=[], edge_estimate_request={'enabled': False})
        origin = 'imported_linux_worker' if report.get('environment', {}).get('platform', '').startswith('Linux') else 'imported_worker'
        report['environment'] = {**report.get('environment', {}), 'execution_origin': origin,
                                 'import_provenance': {'bundle_sha256': hashlib.sha256(content).hexdigest(), 'original_run_id': original_run,
                                    'verification': 'Artifact bytes and held-out metric identities checked; external measurement claims are supplied by the trusted worker, not re-measured on this laptop.'}}
        repository.update_run(run['id'], 'completed', report=report)
        return repository.get_run(run['id'])
    except Exception as exc:
        repository.update_run(run['id'], 'failed', error=f'Worker import failed: {type(exc).__name__}')
        raise
