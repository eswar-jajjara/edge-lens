"""Verify a real Linux conversion report; file size is not MCU compatibility."""
import argparse
import hashlib
import json
from pathlib import Path
parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--results', type=Path, required=True)
args = parser.parse_args()
reports = list((args.results / 'runs').glob('*/report.json'))
assert len(reports) == 1, 'Expected one completed report'
path = reports[0]; report = json.loads(path.read_text())
assert report['source'] == 'measured' and report['environment']['platform'].startswith('Linux')
assert report['dataset']['image_count'] == 300
assert len(report['metrics']) == 3
for artifact in report['artifacts']:
    file = path.parent / (artifact['profile'] + ('.pt2' if artifact['format'] == 'pt2' else '.tflite'))
    assert hashlib.sha256(file.read_bytes()).hexdigest() == artifact['sha256']
    if artifact['format'] == 'tflite': assert artifact['size_bytes'] < 2 * 1024 * 1024, 'Exceeds configured 2 MiB model-file budget'
for metric in report['metrics']:
    assert metric['evaluation_split'] == 'test' and metric['sample_count'] == 300
    assert len(metric['latency_samples_ms']) == 100
    assert metric['artifact_sha256'] in {a['sha256'] for a in report['artifacts']}
print('Verified actual Linux PyTorch / TFLite FP32 / static INT8 evaluation and artifact hashes. Physical ESP32 remains UNAVAILABLE.')
