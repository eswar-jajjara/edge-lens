"""Real QDQ conversion tests use synthetic fixtures, not project accuracy evidence."""
from contextlib import closing
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time
import unittest
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4
import zipfile

from app.core.config import Settings
from app.main import create_app
from app.schemas.runs import CreateRunRequest
from app.services.developer_benchmark import disjoint_inputs


def fixture_zip(seed, count=12, prefix='image'):
    import numpy as np
    from PIL import Image
    rng, stream = np.random.default_rng(seed), io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('labels.json', json.dumps({'red': 0, 'blue': 1}))
        for index in range(count):
            label = index % 2
            pixels = rng.integers(0, 70, (8, 8, 3), dtype=np.uint8)
            pixels[:, :, 2 if label else 0] += 150
            raw = io.BytesIO(); Image.fromarray(pixels).save(raw, format='PNG')
            archive.writestr(f'{"blue" if label else "red"}/{prefix}-{index}.png', raw.getvalue())
    return stream.getvalue()


class QuantizationContractTests(unittest.TestCase):
    def test_runtime_timing_failure_retains_other_profile_without_fake_samples(self):
        from app.services.benchmark import _time_profiles
        def broken(_):
            raise RuntimeError('synthetic timing failure')
        failures = {}
        timings = _time_profiles([('fp32', lambda _: [1], None), ('int8', broken, None)],
                                 {'warmup_runs': 1, 'measured_runs': 3}, failures)
        self.assertEqual(set(timings), {'fp32'})
        self.assertEqual(len(timings['fp32']['latency_samples_ms']), 3)
        self.assertIn('int8', failures)

    def test_three_separate_roles_and_onnx_are_required(self):
        valid = dict(model_id='model_a', strategy='quantization_compare', format='onnx', target='esp32',
                     calibration_dataset_id='cal', validation_dataset_id='val', dataset_id='test')
        self.assertEqual(CreateRunRequest(**valid).quantization.calibration_method, 'MinMax')
        for changes in ({'validation_dataset_id': None}, {'dataset_id': 'val'}, {'format': 'tflite'},
                        {'quantization': {'calibration_method': 'MadeUp'}}, {'quantization': {'per_channel': 'yes'}},
                        {'quantization': {'nodes_to_exclude': ['unknown']}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                CreateRunRequest(**(valid | changes))

    def test_validation_overlap_is_rejected_even_with_renamed_files(self):
        import numpy as np
        arrays = [np.zeros((1, 3, 8, 8), dtype=np.float32)]
        with self.assertRaisesRegex(ValueError, 'Validation and test'):
            disjoint_inputs(arrays, arrays, 'Validation', 'test')


@unittest.skipUnless(os.environ.get('EDGELENS_TEST_CUSTOM') == '1', 'Set EDGELENS_TEST_CUSTOM=1 for real QDQ exports')
class PrecisionIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.manual_seed(71)
        model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.ReLU(),
                                    torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(4, 2)).eval()
        cls.root = Path(__file__).resolve().parents[1] / 'data' / ('precision_test_' + uuid4().hex)
        cls.root.mkdir(parents=True)
        cls.pt2 = cls.root / 'fixture.pt2'
        torch.export.save(torch.export.export(model, (torch.zeros(1, 3, 8, 8),)), cls.pt2)
        cls.app = create_app(Settings(environment='test', data_dir=cls.root, allow_custom_models=True))
        cls.context = create_context = __import__('fastapi.testclient', fromlist=['TestClient']).TestClient(cls.app)
        cls.client = create_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)
        shutil.rmtree(cls.root)

    def model(self, content=None, format='pt2'):
        spec = {'name': 'Synthetic precision fixture', 'format': format, 'input_shape': [1, 3, 8, 8],
                'class_count': 2, 'trusted_source': True}
        result = self.client.post('/api/v1/models', content=content or self.pt2.read_bytes(), headers={'X-Model-Spec': quote(json.dumps(spec))})
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()['id']

    def dataset(self, seed, prefix='image'):
        result = self.client.post('/api/v1/datasets', content=fixture_zip(seed, prefix=prefix), headers={'Content-Type': 'application/zip'})
        self.assertEqual(result.status_code, 201, result.text)
        return result.json()['id']

    def payload(self):
        return dict(model_id=self.model(), strategy='quantization_compare', format='onnx', target='esp32',
                    calibration_dataset_id=self.dataset(1), validation_dataset_id=self.dataset(2), dataset_id=self.dataset(3),
                    settings={'warmup_runs': 1, 'measured_runs': 3, 'threads': 1})

    def finish(self, payload):
        result = self.client.post('/api/v1/runs', json=payload)
        self.assertEqual(result.status_code, 202, result.text)
        run_id = result.json()['id']
        for _ in range(600):
            run = self.client.get('/api/v1/runs/' + run_id).json()
            if run['status'] in {'completed', 'failed'}:
                return run
            time.sleep(.1)
        self.fail('Precision worker timed out')

    def test_01_real_pt2_qdq_round_trip_and_stored_evidence(self):
        from app.services.quantization import build_static_int8
        payload = self.payload()
        with patch('app.services.quantization.build_static_int8', wraps=build_static_int8) as builder:
            run = self.finish(payload)
            self.assertEqual(run['status'], 'completed', run)
            self.assertEqual(len(builder.call_args.args[2]), 12)  # Only calibration inputs.
        report = run['report']
        self.assertEqual([c['status'] for c in report['candidates']], ['completed', 'completed'], report['candidates'])
        self.assertEqual(len(report['metrics']), 3)
        self.assertIsNone(report['selection'])
        self.assertFalse(report['test_data_used_for_selection'])
        self.assertEqual(set(report['datasets']), {'calibration', 'validation', 'test'})
        self.assertEqual(len({d['preprocessed_sha256'] for d in report['datasets'].values()}), 3)
        candidate = report['candidates'][1]
        self.assertGreater(candidate['inventory']['qdq_operation_count'], 0)
        self.assertIn('INT8', candidate['inventory']['weight_dtypes'])
        self.assertEqual(candidate['validation_metrics']['evaluation_split'], 'validation')
        for metric in report['metrics']:
            self.assertEqual(metric['sample_count'], 12)
            self.assertEqual(len(metric['latency_samples_ms']), 3)
            self.assertEqual(metric['provenance']['accuracy_pct']['status'], 'MEASURED')
            self.assertEqual(metric['provenance']['device_latency_ms']['status'], 'UNAVAILABLE')
            self.assertEqual(metric['provenance']['peak_ram_bytes']['status'], 'UNAVAILABLE')
            self.assertEqual(metric['latency_mean_ms'], sum(metric['latency_samples_ms']) / 3)
        self.assertEqual(len(report['predictions']), 36)
        with closing(sqlite3.connect(self.app.state.repository.db_path)) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM run_candidates WHERE run_id=?', (run['id'],)).fetchone()[0], 2)
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM run_datasets WHERE run_id=?', (run['id'],)).fetchone()[0], 3)
        for extension in ('html', 'csv'):
            response = self.client.get(f'/api/v1/runs/{run["id"]}/report.{extension}')
            self.assertEqual(response.status_code, 200)
            self.assertIn('validation', response.text)
            self.assertIn('UNAVAILABLE', response.text)
            self.assertNotIn(str(self.root), response.text)
        baseline = next(a for a in report['artifacts'] if a['profile'] == 'fp32')
        onnx_bytes = self.client.get(baseline['download_url']).content
        uploaded = self.model(onnx_bytes, 'onnx')
        for method in ('MinMax', 'Entropy', 'Percentile'):
            with self.subTest(method=method):
                imported = self.finish(payload | {'model_id': uploaded, 'quantization': {'calibration_method': method, 'per_channel': False}})
                self.assertEqual(imported['status'], 'completed', imported)
                self.assertEqual([c['status'] for c in imported['report']['candidates']], ['completed', 'completed'])
                self.assertEqual(len(imported['report']['metrics']), 2)
                self.assertTrue(all(m['reference_profile'] == 'fp32' for m in imported['report']['metrics']))
                self.assertEqual(imported['report']['candidates'][1]['configuration']['calibration_method'], method)

    def test_02_quantizer_failure_keeps_baseline_and_reason(self):
        payload = self.payload()
        with patch('app.services.quantization.build_static_int8', side_effect=ValueError('Synthetic unsupported-operator test')):
            run = self.finish(payload)
        self.assertEqual(run['status'], 'completed', run)
        candidates = run['report']['candidates']
        self.assertEqual(candidates[0]['status'], 'completed')
        self.assertEqual(candidates[1]['status'], 'failed')
        self.assertIsNone(candidates[1]['test_metrics'])
        self.assertIn('unsupported-operator', candidates[1]['failure']['message'])
        self.assertEqual(len(run['report']['metrics']), 2)
        self.assertIn('incomplete', run['report']['summary']['conclusion'])
        self.assertIn('unsupported-operator', self.client.get(f'/api/v1/runs/{run["id"]}/report.html').text)

    def test_03_renamed_preprocessed_overlap_fails_before_conversion(self):
        payload = self.payload()
        payload['validation_dataset_id'] = self.dataset(1, prefix='renamed')
        with patch('app.services.quantization.build_static_int8') as builder:
            run = self.finish(payload)
        self.assertEqual(run['status'], 'failed')
        self.assertIn('overlap after preprocessing', run['error'])
        builder.assert_not_called()

    def test_04_same_archive_and_builtin_preset_rejected(self):
        payload = self.payload()
        duplicate = self.dataset(1)
        self.assertEqual(self.client.post('/api/v1/runs', json=payload | {'validation_dataset_id': duplicate}).status_code, 422)
        self.assertEqual(self.client.post('/api/v1/runs', json=payload | {'model_id': 'mobilenet_v2'}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
