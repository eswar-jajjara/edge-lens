"""Readable reports preserve measurements and reject misleading comparisons."""
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi.testclient import TestClient
from app.main import create_app
from app.core.config import Settings

from app.services.reports import report_html
from app.services.report_comparison import build_comparison, comparison_html, comparison_csv


def fixture():
    return {'run_id':'run_report_test', 'source':'measured', 'model':{'name':'A <classifier>', 'input_shape':[1,3,8,8], 'layout':'NCHW'},
            'dataset':{'sha256':'d'*64,'image_count':300,'name':'held-out synthetic'},
            'environment':{'platform':'Windows','execution_origin':'local_windows'},
            'metrics':[{'label':'Standard FP32','profile':'standard','accuracy_pct':90,'sample_count':300,'correct_count':270,
                        'artifact_sha256':'a'*64,'dataset_sha256':'d'*64,'evaluation_split':'test',
                        'latency_p50_ms':10,'latency_p95_ms':12,'size_bytes':1048576,'latency_samples_ms':[10,12]},
                       {'label':'Configured candidate','profile':'dashboard','accuracy_pct':91,'sample_count':300,'correct_count':273,
                        'artifact_sha256':'b'*64,'dataset_sha256':'d'*64,'evaluation_split':'test',
                        'latency_p50_ms':8,'latency_p95_ms':11,'size_bytes':524288}],
            'artifacts':[{'sha256':'a'*64,'format':'onnx'}, {'sha256':'b'*64,'format':'onnx'}], 'edge_results':[]}


class ReportLayoutTests(unittest.TestCase):
    def test_html_keeps_raw_precision_and_separates_missing_evidence(self):
        report = fixture()
        report['metrics'][1]['output_max_abs'] = 0.00000012345
        original = copy.deepcopy(report)
        html = report_html(report)
        self.assertEqual(report, original)
        self.assertIn('A &lt;classifier&gt;', html)
        self.assertIn('Top-1 accuracy (%)', html)
        self.assertIn('Median (ms)', html)
        self.assertIn('p95 (ms)', html)
        self.assertIn('1048576', html)
        self.assertIn('1.2345e-07', html)
        self.assertIn('role="img"', html)
        self.assertIn('ESP32 — UNAVAILABLE', html)
        self.assertNotIn('<script', html)

    def test_missing_metrics_do_not_get_zero_bars_or_fake_accuracy(self):
        report = fixture()
        report['metrics'] = [{'label':'Missing measurements','profile':'imported'}]
        html = report_html(report)
        self.assertIn('UNAVAILABLE', html)
        self.assertNotIn('fill="#18797c"', html)
        report['source'] = 'illustrative'
        self.assertNotIn('role="img"', report_html(report))

    def test_observed_same_run_changes_use_percentage_points_and_correct_units(self):
        result = build_comparison([fixture()])
        change = result['same_run_changes'][0]
        self.assertEqual(change['accuracy_delta_pp'], 1)
        self.assertEqual(change['latency_reduction_pct'], 20)
        self.assertEqual(change['size_reduction_pct'], 50)
        self.assertEqual(result['measured_rows'][0]['framework'], 'ONNX Runtime')
        precision = fixture()
        precision['metrics'][0]['profile'] = 'fp32'
        self.assertEqual(build_comparison([precision])['same_run_changes'][0]['accuracy_delta_pp'], 1)
        different_test = fixture()
        different_test['metrics'][1]['dataset_sha256'] = 'e'*64
        self.assertIsNone(build_comparison([different_test])['same_run_changes'][0]['accuracy_delta_pp'])

    def test_reference_values_remain_reference_and_unsafe_links_are_rejected(self):
        ref = {'framework':'PyTorch','model':'Different weights','accuracy_pct':99,
               'source_url':'https://example.org/reference','evidence_status':'MEASURED'}
        result = build_comparison([fixture()], [ref])
        self.assertEqual(result['references'][0]['evidence_status'], 'REFERENCE')
        self.assertIn('CONTEXT ONLY', result['references'][0]['comparison_status'])
        self.assertEqual(result['same_run_changes'][0]['accuracy_delta_pp'], 1)
        self.assertIn('No published-reference advantage', comparison_html(result))
        for url in ('javascript:alert(1)','https://user:secret@example.org/report','file:///report'):
            with self.assertRaises(ValueError):
                build_comparison([fixture()], [ref | {'source_url':url}])
        for number in (float('nan'), float('inf'), -1, 101):
            with self.assertRaises(ValueError):
                build_comparison([fixture()], [ref | {'accuracy_pct':number}])

    def test_legacy_profile_metadata_names_runtime_without_inventing_hash_links(self):
        report = fixture()
        for metric, artifact in zip(report['metrics'], report['artifacts']):
            artifact['profile'] = metric['profile']
            metric.pop('artifact_sha256')
            metric.pop('dataset_sha256')
        result = build_comparison([report])
        self.assertEqual(result['measured_rows'][0]['framework'], 'ONNX Runtime')
        self.assertIsNone(result['measured_rows'][0]['artifact_sha256'])
        self.assertIn('UNAVAILABLE', result['measured_rows'][0]['artifact_identity_status'])
        self.assertIsNone(result['same_run_changes'][0]['accuracy_delta_pp'])
        self.assertEqual(result['experiments'][0]['artifacts'], report['artifacts'])
        report['artifacts'].append(report['artifacts'][0] | {'format':'tflite'})
        self.assertEqual(build_comparison([report])['measured_rows'][0]['framework'], 'Unspecified runtime')
        report['metrics'][1]['artifact_sha256'] = 'z'*64
        self.assertEqual(build_comparison([report])['measured_rows'][1]['framework'], 'Unspecified runtime')

    def test_demos_duplicates_and_fault_only_reports_cannot_enter_comparison(self):
        with self.assertRaises(ValueError): build_comparison([fixture() | {'source':'illustrative'}])
        with self.assertRaises(ValueError): build_comparison([fixture(), fixture()])
        with self.assertRaises(ValueError): build_comparison([fixture() | {'metrics':[]}])

    def test_html_escape_and_csv_formula_protection_are_retained(self):
        report = fixture()
        report['metrics'][0]['label'] = '=HYPERLINK("bad")'
        result = build_comparison([report], [{'framework':'TFLite','model':'<img src=x onerror=alert(1)>', 'source_url':'https://example.org/model'}])
        html = comparison_html(result)
        self.assertNotIn('<img src=x', html)
        self.assertIn('&lt;img', html)
        self.assertIn("'=HYPERLINK", comparison_csv(result))

    def test_local_comparison_endpoint_validates_exports_without_creating_history(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'data') as directory:
            with TestClient(create_app(Settings(environment='test', data_dir=Path(directory), allow_custom_models=True))) as client:
                response = client.post('/api/v1/reports/comparison', json={'reports':[fixture()]})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertIn('role="img"', response.json()['html'])
                self.assertEqual(client.get('/api/v1/runs').json(), [])
                self.assertEqual(client.post('/api/v1/reports/comparison', json={'reports':[fixture() | {'source':'demo'}]}).status_code, 422)
                self.assertEqual(client.post('/api/v1/reports/comparison', content='{"reports": NaN}').status_code, 422)
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / 'data') as directory:
            with TestClient(create_app(Settings(environment='test', data_dir=Path(directory)))) as client:
                self.assertEqual(client.post('/api/v1/reports/comparison', json={'reports':[fixture()]}).status_code, 403)


if __name__ == '__main__': unittest.main()
