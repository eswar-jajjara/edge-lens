"""Artifact identity, provider isolation and safe response retention contracts."""
import copy
import hashlib
import json
from pathlib import Path
from contextlib import contextmanager
import shutil
from uuid import uuid4
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.core.config import Settings
from app.main import create_app
from app.services import edge_impulse
from app.services.evidence import artifact_evaluation, diagnostic_groups, evidence_sections
from app.services.reports import report_html, report_csv


@contextmanager
def workspace_temp():
    parent = Path(__file__).resolve().parents[1] / 'data'
    directory = parent / ('edge_evidence_test_' + uuid4().hex)
    directory.mkdir(parents=True)
    try:
        yield directory
    finally:
        assert directory.resolve().is_relative_to(parent.resolve())
        shutil.rmtree(directory)


def fixture(digest='a' * 64):
    return {'source': 'measured', 'model': {'input_shape': [1, 32, 32, 3], 'layout': 'NHWC'},
            'dataset': {'sha256': 'd' * 64, 'image_count': 300},
            'artifacts': [{'profile': 'reference', 'format': 'tflite', 'sha256': digest}],
            'metrics': [{'profile': 'imported', 'artifact_sha256': digest, 'dataset_sha256': 'd' * 64,
                         'accuracy_pct': 50, 'sample_count': 300, 'evaluation_split': 'test'}], 'edge_results': []}


class EvidenceTests(unittest.TestCase):
    def test_accuracy_link_requires_matching_bytes_dataset_count_and_test_split(self):
        value = fixture()
        self.assertEqual(artifact_evaluation(value, value['artifacts'][0])['status'], 'MEASURED')
        for change in ({'artifact_sha256': 'b' * 64}, {'dataset_sha256': 'e' * 64}, {'sample_count': 299},
                       {'evaluation_split': 'validation'}, {'artifact_sha256': None}, {'accuracy_pct': None}):
            candidate = copy.deepcopy(value); candidate['metrics'][0].update(change)
            self.assertEqual(artifact_evaluation(candidate, candidate['artifacts'][0])['status'], 'UNAVAILABLE')

    def test_other_candidate_and_old_provider_records_never_inherit_accuracy(self):
        value = fixture(); artifact = value['artifacts'][0]
        for digest in (artifact['sha256'], 'b' * 64):
            value['edge_results'] = [{'kind': 'edge_impulse_result', 'model_sha256': digest}]
            self.assertEqual(evidence_sections(value)['edge_impulse']['records'][0]['accuracy_link'], 'UNAVAILABLE')
        value['edge_results'][0].update(model_sha256=artifact['sha256'], evaluated_artifact=artifact_evaluation(value, artifact))
        self.assertEqual(evidence_sections(value)['edge_impulse']['records'][0]['accuracy_link'], 'VERIFIED')
        value['edge_results'][0]['evaluated_artifact']['dataset_sha256'] = 'f' * 64
        self.assertEqual(evidence_sections(value)['edge_impulse']['records'][0]['accuracy_link'], 'UNAVAILABLE')

    def test_reports_separate_estimates_and_unavailable_hardware(self):
        value = fixture()
        sections = evidence_sections(value)
        self.assertEqual(sections['host']['status'], 'MEASURED')
        self.assertEqual(sections['edge_impulse']['status'], 'UNAVAILABLE')
        self.assertEqual(sections['esp32']['status'], 'UNAVAILABLE')
        html = report_html(value)
        self.assertIn('ESP32 — UNAVAILABLE', html)
        self.assertIn('Tested with mocks only', html)
        self.assertIn('artifact_sha256', report_csv(value))

    def test_diagnostic_coverage_excludes_helpers_and_repeated_other_candidates(self):
        value = fixture(); value['layers'] = [
            {'profile': 'int8', 'name': 'conv', 'operation': 'Conv', 'scope': 'calibration_diagnostic', 'sample_count': 3, 'mae': .1, 'max_abs': .2, 'status': 'drift'},
            {'profile': 'int8', 'name': 'gemm', 'operation': 'Gemm', 'scope': 'calibration_diagnostic', 'mae': None, 'max_abs': None, 'status': 'unmapped'},
            {'profile': 'int8', 'name': 'quant', 'operation': 'QuantizeLinear', 'mae': None, 'max_abs': None, 'status': 'unmapped'},
            {'profile': 'alternative', 'name': 'conv', 'operation': 'Conv', 'mae': None, 'max_abs': None, 'status': 'unmapped'}]
        groups = diagnostic_groups(value)
        self.assertEqual((groups[0]['diagnostic_compared'], groups[0]['diagnostic_eligible']), (1, 2))
        self.assertEqual(groups[0]['quantization_helper_count'], 1)
        self.assertEqual(groups[1]['diagnostic_eligible'], 0)
        self.assertIn('Full inventory entries', report_html(value))
        self.assertIn('1 / 2 eligible operations', report_html(value))

    def test_provider_error_redaction_preserves_structure_and_rejects_nonfinite(self):
        secret = 'unit_test_credential'
        value = {'success': True, 'extra': {'api_key': secret, 'note': 'echo ' + secret}, 'tfliteFileBase64': 'bytes', 'memory': {'ram': 12}}
        result = edge_impulse.redact_response(value, secret)
        self.assertNotIn(secret, json.dumps(result)); self.assertNotIn('tfliteFileBase64', result)
        self.assertEqual(result['memory']['ram'], 12)
        with self.assertRaises(ValueError): edge_impulse.redact_response({'latency': float('nan')}, secret)

    def test_upload_checks_the_bytes_actually_sent(self):
        with workspace_temp() as directory, patch('app.services.edge_impulse.call_api') as call:
            path = Path(directory) / 'model.tflite'; path.write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'evaluated artifact hash'):
                edge_impulse.start_profile(path, 1, 'mcu', 'secret', expected_sha256='0' * 64)
            call.assert_not_called()

    def test_target_discovery_uses_project_endpoint_and_exposes_no_other_project_data(self):
        with patch('app.services.edge_impulse.call_api', return_value={'success': True, 'developmentKeys': {'secret': 'never_return'},
                    'latencyDevices': [{'mcu': 'target-mcu', 'name': 'Test target', 'int8Latency': 4}, {'mcu': '../invalid', 'name': 'invalid'}]}) as call:
            self.assertEqual(edge_impulse.list_targets(7, 'secret'), [{'mcu': 'target-mcu', 'name': 'Test target'}])
            call.assert_called_once_with('GET', '/7', 'secret')


class ProfileAPITests(unittest.TestCase):
    def setUp(self):
        self.temp = workspace_temp(); self.root = self.temp.__enter__()
        self.context = TestClient(create_app(Settings(environment='test', data_dir=self.root, allow_custom_models=True)))
        self.client = self.context.__enter__(); self.repo = self.client.app.state.repository
        self.run = self.repo.create_run({'fixture': True}); directory = self.root / 'runs' / self.run['id']; directory.mkdir(parents=True)
        self.path = directory / 'reference.tflite'; self.path.write_bytes(b'0000TFL3fixture')
        self.report = fixture(hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.report['artifacts'][0]['path'] = str(self.path)
        self.repo.update_run(self.run['id'], 'completed', report=self.report)
        self.body = {'artifact_index': 0, 'project_id': 7, 'device': 'test-mcu', 'api_key': 'unit_test_secret', 'consent_upload': True}
        self.endpoint = f'/api/v1/runs/{self.run["id"]}/edge/impulse'

    def tearDown(self):
        self.context.__exit__(None, None, None); self.temp.__exit__(None, None, None)

    def test_incomplete_evaluation_and_mutated_files_block_upload_before_network(self):
        with patch('app.services.edge_impulse.start_profile') as submit:
            self.report['metrics'][0]['artifact_sha256'] = None
            self.repo.update_run(self.run['id'], 'completed', report=self.report)
            self.assertEqual(self.client.post(self.endpoint, json=self.body).status_code, 409)
            submit.assert_not_called()
            self.path.write_bytes(b'changed')
            self.assertEqual(self.client.post(self.endpoint, json=self.body).status_code, 409)
            submit.assert_not_called()

    def test_consent_required_and_receipt_is_saved_without_credentials(self):
        self.assertEqual(self.client.post(self.endpoint, json=self.body | {'consent_upload': False}).status_code, 422)
        with patch('app.services.edge_impulse.start_profile', return_value=123):
            response = self.client.post(self.endpoint, json=self.body)
        self.assertEqual(response.status_code, 201, response.text); job = response.json()
        self.assertEqual(job['evaluated_artifact']['artifact_sha256'], self.report['artifacts'][0]['sha256'])
        with patch('app.services.edge_impulse.profile_result', return_value={'success': True, 'timePerInferenceMs': 3, 'api_key': self.body['api_key'], 'note': self.body['api_key'], 'unknown_field': {'future': 1}}):
            response = self.client.post('/api/v1/edge/impulse/' + job['id'] + '/refresh', json={'api_key': self.body['api_key']})
        self.assertEqual(response.status_code, 200, response.text)
        receipt = response.json(); self.assertEqual(receipt['raw_response']['unknown_field'], {'future': 1})
        self.assertEqual(len(receipt['response_sha256']), 64)
        self.assertNotIn(self.body['api_key'], json.dumps(self.repo.edge_records(self.run['id'])))
        exported = self.client.get(f'/api/v1/runs/{self.run["id"]}/report').json()
        self.assertEqual(exported['evidence_sections']['edge_impulse']['records'][0]['accuracy_link'], 'VERIFIED')
        self.assertEqual(exported['evidence_sections']['esp32']['status'], 'UNAVAILABLE')
        self.assertNotIn(self.body['api_key'], self.client.get(f'/api/v1/runs/{self.run["id"]}/report.html').text)


if __name__ == '__main__': unittest.main()
