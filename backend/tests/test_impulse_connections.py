"""Project-scoped auth, credential retention and estimate preference contracts."""
import hashlib
import json
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from pydantic import ValidationError
from app.core.config import Settings
from app.main import create_app
from app.schemas.runs import EdgeEstimateRequest
from app.services import edge_impulse
from app.services.impulse_connections import ImpulseConnections
from test_edge_evidence import fixture, workspace_temp


class ConnectionTests(unittest.TestCase):
    def test_project_list_filters_private_fields_and_invalid_entries(self):
        with patch('app.services.edge_impulse.call_api', return_value={'success': True, 'projects': [
                {'id': 7, 'name': 'Classifier', 'apiKey': 'secret', 'owner': {'email': 'private'}},
                {'id': True, 'name': 'bad'}, {'id': -1, 'name': 'bad'}, {'id': 8}]}):
            self.assertEqual(edge_impulse.list_projects('credential'), [{'id': 7, 'name': 'Classifier'}])

    def test_connections_are_scoped_expire_and_disconnect(self):
        clock = [0]
        manager = ImpulseConnections(clock=lambda: clock[0])
        with patch('app.services.edge_impulse.list_projects', return_value=[{'id': 7, 'name': 'Classifier'}]):
            connection = manager.connect('secret-test-key')
        self.assertNotIn('secret-test-key', json.dumps(manager.list()))
        self.assertEqual(manager.credential(connection['id'], 7), ('secret-test-key', 'Classifier'))
        with self.assertRaisesRegex(ValueError, 'not accessible'): manager.credential(connection['id'], 8)
        clock[0] = 8 * 3600
        self.assertEqual(manager.list(), [])
        with self.assertRaisesRegex(ValueError, 'expired'): manager.credential(connection['id'], 7)
        with patch('app.services.edge_impulse.list_projects', return_value=[{'id': 7, 'name': 'Classifier'}]):
            connection = manager.connect('secret-test-key')
        manager.disconnect(connection['id'])
        self.assertEqual(manager.list(), [])
        self.assertEqual(manager._items, {})

    def test_provider_failure_never_creates_a_connected_state(self):
        manager = ImpulseConnections()
        with patch('app.services.edge_impulse.list_projects', side_effect=ValueError('HTTP 403')):
            with self.assertRaises(ValueError): manager.connect('secret-test-key')
        self.assertEqual(manager.list(), [])

    def test_yes_requires_a_destination_and_no_cannot_carry_one(self):
        self.assertFalse(EdgeEstimateRequest().enabled)
        for value in ({'enabled': True}, {'enabled': True, 'project_id': 7}, {'enabled': False, 'project_id': 7},
                      {'enabled': True, 'project_id': 7, 'device': 'target', 'api_key': 'must_not_persist'}):
            with self.assertRaises(ValidationError): EdgeEstimateRequest(**value)
        self.assertTrue(EdgeEstimateRequest(enabled=True, project_id=7, device='target').enabled)


class ConnectedProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = workspace_temp(); self.root = self.temp.__enter__()
        self.context = TestClient(create_app(Settings(environment='test', data_dir=self.root, allow_custom_models=True)))
        self.client = self.context.__enter__(); self.repo = self.client.app.state.repository
        self.run = self.repo.create_run({'edge_estimate': {'enabled': True, 'project_id': 7, 'project_name': 'Classifier', 'device': 'test-mcu'}})
        directory = self.root / 'runs' / self.run['id']; directory.mkdir(parents=True)
        artifact = directory / 'reference.tflite'; artifact.write_bytes(b'0000TFL3fixture')
        report = fixture(hashlib.sha256(artifact.read_bytes()).hexdigest()); report['artifacts'][0]['path'] = str(artifact)
        report['edge_estimate_request'] = self.run['request']['edge_estimate']
        self.repo.update_run(self.run['id'], 'completed', report=report)
        with patch('app.services.edge_impulse.list_projects', return_value=[{'id': 7, 'name': 'Classifier'}]):
            response = self.client.post('/api/v1/edge/impulse/connections', json={'api_key': 'private-project-key'})
        self.assertEqual(response.status_code, 201)
        self.connection = response.json()
        self.body = {'artifact_index': 0, 'project_id': 7, 'device': 'test-mcu', 'connection_id': self.connection['id'], 'consent_upload': True}
        self.endpoint = f'/api/v1/runs/{self.run["id"]}/edge/impulse'

    def tearDown(self):
        self.context.__exit__(None, None, None); self.temp.__exit__(None, None, None)

    def test_connected_upload_refresh_provenance_and_secrets(self):
        with patch('app.services.edge_impulse.start_profile', return_value=123) as submit:
            response = self.client.post(self.endpoint, json=self.body)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(submit.call_args.args[3], 'private-project-key')
        job = response.json(); self.assertEqual(job['project_name'], 'Classifier')
        with patch('app.services.edge_impulse.profile_result', return_value={'success': True, 'timePerInferenceMs': 3}):
            result = self.client.post(f'/api/v1/edge/impulse/{job["id"]}/refresh', json={'connection_id': self.connection['id']})
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()['project_name'], 'Classifier')
        for path in ('', '/report', '/report.html', '/report.csv', '/edge'):
            response = self.client.get(f'/api/v1/runs/{self.run["id"]}{path}')
            self.assertNotIn('private-project-key', response.text)
            self.assertNotIn(self.connection['id'], response.text)
        html = self.client.get(f'/api/v1/runs/{self.run["id"]}/report.html').text
        self.assertIn('Classifier', html); self.assertIn('Test estimate choice', html)

    def test_wrong_project_and_disconnect_block_upload_before_network(self):
        with patch('app.services.edge_impulse.start_profile') as submit:
            self.assertEqual(self.client.post(self.endpoint, json=self.body | {'project_id': 8}).status_code, 409)
            response = self.client.delete('/api/v1/edge/impulse/connections/' + self.connection['id'])
            self.assertEqual(response.status_code, 204)
            self.assertEqual(self.client.post(self.endpoint, json=self.body).status_code, 409)
            submit.assert_not_called()

    def test_validation_errors_do_not_echo_malformed_credentials(self):
        response = self.client.post('/api/v1/edge/impulse/connections', json={'api_key': 'badKEY'})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('badKEY', response.text)
        response = self.client.post(self.endpoint, json=self.body | {'api_key': 'private-project-key'})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn('private-project-key', response.text)


if __name__ == '__main__': unittest.main()
