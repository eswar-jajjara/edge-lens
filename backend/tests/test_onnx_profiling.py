"""ONNX transport, resumable workflow and converted-artifact evidence limits."""
import base64
import hashlib
import json
import unittest
from unittest.mock import patch
import httpx
from test_edge_evidence import ProfileAPITests, fixture, workspace_temp
from app.services import edge_impulse
from app.services.evidence import evidence_sections, artifact_evaluation


class TransportTests(unittest.TestCase):
    def test_onnx_multipart_upload_sends_exact_bytes_fixed_shape_and_no_images(self):
        with workspace_temp() as root:
            path = root / 'classifier.onnx'; content = b'ONNX fixture'; path.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest(); calls = []
            def handler(request):
                calls.append(request)
                return httpx.Response(200, json={'success': True, 'id': 17})
            client = httpx.Client(transport=httpx.MockTransport(handler))
            with patch('app.services.edge_impulse.httpx.Client', return_value=client):
                self.assertEqual(edge_impulse.start_profile(path, 7, 'mcu', 'private-key',
                    model_format='onnx', input_shape=[1, 3, 8, 8], expected_sha256=digest), 17)
            self.assertEqual(str(calls[0].url), edge_impulse.BASE + '/7/pretrained-model/upload')
            self.assertEqual(calls[0].headers['x-api-key'], 'private-key')
            body = calls[0].content
            for expected in (content, edge_impulse.onnx_filename(digest).encode(), b'1,3,8,8', b'name="modelFileType"', b'onnx'):
                self.assertIn(expected, body)
            self.assertNotIn(b'representativeFeatures', body)
            self.assertNotIn(b'private-key', body)

    def test_tflite_endpoint_unchanged_and_onnx_shape_hash_fail_before_network(self):
        with workspace_temp() as root, patch('app.services.edge_impulse.call_api', return_value={'success': True, 'id': 3}) as call:
            path = root / 'model.tflite'; path.write_bytes(b'fixture')
            edge_impulse.start_profile(path, 7, 'mcu', 'key')
            call.assert_called_once_with('POST', '/7/jobs/profile-tflite', 'key',
                {'tfliteFileBase64': base64.b64encode(b'fixture').decode(), 'device': 'mcu'})
            call.reset_mock()
            for kwargs in ({'input_shape': [1, 3, -1, 8]}, {'input_shape': [1, 3, 8, 8], 'expected_sha256': '0' * 64}):
                with self.assertRaises(ValueError): edge_impulse.start_profile(path, 7, 'mcu', 'key', model_format='onnx', **kwargs)
            call.assert_not_called()

    def test_redaction_removes_onnx_bytes(self):
        value = edge_impulse.redact_response({'onnxFileBase64': 'bytes', 'modelFile': 'bytes', 'model': {'profileInfo': {}}}, 'key')
        self.assertEqual(value, {'model': {'profileInfo': {}}})


class ONNXAPITests(unittest.TestCase):
    setUp = ProfileAPITests.setUp
    tearDown = ProfileAPITests.tearDown
    def onnx_run(self):
        self.report['artifacts'][0]['format'] = 'onnx'
        self.repo.update_run(self.run['id'], 'completed', report=self.report)
        with patch('app.services.edge_impulse.start_profile', return_value=111) as upload:
            response = self.client.post(self.endpoint, json=self.body)
        self.assertEqual(response.status_code, 201, response.text)
        self.assertEqual(upload.call_args.kwargs['model_format'], 'onnx')
        return response.json()

    def model_response(self):
        return {'success': True, 'model': {'fileName': edge_impulse.onnx_filename(self.report['artifacts'][0]['sha256']),
                    'profileInfo': {'float32': {'timePerInferenceMs': 3, 'memory': {'eon': {'ram': 500}}}}},
                'modelInfo': {'input': {'shape': [1, 32, 32, 3]}}}

    def refresh(self, job):
        return self.client.post('/api/v1/edge/impulse/' + job['id'] + '/refresh', json={'api_key': self.body['api_key']})

    def test_upload_conversion_profile_resume_and_unavailable_converted_accuracy(self):
        job = self.onnx_run()
        with patch('app.services.edge_impulse.job_finished', return_value=False), patch('app.services.edge_impulse.start_pretrained_profile') as start:
            response = self.refresh(job)
            self.assertEqual(response.status_code, 502); start.assert_not_called()
        model = self.model_response()
        with patch('app.services.edge_impulse.job_finished', side_effect=[True, False]), \
                patch('app.services.edge_impulse.pretrained_model', return_value=model), \
                patch('app.services.edge_impulse.start_pretrained_profile', return_value=222) as start:
            self.assertEqual(self.refresh(job).status_code, 502)
            start.assert_called_once()
        persisted = self.repo.get_edge_record(job['id'])
        self.assertEqual((persisted['phase'], persisted['profile_job_id']), ('profiling', 222))
        with patch('app.services.edge_impulse.job_finished', return_value=True), \
                patch('app.services.edge_impulse.pretrained_model', return_value=model), \
                patch('app.services.edge_impulse.start_pretrained_profile') as start:
            response = self.refresh(job)
            self.assertEqual(response.status_code, 200, response.text); start.assert_not_called()
        value = self.client.get(f'/api/v1/runs/{self.run["id"]}/report').json()
        record = value['evidence_sections']['edge_impulse']['records'][0]
        self.assertEqual(record['upload_evaluation_link'], 'VERIFIED')
        self.assertEqual(record['accuracy_link'], 'UNAVAILABLE')
        self.assertEqual(record['evaluation']['status'], 'UNAVAILABLE')
        self.assertEqual(record['uploaded_artifact_evaluation']['accuracy_pct'], 50)
        self.assertEqual(record['profile_job_id'], 222)
        self.assertNotIn(self.body['api_key'], json.dumps(value))
        # Once complete, use the immutable snapshot, even if Studio changes.
        with patch('app.services.edge_impulse.pretrained_model') as provider:
            repeated = self.refresh(job)
            self.assertEqual(repeated.status_code, 200); provider.assert_not_called()
            self.assertEqual(repeated.json()['id'], response.json()['id'])

    def test_same_project_busy_and_changed_provider_model_rejected(self):
        job = self.onnx_run()
        with patch('app.services.edge_impulse.start_profile') as submit:
            self.assertEqual(self.client.post(self.endpoint, json=self.body).status_code, 409)
            submit.assert_not_called()
        with patch('app.services.edge_impulse.job_finished', return_value=True), \
                patch('app.services.edge_impulse.pretrained_model', return_value={'success': True, 'model': {'fileName': 'another.onnx'}}), \
                patch('app.services.edge_impulse.start_pretrained_profile') as start:
            self.assertEqual(self.refresh(job).status_code, 502); start.assert_not_called()
        self.assertEqual(self.repo.get_edge_record(job['id'])['phase'], 'failed')

    def test_ambiguous_profile_timeout_is_not_retried(self):
        job = self.onnx_run()
        with patch('app.services.edge_impulse.job_finished', return_value=True), \
                patch('app.services.edge_impulse.pretrained_model', return_value=self.model_response()), \
                patch('app.services.edge_impulse.start_pretrained_profile', side_effect=ValueError('Could not reach Edge Impulse securely')) as start:
            self.assertEqual(self.refresh(job).status_code, 502)
            self.assertEqual(self.refresh(job).status_code, 502)
            start.assert_called_once()
        self.assertEqual(self.repo.get_edge_record(job['id'])['phase'], 'profile_unknown')

    def test_failed_upload_and_mismatched_profile_job_never_saved_as_result(self):
        job = self.onnx_run()
        with patch('app.services.edge_impulse.job_finished', side_effect=ValueError('Edge Impulse could not process this ONNX model')):
            self.assertEqual(self.refresh(job).status_code, 502)
        self.assertEqual(self.repo.get_edge_record(job['id'])['phase'], 'failed')
        self.repo.update_edge_record(job['id'], {'phase': 'upload_pending'})
        first = self.model_response(); final = self.model_response(); final['model']['profileJobId'] = 999
        with patch('app.services.edge_impulse.job_finished', return_value=True), \
                patch('app.services.edge_impulse.pretrained_model', side_effect=[first, final]), \
                patch('app.services.edge_impulse.start_pretrained_profile', return_value=222):
            self.assertEqual(self.refresh(job).status_code, 502)
            self.assertEqual(self.refresh(job).status_code, 502)
        self.assertFalse(any(r['kind'] == 'edge_impulse_result' for r in self.repo.edge_records(self.run['id'])))

    def test_hardware_package_still_rejects_onnx(self):
        self.onnx_run()
        response = self.client.post(f'/api/v1/runs/{self.run["id"]}/edge/packages', json={'artifact_index': 0})
        self.assertEqual(response.status_code, 422)

    def test_process_restart_recovers_submission_ambiguity_without_resending(self):
        job = self.onnx_run()
        self.repo.update_edge_record(job['id'], {'phase': 'profile_submitting'})
        self.repo.recover_provider_submissions()
        self.assertEqual(self.repo.get_edge_record(job['id'])['phase'], 'profile_unknown')
        with patch('app.services.edge_impulse.start_pretrained_profile') as submit:
            self.assertEqual(self.refresh(job).status_code, 502)
            submit.assert_not_called()


del ProfileAPITests

if __name__ == '__main__': unittest.main()
