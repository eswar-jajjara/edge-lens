import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4
import zipfile
from fastapi.testclient import TestClient
from app.core.config import Settings
from app.main import create_app
from app.repositories import Repository
from app.services.worker_bundle import import_bundle


def bundle(tamper=False, extra=False):
    content = b'test evaluated artifact'; digest = hashlib.sha256(content).hexdigest()
    report = {'source': 'measured', 'run_id': 'worker_original', 'model': {'name': 'Test worker'},
              'environment': {'platform': 'Linux-test'},
              'dataset': {'id': 'worker_dataset', 'name': 'Synthetic test', 'sha256': 'd' * 64, 'image_count': 2},
              'metrics': [{'profile': 'dashboard', 'accuracy_pct': 50., 'artifact_sha256': digest,
                           'dataset_sha256': 'd' * 64, 'sample_count': 2, 'evaluation_split': 'test'}],
              'artifacts': [{'profile': 'dashboard', 'format': 'tflite', 'sha256': digest, 'size_bytes': len(content)}]}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('manifest.json', json.dumps({'schema_version': 1, 'artifacts': [{'index': 0, 'file': 'artifacts/0.tflite'}]}))
        archive.writestr('report.json', json.dumps(report))
        archive.writestr('artifacts/0.tflite', b'changed' if tamper else content)
        if extra: archive.writestr('../outside.txt', 'unexpected')
    return stream.getvalue()


class WorkerBundleTests(unittest.TestCase):
    def directory(self):
        return Path(os.environ.get('EDGELENS_TEST_TMP', tempfile.gettempdir())) / ('worker-import-' + uuid4().hex)

    def test_valid_import_preserves_evaluation_link_and_identifies_remote_origin(self):
        repo = Repository(self.directory())
        result = import_bundle(bundle(), repo)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['report']['environment']['execution_origin'], 'imported_linux_worker')
        self.assertEqual(result['report']['metrics'][0]['sample_count'], 2)
        path = Path(result['report']['artifacts'][0]['path'])
        self.assertTrue(path.is_relative_to(repo.data_dir / 'runs' / result['id']))

    def test_tampered_artifact_and_unexpected_paths_are_rejected_before_creating_run(self):
        repo = Repository(self.directory())
        for content in (bundle(tamper=True), bundle(extra=True)):
            with self.assertRaises(ValueError): import_bundle(content, repo)
        self.assertEqual(repo.list_runs(), [])

    def test_api_import_is_local_only_and_rejects_invalid_zip(self):
        for enabled in (False, True):
            app = create_app(Settings(data_dir=self.directory(), environment='test', allow_custom_models=enabled))
            with TestClient(app) as client:
                response = client.post('/api/v1/worker-reports', content=b'not zip')
                self.assertEqual(response.status_code, 422 if enabled else 403)
                if enabled:
                    result = client.post('/api/v1/worker-reports', content=bundle())
                    self.assertEqual(result.status_code, 201, result.text)
                    self.assertEqual(result.json()['report']['evidence_sections']['esp32']['status'], 'UNAVAILABLE')
