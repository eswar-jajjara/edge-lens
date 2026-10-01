from pathlib import Path
import shutil
import unittest
import uuid
from fastapi.testclient import TestClient
from app.core.config import Settings
from desktop_entry import create_desktop_app


class DesktopEngineTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.directory = self.root / 'data' / ('desktop_test_' + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        self.token = uuid.uuid4().hex + uuid.uuid4().hex
        self.context = TestClient(create_desktop_app(Settings(environment='test', data_dir=self.directory, cors_origins=()), self.root.parent / 'frontend', self.token, 'http://127.0.0.1:8181'))
        self.client = self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)
        shutil.rmtree(self.directory)

    def test_loopback_alone_does_not_grant_access(self):
        self.assertEqual(self.client.get('/api/v1/health').status_code, 403)
        self.assertEqual(self.client.get('/', headers={'X-EdgeLens-Session': 'incorrect'}).status_code, 403)

    def test_authenticated_interface_has_desktop_config(self):
        self.client.cookies.set('edgelens_session', self.token)
        self.assertEqual(self.client.get('/api/v1/health').status_code, 200)
        result = self.client.get('/config.js')
        self.assertIn('desktop:true', result.text)
        self.assertIn("frame-ancestors 'none'", result.headers['content-security-policy'])
        self.assertIn('EdgeLens', self.client.get('/').text)

    def test_cross_origin_writes_are_rejected_with_valid_cookie(self):
        self.client.cookies.set('edgelens_session', self.token)
        result = self.client.post('/api/v1/datasets', content=b'invalid', headers={'Origin': 'https://untrusted.example', 'Content-Type': 'application/zip'})
        self.assertEqual(result.status_code, 403)
        result = self.client.post('/api/v1/datasets', content=b'invalid', headers={'Origin': 'http://127.0.0.1:8181', 'Content-Type': 'application/zip'})
        self.assertEqual(result.status_code, 422)
