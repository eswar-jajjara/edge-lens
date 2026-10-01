import io
import json
from pathlib import Path
import shutil
import time
import unittest
import uuid
import zipfile
from fastapi.testclient import TestClient
from app.core.config import Settings
from app.main import create_app


def archive_bytes():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("labels.json", json.dumps({"class": 0}))
        for name in ("a.png", "b.png"):
            archive.writestr(f"class/{name}", b"\x89PNG\r\n\x1a\n" + b"test fixture")
    return stream.getvalue()


def fixture_runner(request, dataset, output_dir):
    artifact = output_dir / "fixture.onnx"
    artifact.write_bytes(b"test artifact, not a model")
    return {"source": "test_fixture", "summary": {"conclusion": "Fixture only"}, "model": {"name": "Test"},
            "dataset": dataset, "target": {"name": "ESP32", "notes": ["Not measured"]},
            "environment": {"benchmark_scope": "host_cpu"},
            "metrics": [{"profile": "standard", "accuracy_pct": 50}],
            "layers": [{"profile": "standard", "name": "<script>layer</script>", "status": "unmapped"}],
            "artifacts": [{"path": str(artifact), "format": "onnx"}], "methodology": [], "limitations": []}


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(__file__).resolve().parents[1] / "data" / ("test_" + uuid.uuid4().hex)
        self.directory.mkdir(parents=True)
        config = Settings(environment="test", data_dir=self.directory)
        self.caps = {"onnx": {"available": True}, "tflite": {"available": False, "reason": "Linux extra required"}}
        self.app = create_app(config, runner=fixture_runner, capability_provider=lambda: self.caps)
        self.context = TestClient(self.app)
        self.client = self.context.__enter__()
        self.payload = {"model_id": "mobilenet_v2", "format": "onnx", "target": "esp32", "dataset_id": "missing"}

    def tearDown(self):
        self.context.__exit__(None, None, None)
        shutil.rmtree(self.directory)

    def upload(self):
        result = self.client.post("/api/v1/datasets", content=archive_bytes(), headers={"Content-Type": "application/zip", "X-Dataset-Name": "Test fixture"})
        self.assertEqual(result.status_code, 201, result.text)
        self.assertNotIn("path", result.json())
        self.payload["dataset_id"] = result.json()["id"]

    def test_upload_job_persist_report_and_download(self):
        self.upload()
        result = self.client.post("/api/v1/runs", json=self.payload)
        self.assertEqual(result.status_code, 202, result.text)
        run_id = result.json()["id"]
        for _ in range(100):
            run = self.client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] in {"failed", "completed"}:
                break
            time.sleep(0.01)
        self.assertEqual(run["status"], "completed", run)
        self.assertNotIn("path", run["report"]["artifacts"][0])
        self.assertNotIn("path", run["report"]["dataset"])
        self.assertEqual(len(self.client.get("/api/v1/runs").json()), 1)
        self.assertEqual(self.client.get(f"/api/v1/runs/{run_id}/artifacts/0").content, b"test artifact, not a model")
        html = self.client.get(f"/api/v1/runs/{run_id}/report.html").text
        self.assertIn("&lt;script&gt;layer&lt;/script&gt;", html)
        self.assertNotIn("<script>layer", html)
        self.assertIn("accuracy_pct", self.client.get(f"/api/v1/runs/{run_id}/report.csv").text)

    def test_missing_dataset_and_missing_runtime(self):
        self.assertEqual(self.client.post("/api/v1/runs", json=self.payload).status_code, 404)
        self.upload()
        self.payload["format"] = "tflite"
        result = self.client.post("/api/v1/runs", json=self.payload)
        self.assertEqual(result.status_code, 503)
        self.assertEqual(result.json()["detail"]["code"], "RUNTIME_UNAVAILABLE")
        self.assertEqual(self.client.get("/api/v1/runs").json(), [])

    def test_bad_zip_and_bounded_requests(self):
        self.assertEqual(self.client.post("/api/v1/datasets", content=b"bad", headers={"Content-Type": "application/zip"}).status_code, 422)
        for patch in [{"dataset_id": "../../secret"}, {"settings": {"measured_runs": 0}}, {"settings": {"atol": -1}}, {"unexpected": True}]:
            self.assertEqual(self.client.post("/api/v1/runs", json=self.payload | patch).status_code, 422)

    def test_failed_job_is_recorded_without_report(self):
        self.upload()
        def fail(*args):
            raise RuntimeError("expected fixture error")
        self.app.state.validation.runner = fail
        run_id = self.client.post("/api/v1/runs", json=self.payload).json()["id"]
        for _ in range(100):
            run = self.client.get(f"/api/v1/runs/{run_id}").json()
            if run["status"] == "failed":
                break
            time.sleep(.01)
        self.assertIn("expected fixture error", run["error"])
        self.assertEqual(self.client.get(f"/api/v1/runs/{run_id}/report").status_code, 409)

    def test_health_capabilities_cors(self):
        self.assertEqual(self.client.get("/api/v1/health").json()["mode"], "benchmark")
        self.assertTrue(self.client.get("/api/v1/capabilities").json()["persistent_runs"])
        allowed = self.client.get("/api/v1/health", headers={"Origin": "http://localhost:5173"})
        denied = self.client.get("/api/v1/health", headers={"Origin": "https://unlisted.example"})
        self.assertIn("access-control-allow-origin", allowed.headers)
        self.assertNotIn("access-control-allow-origin", denied.headers)


if __name__ == "__main__":
    unittest.main()
