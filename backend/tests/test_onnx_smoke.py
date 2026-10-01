"""Opt-in real conversion smoke test; synthetic inputs are NOT accuracy evidence."""
import io
import json
import os
from pathlib import Path
import shutil
import unittest
import uuid
import zipfile


@unittest.skipUnless(os.getenv("EDGELENS_ML_SMOKE") == "1", "Set EDGELENS_ML_SMOKE=1 for real ONNX conversion (downloads weights)")
class OnnxSmokeTests(unittest.TestCase):
    def test_pretrained_mobilenet_conversion_and_layer_report(self):
        from PIL import Image
        from app.services.datasets import ingest_dataset
        from app.services.benchmark import run_benchmark
        directory = Path(__file__).resolve().parents[1] / "data" / ("smoke_" + uuid.uuid4().hex)
        directory.mkdir(parents=True)
        try:
            contents = io.BytesIO()
            with zipfile.ZipFile(contents, "w") as archive:
                archive.writestr("labels.json", json.dumps({"synthetic": 0}))
                for index, color in enumerate(((100, 70, 40), (40, 90, 140))):
                    encoded = io.BytesIO()
                    Image.new("RGB", (256, 256), color).save(encoded, format="PNG")
                    archive.writestr(f"synthetic/{index}.png", encoded.getvalue())
            dataset = ingest_dataset(contents.getvalue(), "SYNTHETIC SMOKE TEST - arbitrary labels, not accuracy evidence", directory)
            report = run_benchmark({"model_id": "mobilenet_v2", "format": "onnx", "target": "raspberry_pi", "settings": {"warmup_runs": 1, "measured_runs": 3, "threads": 1}}, dataset, directory / "artifacts")
            self.assertEqual(report["source"], "measured")
            self.assertEqual([row["profile"] for row in report["metrics"]], ["pytorch", "standard", "dashboard"])
            self.assertIsNone(report["target"]["latency_ms"])
            self.assertGreater(len(report["layers"]), 20)
            self.assertEqual(len(report["artifacts"]), 3)
            for metric in report["metrics"]:
                self.assertGreater(metric["latency_p50_ms"], 0)
                self.assertEqual(metric["sample_count"], 2)
            print("REAL ONNX SMOKE PASSED:", json.dumps({"layers": len(report["layers"]), "mapped": sum(row["mae"] is not None for row in report["layers"]), "profiles": [row["label"] for row in report["metrics"]]}))
        finally:
            shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
