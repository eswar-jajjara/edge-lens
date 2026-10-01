"""Real SQLite and adversarial archive tests; no ML or web dependencies."""
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import sqlite3
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile
from uuid import uuid4

from app.repositories import Repository
from app.services.datasets import DatasetValidationError, ingest_dataset


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aA1cAAAAASUVORK5CYII=")


def archive_bytes(entries=None, labels=None):
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        if labels is not False:
            archive.writestr("labels.json", json.dumps({"cat": 281, "dog": 207}) if labels is None else labels)
        for name, content in (entries if entries is not None else [("cat/a.png", PNG), ("dog/b.png", PNG)]):
            if isinstance(name, str):
                member = zipfile.ZipInfo(name)
                # Write adversarial paths verbatim even on Windows, where ZipInfo normalizes slashes.
                member.filename = name
                member.orig_filename = name
                archive.writestr(member, content)
            else:
                archive.writestr(name, content)
    return stream.getvalue()


class TemporaryDataTest(unittest.TestCase):
    def setUp(self):
        # Ordinary mkdir inherits the selected test directory's ACL on Windows.
        # Some sandboxed Windows runners cannot access tempfile's private ACL.
        base = Path(os.environ.get("EDGELENS_TEST_TMP", tempfile.gettempdir())).resolve()
        base.mkdir(parents=True, exist_ok=True)
        self.data_dir = base / ("edgelens-storage-" + uuid4().hex)
        self.data_dir.mkdir()

        def cleanup():
            if self.data_dir.resolve().parent != base or self.data_dir.is_symlink():
                raise RuntimeError("Refusing to clean an unexpected test directory")
            shutil.rmtree(self.data_dir)

        self.addCleanup(cleanup)


class RepositoryTests(TemporaryDataTest):
    def setUp(self):
        super().setUp()
        self.repository = Repository(self.data_dir)
        self.report = {
            "metadata": {"benchmark_location": "host", "dataset_hash": "abc"},
            "metrics": [{"profile": "onnx_standard", "label": "Standard ONNX", "accuracy_pct": 80.0,
                         "accuracy_delta_pp": -1.0, "agreement_pct": 99.0, "latency_p50_ms": 5.0,
                         "latency_p95_ms": 6.2, "conversion_seconds": 2.4, "size_bytes": 2048,
                         "output_mae": 0.001, "output_max_abs": 0.005}],
            "layers": [{"profile": "onnx_standard", "name": "features.0", "operation": "Conv2d",
                        "status": "warning", "mae": 0.003, "max_abs": 0.02, "detail": "Exceeds tolerance"},
                       {"profile": "onnx_standard", "name": "classifier", "operation": "Linear",
                        "status": "unavailable", "mae": None, "max_abs": None, "detail": "Graph fusion"}],
            "artifacts": [{"profile": "onnx_standard", "format": "onnx", "path": "runs/a/model.onnx",
                           "sha256": "1234", "size_bytes": 2048}],
        }

    def test_dataset_and_report_survive_new_repository_instance(self):
        metadata = ingest_dataset(archive_bytes(), "Images", self.data_dir)
        saved = self.repository.save_dataset(metadata)
        run = self.repository.create_run({"dataset_id": metadata["id"], "model": "mobilenet_v2"})
        self.assertEqual(run["status"], "queued")
        self.assertIsNone(run["report"])
        self.repository.update_run(run["id"], "completed", report=self.report)
        reopened = Repository(self.data_dir)
        self.assertEqual(reopened.get_dataset(metadata["id"]), saved)
        self.assertEqual(reopened.list_datasets(), [saved])
        self.assertEqual(reopened.get_run(run["id"])["report"], self.report)
        self.assertEqual(reopened.list_runs()[0]["request"]["model"], "mobilenet_v2")

    def test_report_columns_are_queryable_with_null_for_unavailable_layers(self):
        run = self.repository.create_run({})
        self.repository.update_run(run["id"], "completed", report=self.report)
        with closing(sqlite3.connect(self.repository.db_path)) as connection:
            metric = connection.execute("SELECT profile,accuracy_delta_pp,latency_p95_ms FROM run_metrics").fetchone()
            layer = connection.execute("SELECT layer_name,status,mae FROM layer_results WHERE ordinal=1").fetchone()
            artifact = connection.execute("SELECT format,sha256,size_bytes FROM model_artifacts").fetchone()
            mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(metric, ("onnx_standard", -1.0, 6.2))
        self.assertEqual(layer, ("classifier", "unavailable", None))
        self.assertEqual(artifact, ("onnx", "1234", 2048))
        self.assertEqual(mode, "wal")

    def test_report_replacement_removes_old_normalized_rows(self):
        run = self.repository.create_run({})
        self.repository.update_run(run["id"], "completed", report=self.report)
        self.repository.update_run(run["id"], "completed", report={"metrics": [], "layers": [], "artifacts": []})
        with closing(sqlite3.connect(self.repository.db_path)) as connection:
            for table in ("run_metrics", "layer_results", "model_artifacts"):
                self.assertEqual(connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0], 0)

    def test_failed_report_write_rolls_back_run_and_all_normalized_rows(self):
        run = self.repository.create_run({})
        self.repository.update_run(run["id"], "completed", report=self.report)
        invalid = {"metrics": [{"profile": {"invalid": "SQL type"}}]}
        with self.assertRaises(sqlite3.ProgrammingError):
            self.repository.update_run(run["id"], "failed", report=invalid)
        self.assertEqual(self.repository.get_run(run["id"])["report"], self.report)
        self.assertEqual(self.repository.get_run(run["id"])["status"], "completed")
        with closing(sqlite3.connect(self.repository.db_path)) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM layer_results").fetchone()[0], 2)

    def test_recovery_only_fails_queued_and_running_jobs(self):
        queued = self.repository.create_run({})
        running = self.repository.create_run({})
        finished = self.repository.create_run({})
        self.repository.update_run(running["id"], "running")
        self.repository.update_run(finished["id"], "completed", report=self.report)
        self.assertEqual(Repository(self.data_dir).recover_interrupted(), 2)
        for run in (queued, running):
            saved = self.repository.get_run(run["id"])
            self.assertEqual(saved["status"], "failed")
            self.assertEqual(saved["error"]["code"], "INTERRUPTED")
        self.assertEqual(self.repository.get_run(finished["id"])["status"], "completed")
        self.assertEqual(self.repository.recover_interrupted(), 0)

    def test_parameterized_ids_and_input_limits(self):
        self.assertIsNone(self.repository.get_run("' OR 1=1 --"))
        self.assertIsNone(self.repository.get_dataset("missing"))
        with self.assertRaises(KeyError):
            self.repository.update_run("missing", "running")
        with self.assertRaises(ValueError):
            self.repository.create_run({"value": float("nan")})
        with self.assertRaises(ValueError):
            self.repository.list_runs(0)
        with self.assertRaises(ValueError):
            self.repository.update_run("missing", "fictional")

    def test_independent_connections_support_concurrent_writers(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            runs = list(executor.map(lambda index: self.repository.create_run({"index": index}), range(12)))
        self.assertEqual(len({run["id"] for run in runs}), 12)
        self.assertEqual(len(self.repository.list_runs()), 12)


class DatasetTests(TemporaryDataTest):
    def assert_rejected(self, content):
        with self.assertRaises(DatasetValidationError):
            ingest_dataset(content, "Invalid", self.data_dir)
        self.assertFalse((self.data_dir / "datasets").exists(), "Invalid archives must not be saved")

    def test_valid_archive_is_saved_without_extracting_images(self):
        content = archive_bytes()
        metadata = ingest_dataset(content, "Two classes", self.data_dir)
        self.assertEqual(metadata["image_count"], 2)
        self.assertEqual(metadata["class_count"], 2)
        self.assertEqual([{k:v for k,v in x.items() if k != "sha256"} for x in metadata["entries"]], [{"path": "cat/a.png", "label": 281}, {"path": "dog/b.png", "label": 207}])
        self.assertEqual(Path(metadata["path"]).read_bytes(), content)
        self.assertEqual(len(list(self.data_dir.rglob("*"))), 2)  # Directory and ZIP only.

    def test_traversal_absolute_backslash_and_drive_paths_rejected(self):
        for name in ("../bad.png", "/bad.png", "cat/../../bad.png", "cat\\bad.png", "C:/bad.png", "cat//bad.png"):
            with self.subTest(name=name):
                self.assert_rejected(archive_bytes([(name, PNG), ("dog/b.png", PNG)]))

    def test_duplicate_paths_rejected_case_insensitively(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.assert_rejected(archive_bytes([("cat/a.png", PNG), ("cat/a.png", PNG)]))
        self.assert_rejected(archive_bytes([("cat/a.png", PNG), ("CAT/A.PNG", PNG)]))

    def test_symlink_and_encrypted_entries_rejected(self):
        member = zipfile.ZipInfo("cat/a.png")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        self.assert_rejected(archive_bytes([(member, b"../../secret"), ("dog/b.png", PNG)]))
        payload = bytearray(archive_bytes())
        offset = 0
        while (offset := payload.find(b"PK\x01\x02", offset)) != -1:
            payload[offset + 8] |= 1
            offset += 4
        self.assert_rejected(bytes(payload))

    def test_invalid_and_missing_label_mappings_rejected(self):
        for labels in (False, "not json", "[]", "{}", '{"cat":true,"dog":207}', '{"cat":10000,"dog":207}',
                       '{"cat":281,"dog":281}', '{"cat":281,"cat":282,"dog":207}', '{"unknown":281,"dog":207}'):
            with self.subTest(labels=labels):
                self.assert_rejected(archive_bytes(labels=labels))

    def test_too_few_too_many_images_and_non_images_rejected(self):
        for entries in ([('cat/a.png', PNG)], [(f"cat/{index}.png", PNG) for index in range(1001)],
                        [('cat/a.png', PNG), ('cat/code.py', b'print(1)')],
                        [('cat/a.png', b'not an image'), ('dog/b.png', PNG)]):
            with self.subTest(count=len(entries)):
                self.assert_rejected(archive_bytes(entries))

    def test_archive_entry_expansion_and_total_size_limits(self):
        content = archive_bytes()
        for constant, limit in (("MAX_ARCHIVE_BYTES", len(content) - 1), ("MAX_ENTRY_BYTES", 10), ("MAX_EXPANDED_BYTES", 20)):
            with self.subTest(constant=constant), patch(f"app.services.datasets.{constant}", limit):
                self.assert_rejected(content)

    def test_invalid_zip_and_deep_json_are_validation_errors(self):
        self.assert_rejected(b"not zip")
        self.assert_rejected(archive_bytes()[:40])
        self.assert_rejected(archive_bytes(labels="[" * 2000 + "]" * 2000))


if __name__ == "__main__":
    unittest.main()
