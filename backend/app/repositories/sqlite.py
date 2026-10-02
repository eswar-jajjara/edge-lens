"""SQLite persistence: each operation owns its connection and transaction."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


class Repository:
    """A single-instance prototype repository backed by edgelens.sqlite3."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "edgelens.sqlite3"
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > 4:
                raise RuntimeError("This database was created by a newer EdgeLens version; upgrade the application.")
            migrations = Path(__file__).parent / "migrations"
            migration = "".join((migrations / filename).read_text(encoding="utf-8") + "\n"
                                for threshold, filename in ((2, "002_developer_models.sql"), (3, "003_precision_experiments.sql"), (4, "004_deployment_search.sql"))
                                if version < threshold)
            connection.executescript("""
                BEGIN IMMEDIATE;
                CREATE TABLE IF NOT EXISTS datasets (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL,
                    sha256 TEXT, image_count INTEGER NOT NULL, class_count INTEGER NOT NULL,
                    path TEXT NOT NULL, metadata_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('queued','running','completed','failed')),
                    request_json TEXT NOT NULL, error_json TEXT, report_json TEXT
                );
                CREATE INDEX IF NOT EXISTS runs_created ON runs(created_at DESC);
                CREATE TABLE IF NOT EXISTS run_metrics (
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, profile TEXT, label TEXT,
                    accuracy_pct REAL, accuracy_delta_pp REAL, agreement_pct REAL,
                    latency_p50_ms REAL, latency_p95_ms REAL, conversion_seconds REAL,
                    size_bytes INTEGER, output_mae REAL, output_max_abs REAL,
                    payload_json TEXT NOT NULL, PRIMARY KEY(run_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS layer_results (
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, profile TEXT, layer_name TEXT, operation TEXT,
                    status TEXT, mae REAL, max_abs REAL, detail TEXT, payload_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, ordinal)
                );
                CREATE INDEX IF NOT EXISTS layers_status ON layer_results(run_id, status);
                CREATE TABLE IF NOT EXISTS model_artifacts (
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, profile TEXT, format TEXT, path TEXT,
                    sha256 TEXT, size_bytes INTEGER, payload_json TEXT NOT NULL,
                    PRIMARY KEY(run_id, ordinal)
                );

            """ + migration + "\nCOMMIT;")

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save_dataset(self, metadata: dict) -> dict:
        stored = dict(metadata)
        stored.setdefault("created_at", _now())
        stored["path"] = str(stored["path"])
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO datasets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (stored["id"], stored["name"], stored["created_at"], stored.get("sha256"),
                 stored["image_count"], stored["class_count"], stored["path"], _json(stored)),
            )
        return stored

    def get_dataset(self, dataset_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute("SELECT metadata_json FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return json.loads(row["metadata_json"]) if row else None

    def list_datasets(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute("SELECT metadata_json FROM datasets ORDER BY created_at DESC, id DESC").fetchall()
        return [json.loads(row["metadata_json"]) for row in rows]

    def create_run(self, request: dict) -> dict:
        run_id, created_at = "run_" + uuid4().hex, _now()
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO runs VALUES (?, ?, ?, 'queued', ?, NULL, NULL)",
                (run_id, created_at, created_at, _json(request)),
            )
        return self.get_run(run_id)

    @staticmethod
    def _decode_run(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"], "created_at": row["created_at"], "updated_at": row["updated_at"],
            "status": row["status"], "request": json.loads(row["request_json"]),
            "error": json.loads(row["error_json"]) if row["error_json"] is not None else None,
            "report": json.loads(row["report_json"]) if row["report_json"] is not None else None,
        }

    def get_run(self, run_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._decode_run(row) if row else None

    def list_runs(self, limit: int = 30) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM runs ORDER BY created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()
        return [self._decode_run(row) for row in rows]

    def update_run(self, run_id: str, status: str, error: Any = None, report: dict | None = None) -> dict:
        if status not in {"queued", "running", "completed", "failed"}:
            raise ValueError("Invalid run status")
        with self._connection() as connection:
            if connection.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone() is None:
                raise KeyError(run_id)
            connection.execute(
                "UPDATE runs SET status=?, updated_at=?, error_json=?, report_json=? WHERE id=?",
                (status, _now(), _json(error) if error is not None else None,
                 _json(report) if report is not None else None, run_id),
            )
            for table in ("run_metrics", "layer_results", "model_artifacts", "run_candidates", "run_datasets", "sensitivity_results", "deployment_selections"):
                connection.execute(f"DELETE FROM {table} WHERE run_id=?", (run_id,))
            if report is not None:
                self._save_report_rows(connection, run_id, report)
        return self.get_run(run_id)

    @staticmethod
    def _save_report_rows(connection: sqlite3.Connection, run_id: str, report: dict):
        for index, value in enumerate(report.get("sensitivity", [])):
            connection.execute("INSERT INTO sensitivity_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (run_id, index, value.get("node"), value.get("candidate_id"), value.get("parent_candidate"), value.get("accuracy_recovery_pp"), _json(value)))
        if report.get("experiment_type") == "deployment_optimization":
            selection = report["selection"]
            connection.execute("INSERT INTO deployment_selections VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (run_id, selection.get("selected"), selection["objective"], selection["decision"],
                                _json(selection["constraints"]), _json(report.get("diagnostics", {})), _json(selection)))
        for candidate in report.get("candidates", []):
            connection.execute("INSERT INTO run_candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (run_id, candidate["id"], candidate.get("name"), candidate.get("candidate_type"), candidate.get("precision"),
                                candidate["status"], candidate.get("artifact_sha256"), _json(candidate.get("configuration", {})),
                                _json(candidate["failure"]) if candidate.get("failure") else None,
                                _json(candidate["validation_metrics"]) if candidate.get("validation_metrics") else None,
                                _json(candidate["test_metrics"]) if candidate.get("test_metrics") else None, _json(candidate)))
        for role, dataset in report.get("datasets", {}).items():
            connection.execute("INSERT INTO run_datasets VALUES (?, ?, ?, ?, ?, ?)",
                               (run_id, role, dataset["id"], dataset["sha256"], dataset["preprocessed_sha256"], _json(dataset)))
        for index, metric in enumerate(report.get("metrics", [])):
            connection.execute(
                "INSERT INTO run_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, index, metric.get("profile"), metric.get("label"),
                 metric.get("accuracy_pct"), metric.get("accuracy_delta_pp"), metric.get("agreement_pct"),
                 metric.get("latency_p50_ms"), metric.get("latency_p95_ms"), metric.get("conversion_seconds"),
                 metric.get("size_bytes"), metric.get("output_mae"), metric.get("output_max_abs"), _json(metric)),
            )
        for index, layer in enumerate(report.get("layers", [])):
            connection.execute(
                "INSERT INTO layer_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, index, layer.get("profile"), layer.get("name"), layer.get("operation"),
                 layer.get("status"), layer.get("mae"), layer.get("max_abs"), layer.get("detail"), _json(layer)),
            )
        for index, artifact in enumerate(report.get("artifacts", [])):
            connection.execute(
                "INSERT INTO model_artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, index, artifact.get("profile"), artifact.get("format"),
                 artifact.get("path"), artifact.get("sha256"), artifact.get("size_bytes"), _json(artifact)),
            )

    def save_model(self, metadata):
        with self._connection() as connection:
            connection.execute("INSERT INTO custom_models VALUES (?, ?, ?)", (metadata["id"], metadata["created_at"], _json(metadata)))

    def get_model(self, model_id):
        with self._connection() as connection:
            row = connection.execute("SELECT metadata_json FROM custom_models WHERE id=?", (model_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def list_models(self):
        with self._connection() as connection:
            rows = connection.execute("SELECT metadata_json FROM custom_models ORDER BY created_at DESC").fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_edge_record(self, run_id, kind, payload, record_id=None):
        value = dict(payload, id=record_id or "edge_" + uuid4().hex, run_id=run_id, kind=kind, created_at=_now())
        with self._connection() as connection:
            connection.execute("INSERT INTO edge_records VALUES (?, ?, ?, ?, ?)", (value["id"], run_id, kind, value["created_at"], _json(value)))
        return value

    def get_edge_record(self, record_id):
        with self._connection() as connection:
            row = connection.execute("SELECT payload_json FROM edge_records WHERE id=?", (record_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def edge_records(self, run_id):
        with self._connection() as connection:
            rows = connection.execute("SELECT payload_json FROM edge_records WHERE run_id=? ORDER BY created_at", (run_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def recover_interrupted(self) -> int:
        """Fail interrupted work on startup; a previous process cannot finish it."""
        error = {"code": "INTERRUPTED", "message": "The server restarted before this run finished. Start a new run."}
        with self._connection() as connection:
            cursor = connection.execute(
                "UPDATE runs SET status='failed', updated_at=?, error_json=? WHERE status IN ('queued', 'running')",
                (_now(), _json(error)),
            )
            return cursor.rowcount
