"""One CPU worker per API process; deployment intentionally uses one process."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock


class QueueFull(ValueError):
    pass


class ValidationService:
    def __init__(self, repository, data_dir: Path, runner):
        self.repository = repository
        self.data_dir = data_dir
        self.runner = runner
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="benchmark")
        self.lock = Lock()
        self.pending = 0

    def submit(self, payload: dict, dataset: dict) -> dict:
        with self.lock:
            if self.pending >= 2:
                raise QueueFull("Two benchmark jobs are already active. Wait for a job to finish.")
            run = self.repository.create_run(payload)
            self.pending += 1
            try:
                self.executor.submit(self._execute, run, dataset)
            except Exception:
                self.pending -= 1
                self.repository.update_run(run["id"], "failed", error="Could not start the worker.")
                raise
            return run

    def _execute(self, run: dict, dataset: dict) -> None:
        try:
            self.repository.update_run(run["id"], "running")
            directory = self.data_dir / "runs" / run["id"]
            directory.mkdir(parents=True, exist_ok=True)
            payload = dict(run["request"])
            if payload["model_id"].startswith("model_"):
                from app.services.developer_benchmark import run_developer_benchmark
                payload["_model"] = self.repository.get_model(payload["model_id"])
                if payload.get("calibration_dataset_id"):
                    payload["_calibration"] = self.repository.get_dataset(payload["calibration_dataset_id"])
                if payload.get("validation_dataset_id"):
                    payload["_validation"] = self.repository.get_dataset(payload["validation_dataset_id"])
                report = run_developer_benchmark(payload, dataset, directory)
            else:
                report = self.runner(payload, dataset, directory)
            report.update(run_id=run["id"], created_at=run["created_at"], settings=run["request"]["settings"])
            report["edge_estimate_request"] = run["request"].get("edge_estimate", {"enabled": False})
            self.repository.update_run(run["id"], "completed", report=report)
        except Exception as error:
            self.repository.update_run(run["id"], "failed", error=f"{type(error).__name__}: {str(error)[:1200]}")
        finally:
            with self.lock:
                self.pending -= 1

    def close(self):
        self.executor.shutdown(wait=True)
