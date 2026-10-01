-- Additive: existing reports and device records remain readable.
CREATE TABLE IF NOT EXISTS run_candidates (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    candidate_id TEXT NOT NULL, name TEXT, candidate_type TEXT, precision TEXT,
    status TEXT NOT NULL, artifact_sha256 TEXT, configuration_json TEXT NOT NULL,
    failure_json TEXT, validation_metrics_json TEXT, test_metrics_json TEXT,
    payload_json TEXT NOT NULL, PRIMARY KEY(run_id, candidate_id)
);
CREATE INDEX IF NOT EXISTS candidates_status ON run_candidates(run_id, status);
CREATE TABLE IF NOT EXISTS run_datasets (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('calibration','validation','test')),
    dataset_id TEXT NOT NULL REFERENCES datasets(id), sha256 TEXT NOT NULL,
    preprocessed_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL,
    PRIMARY KEY(run_id, role)
);
PRAGMA user_version=3;
