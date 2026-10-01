-- Additive schema migration. The caller supplies BEGIN IMMEDIATE / COMMIT.
CREATE TABLE IF NOT EXISTS custom_models (
    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, metadata_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edge_records (
    id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id),
    kind TEXT NOT NULL, created_at TEXT NOT NULL, payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS edge_run ON edge_records(run_id, created_at);
PRAGMA user_version=2;
