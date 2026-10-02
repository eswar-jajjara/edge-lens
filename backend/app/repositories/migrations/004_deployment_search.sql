-- Additive Phase 2 records. Existing models, experiments and reports remain.
CREATE TABLE IF NOT EXISTS sensitivity_results (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL, node_name TEXT, candidate_id TEXT, parent_candidate TEXT,
    accuracy_recovery_pp REAL, payload_json TEXT NOT NULL,
    PRIMARY KEY(run_id, ordinal)
);
CREATE TABLE IF NOT EXISTS deployment_selections (
    run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
    selected_candidate TEXT, objective TEXT NOT NULL, decision TEXT NOT NULL,
    constraints_json TEXT NOT NULL, diagnostics_json TEXT NOT NULL,
    selection_json TEXT NOT NULL
);
PRAGMA user_version=4;
