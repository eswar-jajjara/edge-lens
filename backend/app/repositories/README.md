# SQLite persistence

`Repository(data_dir)` creates `edgelens.sqlite3` with WAL, foreign keys and a 30-second busy timeout. Each operation owns its connection and transaction. Call `recover_interrupted()` when the single backend worker starts to mark abandoned queued/running jobs as failed.

`datasets` stores archive metadata, labels and the server-created archive path. `runs` retains the entire request and report JSON. `run_metrics`, `layer_results` and `model_artifacts` store individually queryable report records; report replacement updates all three tables atomically. Artifact rows contain filesystem paths, SHA-256 hashes and sizes, rather than model blobs. Preserve the data directory when deploying.

Schema version 1 is created idempotently. Future schema changes need explicit migrations before `PRAGMA user_version` is advanced. SQLite is intended for one backend instance, not a shared network filesystem or multiple job runners.
