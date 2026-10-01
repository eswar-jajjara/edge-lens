# SQLite schema versions

Current schema: 2. Base v1 tables keep datasets, runs, normalized metrics, layer results and artifacts.
The additive v2 migration is `app/repositories/migrations/002_developer_models.sql`; it adds custom_models and edge_records.
Repository startup applies initialization and migration inside BEGIN IMMEDIATE / COMMIT and records PRAGMA user_version.
An unknown newer schema is rejected instead of silently downgraded. Existing rows are preserved.
The SQL is packaged with the app. Back up the complete application data directory while EdgeLens is closed before upgrades.
