# SQLite schema versions

Current schema: 4. Base v1 tables keep datasets, runs, normalized metrics, layer results and artifacts.
The additive v2 migration is `app/repositories/migrations/002_developer_models.sql`; it adds custom_models and edge_records.
Version 3 is `app/repositories/migrations/003_precision_experiments.sql`; it adds run_candidates and run_datasets.
Version 4 is `app/repositories/migrations/004_deployment_search.sql`; it adds sensitivity_results and deployment_selections.
Repository startup applies initialization and migration inside BEGIN IMMEDIATE / COMMIT and records PRAGMA user_version.
An unknown newer schema is rejected instead of silently downgraded. Existing rows are preserved.
The SQL is packaged with the app. Back up the complete application data directory while EdgeLens is closed before upgrades.
