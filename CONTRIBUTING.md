# Contributing to EdgeLens

This guide is for teammates making changes to the public source repository. For installing and using the Windows desktop tool, start with the [README](README.md) and [team setup guide](docs/TEAM-SETUP.md).

## Local development

Use Windows 10/11 x64 with Python 3.12 x64 and Node.js 22.12+ x64. Run the pinned setup once, then start the desktop tool:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-desktop.ps1
npm run desktop
```

## Repository map

| Path | Responsibility |
| --- | --- |
| `backend/app/api` | Local FastAPI endpoints |
| `backend/app/services` | Conversion, validation, measurements, reports, and provider integrations |
| `backend/app/repositories` | SQLite persistence and schema migrations |
| `backend/tests` | Python engine and API tests |
| `frontend/src` | Desktop user interface |
| `desktop` | Electron main process and preload bridge |
| `scripts` | Setup, build, diagnostic, and worker utilities |
| `docs` | Setup, methodology, feature, and verification guides |

The Electron shell starts a private local Python engine. User data lives outside the source checkout in `%APPDATA%\EdgeLens\data`.

## Before opening a pull request

1. Keep a change focused and describe the user-visible behavior.
2. Update the relevant guide when setup, supported inputs, measurement meaning, or limitations change.
3. Run the applicable existing checks and include their result in the pull request. Do not report a check as passing unless it ran.
4. For database changes, add a numbered migration, keep migration backups and newer-schema refusal intact, and cover the change with the existing storage test suite.
5. For benchmark changes, preserve provenance and keep host measurements, Edge Impulse estimates, and physical-device measurements clearly separated.
6. Review `git status` before committing. Do not add `.env` files, API keys, model files, image datasets, SQLite databases, reports, caches, or generated build folders.

## Evidence and security rules

- Keep test data splits and model preprocessing explicit.
- Label host results as host measurements; selecting a deployment goal does not measure that device.
- Mark unsupported mappings or unmeasured values unavailable instead of implying zero drift or perfect accuracy.
- Do not claim one converter is universally better. State the model, test set, runtime, hardware, and measured differences.
- Treat uploaded PT2 files as executable serialized content. EdgeLens is not a sandbox for untrusted models.
- Edge Impulse uploads require explicit user selection and project credentials. Never commit project keys or private model artifacts.