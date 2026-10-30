# Desktop delivery and optional Linux worker

## Windows desktop setup (source, no separate installer)

EdgeLens is shared as source from [the public GitHub repository](https://github.com/eswar-jajjara/edge-lens). There is no separate EdgeLens setup executable or installer release. Teammates need Windows 10/11 x64, 64-bit Python 3.12 and 64-bit Node.js 22.12 or later. Git is optional when downloading the source ZIP. The first setup downloads the locked CPU runtime and can take several minutes.

Download and extract the repository, then double-click `run-edgelens.cmd`. On first launch it detects missing local dependencies, runs `scripts/setup-desktop.ps1`, and opens the desktop app after setup succeeds. The same command opens EdgeLens on later runs. No administrator access, global Python packages or WSL installation are required for the ordinary ONNX workflow.

The steps can also be run separately:

```powershell
.\setup-windows.cmd
.\start-desktop.cmd
```

Use `check-windows.cmd` to diagnose an existing setup. The source bootstrap creates `backend\.venv`, installs the pinned requirements and npm dependencies, builds the UI and verifies a small PT2-to-ONNX conversion. It does not change the machine's PowerShell execution policy.

The desktop engine uses a private local connection. The first-run environment check shows engine status, the local data folder and the optional TFLite conversion worker. EdgeLens never installs WSL automatically. PT2-to-TFLite conversion uses the optional Linux worker; ONNX benchmarking and supported imported TFLite inference run without it.

Models, datasets, reports and SQLite history remain on that laptop under `%APPDATA%\EdgeLens\data`; GitHub contains source code only. Close EdgeLens and back up the full data folder before a manual restore. Before a database migration the app creates and verifies a SQLite backup beside the database. An older app refuses a database with a newer schema. Removing the source checkout does not remove this app data.

The optional developer packaging command `npm run desktop:package` creates a relocatable `release/win-unpacked` folder. That folder contains an internal Electron executable and supporting runtime files; it is not a setup installer and is not published by the project. `.github/workflows/windows-desktop.yml` validates source setup and the desktop runtime on a hosted Windows runner. It does not publish a binary release or establish clean-VM installation results.

Only sample pretrained presets need access to `download.pytorch.org`; uploaded models are processed locally. Edge Impulse estimates require an explicit upload choice and project credentials. The runtime is CPU-only. No auto-update service is configured.

For teammate prerequisites, the one-click setup, model testing and troubleshooting, see [TEAM-SETUP.md](TEAM-SETUP.md).

## Developer browser preview

The desktop interface also has a browser development mode:

```powershell
npm run dev
# In a second terminal:
cd backend
$env:EDGELENS_ALLOW_CUSTOM_MODELS='1' # trusted local development only
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The Node development server on port 5173 proxies `/api` to the local backend on port 8000. Browser mode initially displays the clearly labelled demo; desktop mode opens the real local workspace.

## Optional Linux worker/server

The existing container deployment is retained for later cloud work. It is separate from the Windows desktop connection; remote-worker selection is not implemented in the desktop release.

```sh
docker compose config --quiet
docker compose up --build -d
```

Compose exposes the interface/API at `127.0.0.1:8080` and uses a named `edgelens_data` volume for datasets, SQLite, reports and weights. Default `ML_EXTRA=onnx` installs the CPU ONNX worker. `ML_EXTRA=none` installs only the API and storage. `ML_EXTRA=tflite` adds the optional Linux LiteRT Torch dependencies; this heavier configuration has not been runtime-verified here.

Run only one backend process. On a VM, use an SSH tunnel to reach `localhost:8080` for a private prototype. Shared/public service deployment needs authentication, TLS, resource-isolated jobs, per-user storage and durable job leasing; those are not implemented. Cloud CPU benchmarks must remain labelled cloud/host measurements, not physical edge-device performance.

`docker compose down` retains data. Removing its named volume deletes stored experiments. No cloud account, credentials, domain or public endpoint has been created.

## Continuous integration

The main workflow checks source assets, frontend API tests, build output, Compose configuration, and backend API/storage/statistics tests. The Windows workflow tests teammate setup in a fresh source folder containing spaces, checks a real tiny conversion and starts the native desktop in smoke-test mode. The pretrained ONNX smoke test remains opt-in because it downloads weights. These workflows validate source behavior; they do not publish an installer.
