# Desktop delivery and optional Linux worker

## Windows desktop (primary)

`npm run desktop:package` creates `release/win-unpacked/EdgeLens.exe` and its supporting files. Distribute the entire folder, not only the executable. The build bundles Electron, a portable copy of the active virtual environment's Python runtime, and the CPU ML packages. Python source remains available for PyTorch export introspection.

The app starts its own private local engine and automatically connects to it. Runtime data lives under `%APPDATA%/EdgeLens/data`; the installation directory contains code and libraries only. Back up that data folder while EdgeLens is closed to preserve SQLite, datasets, converted artifacts and cached weights together. It is a single-user application.

Only sample pretrained presets need initial access to `download.pytorch.org`; uploaded models run locally. Explicit Edge Impulse profiling needs its API and credentials. The runtime is CPU-only. The portable build is unsigned; a code-signing certificate, installer and auto-update service have not been configured. Packaging is separate from publishing; nothing is uploaded to a public service.

## Developer browser preview

The desktop interface also has a browser development mode:

```powershell
npm run dev
# A second terminal:
cd backend
$env:EDGELENS_ALLOW_CUSTOM_MODELS='1' # trusted local development only
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The Node development server on5173 proxies `/api` to the local backend8000. Browser mode initially displays the clearly labelled demo; desktop mode initially opens the real local workspace.

## Optional Linux worker/server

The existing container deployment is retained for later cloud work. It is separate from the Windows desktop connection; remote-worker selection is not implemented in the desktop release.

```sh
docker compose config --quiet
docker compose up --build -d
```

Compose exposes the interface/API at `127.0.0.1:8080` and uses a named `edgelens_data` volume for datasets, SQLite, reports and weights. Default `ML_EXTRA=onnx` installs the CPU ONNX worker. `ML_EXTRA=none` installs only the API and storage. `ML_EXTRA=tflite` adds the optional Linux LiteRT Torch dependencies; this heavier configuration has not been runtime-verified here.

Run only one backend process. On a VM, use an SSH tunnel to reach localhost8080 for a private prototype. Shared/public service deployment needs authentication, TLS, resource-isolated jobs, per-user storage and durable job leasing; those are not implemented. Cloud CPU benchmarks must remain labelled cloud/host measurements, not physical edge-device performance.

`docker compose down` retains data. Removing its named volume deletes stored experiments. No cloud account, credentials, domain or public endpoint has been created.

## Continuous integration

The workflow checks source assets, frontend API tests, build output, Compose configuration, and backend API/storage/statistics tests. The real ONNX smoke test is opt-in because it downloads weights. Windows executable packaging is performed locally; this workflow does not publish binaries.
