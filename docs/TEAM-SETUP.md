# Run EdgeLens on a teammate's Windows laptop

Use Windows 10/11 **x64**. The portable release and source setup both use the same pinned CPU PyTorch, ONNX, TFLite inference and USB runtime versions. Neither needs a GPU. Windows on ARM is not a verified target for this build.

## Easiest: ready-to-run desktop ZIP

For Phase 2 deployment search, use **current source** as described below. The
v0.4.1 ZIP runs the earlier feature set and cannot read Phase 1/2 databases.

1. Open [EdgeLens v0.4.1](https://github.com/eswar-jajjara/edge-lens/releases/tag/v0.4.1).
2. Download **EdgeLens-0.4.1-windows-x64.zip** under Assets. GitHub's automatically generated **Source code** ZIP is the developer source, without installed runtimes.
3. Extract the complete ZIP to a writable local folder, for example `Documents\EdgeLens`. Do not run inside the ZIP or copy only the EXE.
4. Double-click **EdgeLens.exe**. Python, Electron and the model runtimes are included; Python and Node.js do not need to be installed separately.
5. If it does not open, run **check-runtime.cmd**. Send the displayed error and `%LOCALAPPDATA%\EdgeLens\setup-report.json` to the team. Check that all files were extracted and that the laptop is x64. This is an unsigned prototype; do not disable Windows security software to run it.

The initial download is large because it includes CPU PyTorch. Allow about 3 GB free disk space for the ZIP and extracted folder. An Internet connection is needed for the download, built-in pretrained model downloads, or optional Edge Impulse profiling. Uploaded local models can be benchmarked without an Internet connection after extraction.

If a runtime check reports a missing DLL or Visual C++ runtime, install or repair the [Microsoft Visual C++ v14 Redistributable for x64](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist), then reopen EdgeLens. This Windows system dependency is commonly installed already; it is separate from Python/Node.js. Use Microsoft's installer and follow your laptop's normal software permissions.

## Develop from GitHub source

1. Install [64-bit Python 3.12](https://www.python.org/downloads/windows/) and [64-bit Node.js 22.12 or newer](https://nodejs.org/en/download). Restart the terminal after installation. Python 3.13/3.14 is not the tested runtime for this snapshot.
2. Clone the repository, or choose **Code → Download ZIP** and extract it. The project can live on any drive; it does not require the original author's `D:\projects` folder.
3. Double-click **setup-windows.cmd**. It creates `backend\.venv`, installs the exact Python snapshot and npm lockfile, builds the interface and performs a tiny real PT2-to-ONNX conversion check. The first setup downloads large packages; keep Internet access available. It does not install global packages or require an administrator account.
4. After setup succeeds, double-click **start-desktop.cmd**. It builds the current interface, opens the desktop app and starts the private Python engine automatically.
5. Run **check-windows.cmd** to verify dependencies again without installing or changing them. Its report is `.cache\setup-report.json`.

For a terminal-based setup:

```powershell
cd "C:\path\to\edge-lens"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-desktop.ps1
.\start-desktop.cmd
```

The command's execution-policy setting applies only to that setup process. The script does not change the machine's PowerShell policy. Managed school/company devices may require their administrator's normal software approval.

If the Python launcher cannot find your Python 3.12 installation:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-desktop.ps1 -PythonPath "C:\path\to\Python312\python.exe"
```

Do not copy another laptop's `.venv` or `node_modules`; create them with setup. The setup refuses to reuse an incompatible Python environment instead of deleting it. If you intentionally installed a different Python environment, rename its `backend\.venv` folder while EdgeLens is closed, then rerun setup.

## If an older build reports a newer database

The v0.4.1 portable build cannot read the newer databases used by v0.5.0/0.6.0.
If the error says "This database was created by a newer EdgeLens version",
close that build and use the current source version. From the project folder,
run `npm run desktop`, or use the updated `start-desktop.cmd` after source setup.
The source launcher always starts the current source, even when an older
`release/win-unpacked/EdgeLens.exe` exists.

Keep `%APPDATA%/EdgeLens/data`; do not delete the database or lower its schema
version. The current application can open the upgraded database and preserve
saved reports. A portable build must be rebuilt from the current source using
`npm run desktop:package` before its EXE can use Phase 1/2 data. Version 0.6.0
adds SQLite migration 4, preserving previous reports and recording deployment
selections and operation sensitivity results.

## What is shared and what is local

- **Shared:** application code, algorithms, supported conversion profiles and pinned runtime versions.
- **Local to each laptop:** uploaded models, datasets, SQLite database, history, generated reports and converted artifacts under `%APPDATA%\EdgeLens\data`. Downloading GitHub source does not download someone else's experiments.
- **Compare results fairly:** use the same model file, preprocessing, calibration/validation/test ZIPs, conversion profile, constraints and settings. Numerical results may have small differences across CPUs. Latency changes with CPU speed, thread settings and background load; identical software does not guarantee identical timing.
- **Share an experiment:** exchange the original model and labelled datasets separately, or export the HTML/CSV/JSON report. Close EdgeLens before deliberately copying/backing up the complete data folder; do not merge individual SQLite files.
- **ESP32:** firmware preparation is included. Actual flashing still needs the board's correct USB driver, ESP-IDF toolchain and chip selection. A generated package is not a measured board result.
- **Windows TFLite:** model import/inference is included. PyTorch-to-TFLite conversion still requires the separate optional Linux worker.

The runtime diagnostic uses a tiny synthetic classifier and verifies conversion plumbing. It is not evidence that the converter improves a developer's model accuracy. See [the developer guide](DEVELOPER-GUIDE.md) to run your own labelled classifier validation.
