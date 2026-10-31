# Set up EdgeLens 0.10.0 on a Windows teammate’s laptop

EdgeLens is a desktop developer tool for image-classifier conversion and validation. It runs its Python engine locally, shows layer diagnostics and stores experiments in SQLite. It supports exported PT2, single-file ONNX and TFLite classifiers with labelled image datasets. Edge Impulse can provide separate target estimates; those estimates are not ESP32 measurements.

## Quick setup (no separate installer EXE)

EdgeLens is shared as source from [the public GitHub repository](https://github.com/eswar-jajjara/edge-lens). There is no separate EdgeLens setup executable to download. The Windows desktop runtime still includes Electron internally; the teammate setup uses the source launcher rather than an installer program.

Use Windows 10/11 x64. Install these once:

- [Python 3.12 x64](https://www.python.org/downloads/windows/). Enable the Python launcher or add Python to PATH.
- [Node.js 22.12 or later, x64](https://nodejs.org/en/download).
- [Git for Windows](https://git-scm.com/downloads/win) only if you want to clone and update from a terminal.

No administrator access, GPU, ESP32 board or WSL installation is needed for setup and ONNX tests. Allow several GB of disk space and Internet access for the initial CPU PyTorch and npm downloads. Restart Windows Terminal after installing Python or Node.js.

Get the source by either method:

1. On GitHub, choose **Code → Download ZIP**, extract it, and open the extracted `edge-lens` folder; or
2. Open PowerShell and run:

   ```powershell
   cd "$env:USERPROFILE\Documents"
   git clone https://github.com/eswar-jajjara/edge-lens.git
   cd edge-lens
   ```

Double-click **`run-edgelens.cmd`**. On the first launch it installs the pinned local dependencies, builds the interface and checks a small PT2-to-ONNX conversion; then it opens the desktop tool. This first setup can take several minutes. Next time, double-click the same file to open EdgeLens.

For separate steps, double-click `setup-windows.cmd` once and then `start-desktop.cmd`. `check-windows.cmd` checks an existing setup and writes a diagnostic report to `.cache\setup-report.json`. The setup script creates `backend\.venv` inside this project, uses the npm lockfile and does not change the computer’s PowerShell execution policy. It does not install Python packages globally.

Keep the source checkout in any writable folder, including a path with spaces. Do not copy another person’s `.venv` or `node_modules`; each laptop creates its own. If Python 3.12 is not detected, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-desktop.ps1 -PythonPath "C:\path\to\Python312\python.exe"
```

If setup reports a missing Visual C++ DLL, install or repair the [Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) and run setup again. If an existing environment is incompatible, close EdgeLens, rename `backend\.venv`, and run `setup-windows.cmd`; it does not erase that environment automatically.

## Your first model test

1. Open **Overview → Local benchmark** and select a trusted exported classifier. Enter its input shape, layout, class count and preprocessing settings. A weights-only `.pth` file is not a complete model export.
2. Upload labelled JPG/PNG images as a ZIP with a root `labels.json` mapping each class folder to its output index. Use the exact class order expected by the model.
3. Select the model, dataset, conversion format and deployment goal. Start with **Fixed profiles / benchmark existing model**. Use separate calibration, validation and held-out test datasets for precision comparisons or deployment search.
4. Choose **No · benchmark on this laptop only** for the first test. Run the benchmark and wait for completion.
5. Open **Layer diagnostics** for operation-level findings and **Reports & history** for saved metrics and report export. A mapping marked unavailable means the tool cannot make an aligned comparison for that layer; it is not evidence of zero error.
6. For internet comparisons, use the measured JSON/CSV exports and the checked examples in [`reference-examples.json`](reference-examples.json). Keep those published references labelled with their source and conditions; do not present unrelated hardware, data or preprocessing values as a controlled comparison.

For detailed setup values, dataset preparation and interpretation, see the [developer guide](DEVELOPER-GUIDE.md), [manual software test sequence](MANUAL-SOFTWARE-TESTS.md), and [reviewer walkthrough](REVIEWER-TEST-WALKTHROUGH.md). The [final comparison workflow](FINAL-COMPARISON-WORKFLOW.md) explains how to assemble a report from the saved runs and published references.

## TFLite conversion and edge estimates

Windows can run supported imported TFLite models. PyTorch-to-TFLite conversion uses the optional Linux worker documented in [Stages 1–5](STAGES-1-5.md); WSL is not installed automatically, and WSL conversion on a teammate’s laptop has not been claimed as verified. The source setup alone does not install that worker.

The **Edge Impulse** page connects projects with a project API key, loads available targets and keeps provider estimates separate from local measurements. Follow the [connection guide](EDGE-IMPULSE-CONNECTION.md) or the [ONNX profiling guide](EDGE-IMPULSE-ONNX.md). Choosing **Yes** for an experiment uploads its evaluated artifact after the local benchmark. Do not upload private artifacts unless intended. Physical-device timing and memory remain **UNAVAILABLE** until a matching real-device report is recorded; selecting Raspberry Pi or ESP32 as a goal does not simulate that hardware.

To remove a saved upload from the dropdowns, open **Manage saved models and datasets · SQLite** under the experiment setup and choose **Remove from SQLite**. This removes the library record only. The original model or dataset file in the data folder and the dataset/model snapshots in completed reports are preserved. A model or dataset used by a queued or running test cannot be removed until that test finishes. Short ID suffixes distinguish duplicate display names in the dropdowns.

## Data, updates and sharing

- Models, datasets, SQLite, reports and converted artifacts stay under `%APPDATA%\EdgeLens\data` on that laptop. GitHub shares the source only.
- Edge Impulse keys are local connection credentials; teammates connect their own project keys. Never commit API keys, model weights, private datasets or personal reports.
- Close EdgeLens and back up the full data folder before a manual restore. Before a schema migration, the app creates and verifies a SQLite backup next to the database. An older app refuses a newer database schema.
- To update a Git clone, close EdgeLens and run `git pull --ff-only`, then double-click `run-edgelens.cmd`. To update a ZIP checkout, download and extract the latest source into a new folder and run the launcher there. Keep the data folder intact; it is separate from the source folder.
- To contribute directly, the repository owner must invite the teammate as a collaborator. Otherwise, fork the public repository and submit a pull request.

The older [v0.4.1 portable ZIP](https://github.com/eswar-jajjara/edge-lens/releases/tag/v0.4.1) predates later features and database schemas. Use the current GitHub source and `run-edgelens.cmd` for this project version. Do not downgrade the database or delete `%APPDATA%\EdgeLens\data` to make the old build open.

## Optional developer packaging

Developers can run `npm run desktop:package` to assemble the relocatable `release\win-unpacked` folder for local inspection. It contains the Electron executable plus its runtime files and is not an installer or the recommended teammate setup. GitHub Actions validates the source setup and desktop smoke test but does not publish an installer executable. The optional container deployment for future cloud work is described in [DEPLOYMENT.md](DEPLOYMENT.md).
