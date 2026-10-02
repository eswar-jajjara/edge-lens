# Set up EdgeLens 0.9.0 on a Windows teammate’s laptop

EdgeLens is a desktop developer tool. It runs a private Python conversion and
benchmark engine on your laptop, shows layer diagnostics, and stores experiments
in local SQLite. It accepts exported image-classification models (PT2, single-file
ONNX, TFLite) and labelled image ZIPs. Optional Edge Impulse profiling supplies
target resource estimates; it does not simulate exact ESP32 measurements.

## 1. Get the current GitHub source

The repository is public: [eswar-jajjara/edge-lens](https://github.com/eswar-jajjara/edge-lens).
Your teammate can view, clone or download it without an invitation. To contribute
directly, they need a separate collaborator invitation from the owner, or can
fork it and submit a pull request. Source access does not grant access to your
private Edge Impulse account, models, datasets or saved experiments.

Use Windows 10/11 **x64**, not Windows on ARM for this tested setup. Install:

- [Python 3.12 x64](https://www.python.org/downloads/windows/).
- [Node.js 22.12+ x64](https://nodejs.org/en/download).
- [Git for Windows](https://git-scm.com/downloads/win) if you want clone/update commands.

Restart your terminal after installation. Python 3.13/3.14 is not the tested
runtime snapshot. A GPU and physical ESP32 are not required. Allow several GB
of disk space and Internet access for the initial CPU runtime installation.

On GitHub choose **Code → Download ZIP**, extract it, and open the folder that
contains `setup-windows.cmd`. Or use PowerShell:

```powershell
cd "$env:USERPROFILE\Documents"
git clone https://github.com/eswar-jajjara/edge-lens.git
cd edge-lens
.\setup-windows.cmd
.\start-desktop.cmd
```

The repository can be on any writable drive. It does not depend on the author's
`D:\projects` directory. Do not copy a teammate's `.venv` or `node_modules`.

## 2. Install once, then open the tool

1. Double-click **setup-windows.cmd**. It creates `backend\.venv`, installs the
   pinned Python packages and npm lockfile, builds the interface and verifies a
   tiny real PT2-to-ONNX conversion. The first setup can take time downloading
   CPU PyTorch. Setup uses local dependencies and does not require a global
   Python package installation.
2. After setup succeeds, double-click **start-desktop.cmd**. It builds the current
   interface and opens the Electron desktop window with its private engine.
   Close older EdgeLens windows first: only one desktop instance is allowed.
3. Use **check-windows.cmd** to verify the existing installation. The report is
   `.cache\setup-report.json`; share the displayed error if setup fails.

If Python 3.12 is not found, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-desktop.ps1 -PythonPath "C:\path\to\Python312\python.exe"
```

The execution-policy option applies to that process only. The script does not
change your machine's policy. If a missing native DLL is reported, install/repair
the [Microsoft Visual C++ x64 runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
and rerun the check. If setup finds an incompatible existing environment, close
EdgeLens, rename `backend\.venv`, then rerun setup; it will not erase it for you.

## 3. Run your first experiment

1. Open **Overview → Local benchmark**. Upload a trusted exported classifier.
   Enter its exact input shape, layout, class count and preprocessing settings.
   A weights-only `.pth` file is not a complete architecture export.
2. Upload labelled JPG/PNG images as a ZIP with root `labels.json`, mapping each
   class folder to its model output index. Use the model's actual class order.
3. Select the model, dataset, format and deployment goal. Start with the default
   fixed-profile test. FP32/INT8 comparison and deployment search need separate
   calibration, validation and held-out test ZIPs.
4. Answer **Estimate this test in Edge Impulse? No** for the first local test.
   Run the benchmark and wait for completion.
5. Open **Layer diagnostics** to read numerical comparisons or unavailable
   mappings. **Reports & history** stores and exports the complete evidence.
   **Delete** beside a finished test asks before permanently deleting that test's
   history/artifacts; uploaded models and datasets are preserved.
6. Optional: connect in the separate **Edge Impulse** section, load targets, then
   choose **Yes**, project and target in Overview. An evaluated TFLite artifact
   is required. **Edge hardware** is for future physical board tests.

See [the developer guide](DEVELOPER-GUIDE.md) for model/dataset settings and
[the Edge Impulse walkthrough](EDGE-IMPULSE-CONNECTION.md) for browser sign-in,
project-key connection, Yes/No tests, saved estimates and deletion.

An optional reproducible 32x32 INT8 pipeline fixture is available:

```powershell
.\backend\.venv\Scripts\python.exe scripts/create-edge-profile-demo.py
```

Follow [edge profiling](EDGE-PROFILING.md) for its generated model, spec and dataset
paths. This synthetic fixture tests plumbing; its accuracy does not prove
conversion superiority.

## 4. Update from GitHub

Close EdgeLens, open the cloned repository in PowerShell, then:

```powershell
git pull --ff-only
.\setup-windows.cmd
.\start-desktop.cmd
```

If Git reports local edits, preserve and review them before updating. ZIP users
can download/extract the current source into a new folder and run setup there.
Each source launcher uses the same local data root on that laptop.

## What stays on each laptop

- Uploaded models, datasets, SQLite, reports and converted files live in
  `%APPDATA%\EdgeLens\data`. GitHub shares code, not that private data.
- API keys stay only in local engine memory and must be reconnected after
  restarting. Never commit credentials, datasets or personal reports.
- Use the same model bytes, preprocessing, labelled splits, converter settings
  and constraints for comparisons. CPU timing will differ between laptops.
- Host results are **MEASURED**, Edge Impulse results **ESTIMATED**, and physical
  device results **UNAVAILABLE** until a matching hardware report is recorded.
- Windows TFLite import/inference works; PT2-to-TFLite conversion needs the
  optional Linux worker. Raspberry Pi hardware benchmarking is future work.

## Older portable downloads

The public v0.4.1 ZIP is an earlier prototype. It has no 0.8.0 connection/Yes-No
features and cannot read upgraded schema-4 history. Use current source for this
version. Do not delete `%APPDATA%\EdgeLens\data` or lower the SQLite schema to
make an old build run. `start-desktop.cmd` starts current source even if an old
`release\win-unpacked\EdgeLens.exe` is present. Developers can create a current
portable folder with `npm run desktop:package`; no new public ZIP accompanies
this source update.


## New diagnostic and Linux-worker features

Version 0.9.0 adds **Layer diagnostics → Run seven checks**, structural graph
evidence, 100-sample timing defaults and sampled process RSS. TFLite conversion
uses a separate verified Linux worker; Windows runs evaluated TFLite files and
imports worker reports through **Reports & history**. Follow
[Stages 1–5](STAGES-1-5.md). Your teammate can reproduce the synthetic Linux test
with the GitHub workflow without uploading private models. Local WSL setup still
requires verification on their laptop. Physical ESP32 results remain unavailable.
