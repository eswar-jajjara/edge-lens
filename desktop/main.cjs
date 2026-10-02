const { app, BrowserWindow, Menu, dialog, session } = require('electron');
const { spawn } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

let engine, mainWindow, shuttingDown = false;
const isSmoke = process.argv.includes('--smoke-test');
const root = path.resolve(__dirname, '..');
app.setName('EdgeLens');
if (process.env.EDGELENS_DESKTOP_DATA) app.setPath('userData', process.env.EDGELENS_DESKTOP_DATA);
if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { if (mainWindow) { if (mainWindow.isMinimized()) mainWindow.restore(); mainWindow.focus(); } });
  app.whenReady().then(start).catch(error => {
    shuttingDown = true;
    if (engine && !engine.killed) engine.kill();
    if (isSmoke) console.error(error.stack);
    else dialog.showErrorBox('EdgeLens could not start', `${error.message}\n\nSee the desktop setup instructions in README.md.`);
    app.exit(1);
  });
}

function launchEngine(token) {
  const engineDir = app.isPackaged ? path.join(process.resourcesPath, 'engine') : path.join(root, 'backend');
  const python = app.isPackaged ? path.join(engineDir, 'python.exe') : path.join(engineDir, '.venv', 'Scripts', 'python.exe');
  const ui = app.isPackaged ? path.join(process.resourcesPath, 'ui') : path.join(root, 'dist');
  if (!fs.existsSync(python) || !fs.existsSync(path.join(ui, 'index.html'))) throw new Error('The local Python engine or interface is missing. Run the project setup and build commands.');
  const dataDir = path.join(app.getPath('userData'), 'data');
  fs.mkdirSync(dataDir, { recursive: true });
  const environment = { ...process.env, APP_ENV: 'production', EDGELENS_ALLOW_CUSTOM_MODELS: '1', CORS_ORIGINS: '', EDGELENS_DATA_DIR: dataDir, EDGELENS_DESKTOP_TOKEN: token, TORCH_HOME: path.join(dataDir, 'weights'), PYTHONUTF8: '1', PYTHONUNBUFFERED: '1' };
  delete environment.PYTHONPATH;
  delete environment.PYTHONHOME;
  if (app.isPackaged) environment.PYTHONHOME = engineDir;
  engine = spawn(python, ['-u', path.join(engineDir, 'desktop_entry.py'), '--frontend-dir', ui], { cwd: engineDir, env: environment, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] });
  let recentError = '', output = '';
  engine.stderr.on('data', chunk => { recentError = (recentError + chunk.toString()).slice(-5000); });
  engine.on('exit', code => {
    if (!shuttingDown && mainWindow) {
      if (isSmoke) console.error(`ENGINE_EXIT ${code}: ${recentError}`);
      else dialog.showErrorBox('The conversion engine stopped', `Close and reopen EdgeLens. Saved reports remain in SQLite.\n\n${recentError.slice(-1800)}`);
    }
  });
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => reject(new Error(`The Python engine did not become ready. ${recentError}`)), 60000);
    engine.once('error', error => { clearTimeout(timeout); reject(error); });
    engine.once('exit', code => { clearTimeout(timeout); reject(new Error(`The Python engine exited (${code}). ${recentError}`)); });
    engine.stdout.on('data', chunk => {
      output += chunk.toString();
      const lines = output.split(/\r?\n/); output = lines.pop();
      for (const line of lines) {
        try { const value = JSON.parse(line); if (Number.isInteger(value.edgelens_port)) { clearTimeout(timeout); resolve(`http://127.0.0.1:${value.edgelens_port}`); } } catch { /* Worker progress is not the ready message. */ }
      }
    });
  });
}

async function start() {
  const token = randomBytes(32).toString('hex');
  const origin = await launchEngine(token);
  await session.defaultSession.cookies.set({ url: origin, name: 'edgelens_session', value: token, httpOnly: true, sameSite: 'strict' });
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  session.defaultSession.setPermissionCheckHandler(() => false);
  mainWindow = new BrowserWindow({ width: 1440, height: 960, minWidth: 900, minHeight: 650, show: false, title: 'EdgeLens — Model Conversion Studio', icon: app.isPackaged ? path.join(process.resourcesPath, 'icon.ico') : path.join(__dirname, 'icon.ico'), backgroundColor: '#f4f7f9', webPreferences: { nodeIntegration: false, contextIsolation: true, sandbox: true } });
  mainWindow.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  mainWindow.webContents.on('will-navigate', (event, url) => { if (new URL(url).origin !== origin) event.preventDefault(); });
  mainWindow.webContents.on('will-attach-webview', event => event.preventDefault());
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { label: 'File', submenu: [{ label: 'Reports', click: () => mainWindow.loadURL(`${origin}/#report`) }, { type: 'separator' }, { label: 'Exit', accelerator: 'Alt+F4', click: () => mainWindow.close() }] },
    { label: 'Edit', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: 'View', submenu: [{ role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'togglefullscreen' }] },
    { label: 'Help', submenu: [{ label: 'About EdgeLens', click: () => dialog.showMessageBox(mainWindow, { title: 'EdgeLens', message: `EdgeLens ${app.getVersion()} · Model Conversion Studio`, detail: 'Desktop developer tool • Local Python engine • SQLite experiment reports\nDeployment search uses calibration diagnostics and validation constraints; selection is frozen before test evaluation.\nBenchmarks describe this computer; device results require matching physical reports.\nTFLite conversion requires a Linux worker.' }) }] },
  ]));
  mainWindow.on('close', event => {
    if (shuttingDown || isSmoke) return;
    event.preventDefault();
    fetch(`${origin}/api/v1/runs`, { headers: { 'X-EdgeLens-Session': token } }).then(response => response.json()).then(async runs => {
      if (runs.some(run => ['running', 'queued'].includes(run.status))) {
        const choice = await dialog.showMessageBox(mainWindow, { type: 'question', title: 'Benchmark in progress', message: 'Closing EdgeLens will stop the active benchmark.', buttons: ['Keep working', 'Close EdgeLens'], defaultId: 0, cancelId: 0 });
        if (choice.response !== 1) return;
      }
      shuttingDown = true; mainWindow.close();
    }).catch(() => { shuttingDown = true; mainWindow.close(); });
  });
  await mainWindow.loadURL(origin);
  if (isSmoke) {
    const response = await fetch(`${origin}/api/v1/capabilities`, { headers: { 'X-EdgeLens-Session': token } });
    const capabilities = await response.json();
    const page = await mainWindow.webContents.executeJavaScript('({title:document.title,desktop:window.EDGELENS_CONFIG.desktop,heading:document.querySelector("h1").textContent,precisionControls:Boolean(document.querySelector("#run-strategy option[value=quantization_compare]")),validationControl:Boolean(document.querySelector("#validation-select")),deploymentControls:Boolean(document.querySelector("#deployment-controls")),deploymentStrategy:Boolean(document.querySelector("#run-strategy option[value=deployment_search]")),evidenceFilters:Boolean(document.querySelector("#layer-candidate") && document.querySelector("#layer-view") && document.querySelector("#more-layers")),profileTargets:Boolean(document.querySelector("#load-ei-targets") && document.querySelector("#ei-consent"))})');
    if (!page.desktop || !page.precisionControls || !page.validationControl || !page.deploymentControls || !page.deploymentStrategy || !page.evidenceFilters || !page.profileTargets || !capabilities.deployment_search?.available) throw new Error('Desktop deployment controls or local engine capabilities are missing.');
    console.log(JSON.stringify({ desktop_smoke: 'passed', capabilities, page }));
    shuttingDown = true; app.quit();
  } else mainWindow.show();
}

app.on('window-all-closed', () => app.quit());
app.on('before-quit', () => { shuttingDown = true; if (engine && !engine.killed) engine.kill(); });
app.on('quit', () => { if (engine && !engine.killed) engine.kill(); });
