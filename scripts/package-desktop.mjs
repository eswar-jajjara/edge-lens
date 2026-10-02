// Electron's documented manual distribution layout; no installer or signing step.
import { cp, mkdir, readFile, copyFile, unlink, writeFile, stat, realpath, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
const root = path.resolve(fileURLToPath(new URL('../', import.meta.url)));
if (process.platform !== 'win32') throw new Error('This package target is Windows.');
const output = path.join(root, 'release', 'win-unpacked');
const electron = path.join(root, 'node_modules', 'electron', 'dist');
const engine = path.join(root, 'desktop', 'engine');
await stat(path.join(engine, 'python.exe'));
await stat(path.join(root, 'dist', 'index.html'));
await mkdir(output, { recursive: true });
await cp(electron, output, { recursive: true });
await copyFile(path.join(output, 'electron.exe'), path.join(output, 'EdgeLens.exe'));
await unlink(path.join(output, 'electron.exe'));
const resources = path.join(output, 'resources');
await mkdir(resources, { recursive: true });
if ((await realpath(resources)).toLowerCase() !== path.resolve(resources).toLowerCase()) throw new Error('Refusing an unexpected package resource path.');
for (const name of ['app', 'engine', 'ui']) {
  const generated = path.join(resources, name);
  try {
    if ((await realpath(generated)).toLowerCase() !== path.resolve(generated).toLowerCase()) throw new Error('Refusing to replace a linked package directory.');
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  await rm(generated, { recursive: true, force: true });
}
const appDir = path.join(resources, 'app');
await mkdir(path.join(appDir, 'desktop'), { recursive: true });
await copyFile(path.join(root, 'desktop', 'main.cjs'), path.join(appDir, 'desktop', 'main.cjs'));
await copyFile(path.join(root, 'desktop', 'preload.cjs'), path.join(appDir, 'desktop', 'preload.cjs'));
const source = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'));
await writeFile(path.join(appDir, 'package.json'), JSON.stringify({ name: source.name, productName: 'EdgeLens', version: source.version, main: 'desktop/main.cjs' }, null, 2));
await cp(engine, path.join(resources, 'engine'), { recursive: true });
await cp(path.join(root, 'dist'), path.join(resources, 'ui'), { recursive: true });
await cp(path.join(root, 'docs'), path.join(output, 'docs'), { recursive: true });
await copyFile(path.join(root, 'desktop', 'icon.ico'), path.join(resources, 'icon.ico'));
await writeFile(path.join(output, 'READ ME.txt'), `EdgeLens ${source.version} — Model Conversion Studio\r\n\r\nWindows 10/11 x64. Extract this entire folder, then double-click EdgeLens.exe.\r\nKeep all supporting files together. Python, Node.js and a GPU are not needed separately.\r\nCPU PyTorch/ONNX, LiteRT inference and USB serial support are included.\r\nIf startup fails, double-click check-runtime.cmd and read docs/TEAM-SETUP.md.\r\nUpload your own trusted PT2, ONNX or TFLite classifier. Sample presets may download pretrained weights.\r\nYour data is saved separately in %APPDATA%\\EdgeLens\\data on this laptop.\r\nLatency differs between computers. TFLite conversion needs an optional Linux worker.\r\nThis is an unsigned portable prototype, not a signed installer.\r\n`);
await writeFile(path.join(output, 'check-runtime.cmd'), [
  '@echo off', 'setlocal', 'cd /d "%~dp0"',
  'set "PYTHONHOME=%~dp0resources\\engine"', 'set "PYTHONPATH="',
  '"%~dp0resources\\engine\\python.exe" "%~dp0resources\\engine\\check_install.py" --frontend-dir "%~dp0resources\\ui" --report "%LOCALAPPDATA%\\EdgeLens\\setup-report.json"',
  'if errorlevel 1 (echo Runtime check failed. See the error above.)',
  'echo Diagnostic report: %LOCALAPPDATA%\\EdgeLens\\setup-report.json', 'pause', '',
].join('\r\n'));
console.log(`Portable desktop application: ${path.join(output, 'EdgeLens.exe')}`);
