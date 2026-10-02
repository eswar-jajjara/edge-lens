@echo off
setlocal
cd /d "%~dp0"
rem Source checkout: always launch the current source, never an older packaged EXE.
if not exist "node_modules\electron\dist\electron.exe" (
  echo Double-click setup-windows.cmd first. See docs\TEAM-SETUP.md.
  pause
  exit /b 1
)
if not exist "backend\.venv\Scripts\python.exe" (
  echo Double-click setup-windows.cmd first to install the Python engine.
  pause
  exit /b 1
)
node scripts\build.mjs
if errorlevel 1 (
  echo Interface build failed. Run setup-windows.cmd and try again.
  pause
  exit /b 1
)
start "" "node_modules\electron\dist\electron.exe" .
