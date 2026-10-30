@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-desktop.ps1" %*
if errorlevel 1 (
  echo.
  echo Setup failed. Read the message above and docs\TEAM-SETUP.md.
  pause
  exit /b 1
)
echo.
echo Setup complete. Double-click run-edgelens.cmd to open EdgeLens.
pause
