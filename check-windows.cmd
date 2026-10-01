@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-desktop.ps1" -VerifyOnly
if errorlevel 1 (
  echo.
  echo Check failed. Read the message above or run setup-windows.cmd.
  pause
  exit /b 1
)
pause
