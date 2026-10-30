@echo off
setlocal
cd /d "%~dp0"

rem First launch installs the pinned local runtimes; later launches open EdgeLens.
if not exist "backend\.venv\Scripts\python.exe" goto :setup
if not exist "node_modules\electron\dist\electron.exe" goto :setup
goto :launch

:setup
echo First run: setting up EdgeLens for this Windows account.
echo This may take several minutes and downloads the CPU PyTorch runtime.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-desktop.ps1"
if errorlevel 1 goto :failed

:launch
call "%~dp0start-desktop.cmd"
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo EdgeLens could not start. Read docs\TEAM-SETUP.md for troubleshooting.
pause
exit /b 1
