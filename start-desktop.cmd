@echo off
cd /d "%~dp0"
if exist "release\win-unpacked\EdgeLens.exe" (
  start "" "release\win-unpacked\EdgeLens.exe"
  exit /b
)
if not exist "node_modules\electron\dist\electron.exe" (
  echo Run npm install and npm run build first. See README.md.
  pause
  exit /b 1
)
start "" "node_modules\electron\dist\electron.exe" .
