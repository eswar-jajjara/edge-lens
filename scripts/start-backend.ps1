$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$backendPath = Join-Path $projectRoot 'backend'
$pythonPath = Join-Path $backendPath '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    Write-Host 'First set up the backend environment using the commands in README.md.'
    exit 1
}
$env:EDGELENS_ALLOW_CUSTOM_MODELS = '1'
Push-Location $backendPath
try { & $pythonPath -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 }
finally { Pop-Location }
