[CmdletBinding()]
param(
    [string]$PythonPath,
    [switch]$VerifyOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
$pythonExe = Join-Path $projectRoot 'backend\.venv\Scripts\python.exe'
$requirements = Join-Path $projectRoot 'backend\requirements-windows-tested.txt'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Program failed (exit $LASTEXITCODE)." }
}

function Assert-Python {
    param([string]$Program, [string[]]$Prefix = @())
    $result = & $Program @Prefix -c 'import sys,struct; print(sys.version_info.major*10000+sys.version_info.minor*100+struct.calcsize(chr(80))*8)' 2>$null
    if ($LASTEXITCODE -ne 0 -or "$result".Trim() -ne '31264') {
        throw 'Use 64-bit Python 3.12 for this tested Windows build. Install it from https://www.python.org/downloads/windows/ and restart the terminal.'
    }
}

try {
    if ($env:OS -ne 'Windows_NT' -or -not [Environment]::Is64BitOperatingSystem) {
        throw 'This setup supports Windows 10/11 x64. Use the portable Windows x64 release on a compatible laptop.'
    }
    Write-Host 'EdgeLens Windows setup: dependencies and data stay separate.'
    $node = Get-Command node.exe -ErrorAction SilentlyContinue
    $npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $node -or -not $npm) {
        throw 'Install 64-bit Node.js 22.12 or newer from https://nodejs.org/en/download, restart the terminal, then run setup-windows.cmd again. The portable release does not need Node.js.'
    }
    $nodeVersion = & $node.Source --version
    if ($LASTEXITCODE -ne 0 -or [Version]("$nodeVersion".Trim().TrimStart('v')) -lt [Version]'22.12.0') { throw 'Node.js 22.12 or newer is required.' }
    $nodeArchitecture = & $node.Source -p 'process.arch'
    if ($LASTEXITCODE -ne 0 -or "$nodeArchitecture".Trim() -ne 'x64') { throw 'Use the Windows x64 Node.js installer.' }
    if (Test-Path -LiteralPath $pythonExe) {
        Assert-Python -Program $pythonExe
    } else {
        if ($VerifyOnly) { throw 'The backend environment is missing. Run setup-windows.cmd first.' }
        $prefix = @()
        if ($PythonPath) {
            $selectedPython = $PythonPath
        } else {
            $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
            if ($launcher) { $selectedPython = $launcher.Source; $prefix = @('-3.12') }
            else {
                $pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
                if (-not $pythonCommand) { throw 'Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/ and restart the terminal.' }
                $selectedPython = $pythonCommand.Source
            }
        }
        Assert-Python -Program $selectedPython -Prefix $prefix
        Write-Host 'Creating the project Python environment...'
        Invoke-Checked -Program $selectedPython -Arguments ($prefix + @('-m', 'venv', (Join-Path $projectRoot 'backend\.venv')))
    }
    Push-Location $projectRoot
    try {
        if (-not $VerifyOnly) {
            $env:PIP_NO_INPUT = '1'
            $env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
            Invoke-Checked -Program $pythonExe -Arguments @('-m', 'ensurepip', '--upgrade')
            Write-Host 'Installing the tested CPU PyTorch pair (large first download)...'
            Invoke-Checked -Program $pythonExe -Arguments @('-m', 'pip', 'install', 'torch==2.14.0+cpu', 'torchvision==0.29.0+cpu', '--index-url', 'https://download.pytorch.org/whl/cpu')
            Write-Host 'Installing the tested ONNX, TFLite inference, API and USB packages...'
            Invoke-Checked -Program $pythonExe -Arguments @('-m', 'pip', 'install', '-r', $requirements)
            Write-Host 'Installing the locked desktop dependencies...'
            Invoke-Checked -Program $npm.Source -Arguments @('ci', '--no-audit', '--no-fund')
            Write-Host 'Downloading the locked Electron desktop runtime...'
            Invoke-Checked -Program $node.Source -Arguments @('node_modules/electron/install.js')
            Invoke-Checked -Program $node.Source -Arguments @('scripts/build.mjs')
        }
        if (-not (Test-Path -LiteralPath (Join-Path $projectRoot 'node_modules\electron\dist\electron.exe'))) {
            throw 'Electron is missing. Run setup-windows.cmd to install the desktop dependencies.'
        }
        if (-not $VerifyOnly) { Invoke-Checked -Program $pythonExe -Arguments @('-m', 'pip', 'check') }
        $report = Join-Path $projectRoot '.cache\setup-report.json'
        Invoke-Checked -Program $pythonExe -Arguments @('backend/check_install.py', '--frontend-dir', (Join-Path $projectRoot 'dist'), '--report', $report)
        Write-Host "Verified. Diagnostic report: $report"
        Write-Host 'Open start-desktop.cmd. SQLite and model data are private to this laptop under %APPDATA%\EdgeLens\data.'
    } finally { Pop-Location }
} catch {
    Write-Host "Setup/check failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
