$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\backend"

$pythonCommand = "py"
try {
    & $pythonCommand --version | Out-Null
} catch {
    $pythonCommand = "python"
}

if (-not (Test-Path ".venv")) {
    & $pythonCommand -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
