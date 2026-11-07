$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
if (-not (Test-Path -LiteralPath '.venv/Scripts/python.exe')) {
    python -m venv --system-site-packages .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed' }
}
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
& $taskPython -m pip install 'setuptools>=68' wheel
if ($LASTEXITCODE -ne 0) { throw 'Build tools installation failed' }
& $taskPython -m pip install --no-build-isolation -e '.[dev,vision]'
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
Write-Output 'Ready. This environment may reuse existing system CUDA/PyTorch packages.'
