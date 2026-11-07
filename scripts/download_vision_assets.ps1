$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $taskRoot '.venv/Scripts/python.exe') (Join-Path $PSScriptRoot 'download_vision_assets.py')
if ($LASTEXITCODE -ne 0) { throw 'Asset download or validation failed' }
