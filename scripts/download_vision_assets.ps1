$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskAssets = @(
    @{ Path = 'models/yolo11n.pt'; Url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt' },
    @{ Path = 'models/bus.jpg'; Url = 'https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg' }
    @{ Path = 'models/tableware-cc0.jpg'; Url = 'https://upload.wikimedia.org/wikipedia/commons/4/40/Fiber_One_in_a_bowl_with_milk_and_a_spoon.jpg' }
)
foreach ($taskAsset in $taskAssets) {
    $taskTarget = Join-Path $taskRoot $taskAsset.Path
    New-Item -ItemType Directory -Path (Split-Path -Parent $taskTarget) -Force | Out-Null
    if (-not (Test-Path -LiteralPath $taskTarget)) {
        Invoke-WebRequest -Uri $taskAsset.Url -OutFile $taskTarget
    }
}
& (Join-Path $taskRoot '.venv/Scripts/python.exe') (Join-Path $PSScriptRoot 'download_vision_assets.py')
if ($LASTEXITCODE -ne 0) { throw 'Asset validation failed' }
