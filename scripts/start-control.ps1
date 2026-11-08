[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8766
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
$baseUrl = "http://127.0.0.1:$Port"
$operatorUrl = "$baseUrl/operator"

function Get-ControlStatus {
    try {
        $status = Invoke-RestMethod -Uri "$baseUrl/api/status" -TimeoutSec 2
    }
    catch {
        return $null
    }

    if (-not $status.control_token -or -not $status.mode -or
        $status.hardware_motion_enabled -isnot [bool]) {
        throw "Unexpected status response at $baseUrl/api/status. Select a different port with -Port."
    }
    return $status
}

$startupLock = [System.Threading.Mutex]::new($false, "Local\PetMealControlStart-$Port")
$lockHeld = $false
try {
    try {
        $lockHeld = $startupLock.WaitOne(25000)
    }
    catch [System.Threading.AbandonedMutexException] {
        $lockHeld = $true
    }
    if (-not $lockHeld) {
        throw "Another startup is still in progress on port $Port."
    }

    $status = Get-ControlStatus
    $reused = $null -ne $status
    $result = [ordered]@{
        url = $baseUrl
        operator_url = $operatorUrl
        reused = $reused
    }

    if (-not $reused) {
        $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
        if ($Port -in $listeners.Port) {
            throw "Port $Port already has a listener without a healthy Pet Meal status endpoint. Select a different port with -Port."
        }
        if (-not (Test-Path -LiteralPath $taskPython -PathType Leaf)) {
            throw 'Python environment missing. Run .\scripts\bootstrap.ps1 from the repository root.'
        }

        $logDir = Join-Path $taskRoot 'artifacts/local-server'
        New-Item -Path $logDir -ItemType Directory -Force | Out-Null
        $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
        $result.stdout = Join-Path $logDir "monitor-$Port-$stamp.out.log"
        $result.stderr = Join-Path $logDir "monitor-$Port-$stamp.err.log"
        $process = Start-Process -FilePath $taskPython -WorkingDirectory $taskRoot `
            -ArgumentList @('-u', '-m', 'pet_meal', 'monitor', '--port', "$Port") `
            -WindowStyle Hidden -RedirectStandardOutput $result.stdout `
            -RedirectStandardError $result.stderr -PassThru
        $result.pid = $process.Id

        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        do {
            $status = Get-ControlStatus
            if ($null -ne $status -or $process.HasExited) { break }
            Start-Sleep -Milliseconds 200
        } while ($timer.Elapsed.TotalSeconds -lt 15)

        if ($null -eq $status) {
            throw "Control service did not become ready on port $Port. See $($result.stderr)."
        }
    }

    $operatorPage = Invoke-WebRequest -Uri $operatorUrl -TimeoutSec 3 -UseBasicParsing
    if ($operatorPage.StatusCode -ne 200 -or $operatorPage.Content -notlike '*src="/operator.js"*') {
        throw "The operator page at $operatorUrl did not return the Pet Meal interface."
    }
    $result.running = $true
    $result.mode = $status.mode
    $result.status = $status.status
    $result.active = $status.active
    $result.hardware_motion_enabled = $status.hardware_motion_enabled

    if (-not $reused) {
        $startupFile = if ($Port -eq 8766) { 'startup.json' } else { "startup-$Port.json" }
        $result | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir $startupFile) -Encoding UTF8
    }
    $result | ConvertTo-Json
}
finally {
    if ($lockHeld) { $startupLock.ReleaseMutex() }
    $startupLock.Dispose()
}
