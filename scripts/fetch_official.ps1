$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskLock = Get-Content -LiteralPath (Join-Path $taskRoot 'external.lock.json') -Raw | ConvertFrom-Json
foreach ($taskRepo in $taskLock.repositories) {
    $taskDestination = Join-Path $taskRoot $taskRepo.checkout
    if (Test-Path -LiteralPath $taskDestination) {
        $taskRevision = git -C $taskDestination rev-parse HEAD
        if ($LASTEXITCODE -ne 0 -or $taskRevision -ne $taskRepo.commit) {
            throw "Existing checkout differs from locked revision: $($taskRepo.name)"
        }
        Write-Output "Already present: $($taskRepo.name) @ $taskRevision"
        continue
    }
    git clone --no-checkout --filter=blob:none $taskRepo.url $taskDestination
    if ($LASTEXITCODE -ne 0) { throw "Clone failed: $($taskRepo.name)" }
    git -C $taskDestination checkout --detach $taskRepo.commit
    if ($LASTEXITCODE -ne 0) { throw "Checkout failed: $($taskRepo.name)" }
}
