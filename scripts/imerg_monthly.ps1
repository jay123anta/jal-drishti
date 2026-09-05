# JalDrishti IMERG monthly top-up: fetch newly-published IMERG days + commit.
# IMERG Final has ~months of latency and is a validation/upstream-coverage
# reference, NOT a live model input, so it runs monthly - not in the 3-hourly
# pipeline. Registered as the Scheduled Task 'JalDrishtiIMERG' (1st of month).
# Safe to run by hand:  powershell -ExecutionPolicy Bypass -File scripts\imerg_monthly.ps1
param([switch]$NoCommit)

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "$repo\logs" | Out-Null
$log = "$repo\logs\imerg_$(Get-Date -Format yyyy-MM).log"

# IMERG_EMAIL is a persistent User env var; fall back to it if not in-process
if (-not $env:IMERG_EMAIL) {
    $env:IMERG_EMAIL = [Environment]::GetEnvironmentVariable('IMERG_EMAIL','User')
}

"=== $(Get-Date -Format s) imerg monthly start ===" | Out-File -Append -Encoding utf8 $log
cmd /c "python backend\fetch_imerg.py >> `"$log`" 2>&1"
$rc = $LASTEXITCODE
"=== $(Get-Date -Format s) fetch_imerg exit $rc ===" | Out-File -Append -Encoding utf8 $log

# refresh the three-way validation on the newly-extended history (best-effort)
if ($rc -eq 0) {
    cmd /c "python backend\three_way_rain.py >> `"$log`" 2>&1"
}

if ($rc -eq 0 -and -not $NoCommit) {
    cmd /c "python scripts\commit_archive.py >> `"$log`" 2>&1"
}

# keep 24 monthly logs
Get-ChildItem "$repo\logs\imerg_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddMonths(-24) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $rc
