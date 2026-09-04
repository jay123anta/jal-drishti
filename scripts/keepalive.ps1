# JalDrishti keep-alive: one pipeline run + daily archive commit + log rotation.
# Called by the Scheduled Task registered with install_keepalive.ps1 (every 3 h).
# Safe to run by hand:  powershell -ExecutionPolicy Bypass -File scripts\keepalive.ps1
param([switch]$NoCommit)

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "$repo\logs" | Out-Null
$log = "$repo\logs\pipeline_$(Get-Date -Format yyyy-MM-dd).log"

"=== $(Get-Date -Format s) run start ===" | Out-File -Append -Encoding utf8 $log
# cmd redirection avoids PowerShell 5.1 NativeCommandError noise on stderr
cmd /c "python backend\run_pipeline.py >> `"$log`" 2>&1"
$rc = $LASTEXITCODE
"=== $(Get-Date -Format s) pipeline exit $rc ===" | Out-File -Append -Encoding utf8 $log

if ($rc -eq 0 -and -not $NoCommit) {
    cmd /c "python scripts\commit_archive.py >> `"$log`" 2>&1"
}

# keep 30 days of logs
Get-ChildItem "$repo\logs\pipeline_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $rc
