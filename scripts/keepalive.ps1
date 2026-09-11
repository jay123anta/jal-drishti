# JalDrishti keep-alive: one pipeline run + archive commit + log rotation.
# Called by two Scheduled Tasks registered with install_keepalive.ps1:
#   JalDrishtiPipeline - every 3 h on the clock
#   JalDrishtiCatchup  - when the laptop wakes, is unlocked or someone logs on,
#                        so a run missed while it was asleep happens straight
#                        away instead of up to an hour later
# Safe to run by hand:  powershell -ExecutionPolicy Bypass -File scripts\keepalive.ps1
#   -NoCommit                   run the pipeline but do not commit or push
#   -SkipIfFresherThanHours N   do nothing if a run that SUCCEEDED started under N h ago
#   -DryRun                     go through every check, start nothing, change nothing
param(
    [switch]$NoCommit,
    [double]$SkipIfFresherThanHours = 0,
    [switch]$DryRun
)

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
New-Item -ItemType Directory -Force -Path "$repo\logs" | Out-Null
$log = "$repo\logs\pipeline_$(Get-Date -Format yyyy-MM-dd).log"
# start time of the last run that exited 0 (logs\ is gitignored)
$stamp = "$repo\logs\last_success.txt"

function Note([string]$msg) {
    "=== $(Get-Date -Format s) $msg ===" | Out-File -Append -Encoding utf8 $log
}

# Freshness guard. Keyed on the last SUCCESSFUL run, so a run that was killed
# part-way (laptop went to sleep, 30-minute limit hit) never counts as done.
if ($SkipIfFresherThanHours -gt 0 -and (Test-Path $stamp)) {
    try {
        $last = [datetime]::ParseExact((Get-Content $stamp -Raw).Trim(), "s",
                                       [Globalization.CultureInfo]::InvariantCulture)
        $age = ((Get-Date) - $last).TotalHours
        if ($age -lt $SkipIfFresherThanHours) {
            Note ("skipped: last good run started {0:N1} h ago, under {1} h" -f $age, $SkipIfFresherThanHours)
            exit 0
        }
    } catch {
        Note "last_success.txt unreadable - running anyway"
    }
}

# One run at a time across both tasks and any manual run. The second caller
# leaves quietly rather than two pipelines writing the same files at once.
$mutex = New-Object System.Threading.Mutex($false, "Local\JalDrishtiPipeline")
$got = $false
try { $got = $mutex.WaitOne(0) } catch [System.Threading.AbandonedMutexException] { $got = $true }
if (-not $got) { Note "skipped: another run is already in progress"; exit 0 }

$rc = 1
try {
    $started = Get-Date -Format s
    Note "run start"
    if ($DryRun) {
        Note "dry run: checks passed, pipeline not started"
        $rc = 0
    } else {
        # cmd redirection avoids PowerShell 5.1 NativeCommandError noise on stderr
        cmd /c "python backend\run_pipeline.py >> `"$log`" 2>&1"
        $rc = $LASTEXITCODE
    }
    Note "pipeline exit $rc"

    if ($rc -eq 0 -and -not $DryRun) {
        $started | Out-File -Encoding ascii $stamp
        if (-not $NoCommit) {
            cmd /c "python scripts\commit_archive.py >> `"$log`" 2>&1"
        }
    }
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}

# keep 30 days of logs
Get-ChildItem "$repo\logs\pipeline_*.log" -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } |
    Remove-Item -Force -ErrorAction SilentlyContinue

exit $rc
