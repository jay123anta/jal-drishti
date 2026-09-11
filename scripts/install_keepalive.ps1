# Register (or remove) the JalDrishti keep-alive Scheduled Tasks on this machine.
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_keepalive.ps1            # every 3 h
#   powershell -ExecutionPolicy Bypass -File scripts\install_keepalive.ps1 -EveryHours 6
#   powershell -ExecutionPolicy Bypass -File scripts\install_keepalive.ps1 -Uninstall
#   powershell -ExecutionPolicy Bypass -File scripts\install_keepalive.ps1 -Status
#
# Two tasks, both running scripts\keepalive.ps1 as the current user:
#
#   JalDrishtiPipeline  every N hours on the clock. Wakes the machine if the
#                       power plan allows it, runs on battery, never overlaps
#                       itself, 30-minute limit. Skips if a good run started
#                       under N-1 hours ago, so a wake-up catch-up and the next
#                       clock slot do not run back to back.
#   JalDrishtiCatchup   fires when the laptop wakes from sleep, is unlocked or
#                       someone logs on, after a 2-minute pause for the network
#                       to reconnect. Runs only if the last good run started N
#                       or more hours ago - that is, a slot was missed while the
#                       laptop slept - so an ordinary unlock costs nothing.
#                       Windows' own "start when available" catch-up was seen
#                       lagging 30-60 minutes behind waking; this closes that.
#
# The two cannot collide: keepalive.ps1 holds a lock for the whole run.
# No admin rights needed.
param(
    [int]$EveryHours = 3,
    [switch]$Uninstall,
    [switch]$Status
)

$task = "JalDrishtiPipeline"
$catch = "JalDrishtiCatchup"
$repo = Split-Path -Parent $PSScriptRoot
$script = "$repo\scripts\keepalive.ps1"

if ($Status) {
    $missing = $false
    foreach ($n in @($task, $catch)) {
        $t = Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue
        if (-not $t) { Write-Host "Task '$n' is NOT installed."; $missing = $true; continue }
        $i = Get-ScheduledTaskInfo -TaskName $n
        Write-Host "Task '$n': state=$($t.State) last=$($i.LastRunTime) result=$($i.LastTaskResult) next=$($i.NextRunTime)"
    }
    $stampFile = "$repo\logs\last_success.txt"
    if (Test-Path $stampFile) { Write-Host "Last good run started: $((Get-Content $stampFile -Raw).Trim())" }
    Get-ChildItem "$repo\logs\pipeline_*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1 |
        ForEach-Object { Write-Host "Latest log: $($_.FullName)"; Get-Content $_.FullName -Tail 3 }
    if ($missing) { exit 1 } else { exit 0 }
}

if ($Uninstall) {
    foreach ($n in @($task, $catch)) {
        Unregister-ScheduledTask -TaskName $n -Confirm:$false -ErrorAction SilentlyContinue
        Write-Host "Task '$n' removed."
    }
    exit 0
}

$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { Write-Error "python not found on PATH"; exit 1 }

$baseArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$script`""
$settingsCommon = @{
    AllowStartIfOnBatteries = $true
    DontStopIfGoingOnBatteries = $true
    MultipleInstances = "IgnoreNew"
    ExecutionTimeLimit = (New-TimeSpan -Minutes 30)
}

# ---- clock task ----
$fresh = [Math]::Max(0, $EveryHours - 1)
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "$baseArgs -SkipIfFresherThanHours $fresh" -WorkingDirectory $repo
# No -RepetitionDuration: on Windows 10/11 that means "repeat indefinitely"
# ([TimeSpan]::MaxValue serialises to an XML duration the scheduler rejects).
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
    -RepetitionInterval (New-TimeSpan -Hours $EveryHours)
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun @settingsCommon

# ---- catch-up task ----
# Wake, unlock and logon. Event and session triggers are not offered by
# New-ScheduledTaskTrigger, so they are built as CIM trigger objects.
$user = "$env:USERDOMAIN\$env:USERNAME"
$ns = "Root/Microsoft/Windows/TaskScheduler"
# Kernel-Power 107 = resumed from sleep, 507 = left Modern Standby;
# Power-Troubleshooter 1 = returned from a low-power state.
$query = "<QueryList><Query Id=`"0`" Path=`"System`"><Select Path=`"System`">" +
         "*[System[(Provider[@Name='Microsoft-Windows-Kernel-Power'] and (EventID=107 or EventID=507))" +
         " or (Provider[@Name='Microsoft-Windows-Power-Troubleshooter'] and EventID=1)]]" +
         "</Select></Query></QueryList>"
$wake = New-CimInstance -CimClass (Get-CimClass -Namespace $ns -ClassName MSFT_TaskEventTrigger) -ClientOnly
$wake.Enabled = $true
$wake.Subscription = $query
$wake.Delay = "PT2M"
$unlock = New-CimInstance -CimClass (Get-CimClass -Namespace $ns -ClassName MSFT_TaskSessionStateChangeTrigger) -ClientOnly
$unlock.Enabled = $true
$unlock.StateChange = 8          # TASK_SESSION_UNLOCK
$unlock.UserId = $user
$unlock.Delay = "PT2M"
$logon = New-ScheduledTaskTrigger -AtLogOn -User $user
$logon.Delay = "PT2M"
$catchAction = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "$baseArgs -SkipIfFresherThanHours $EveryHours" -WorkingDirectory $repo
$catchSettings = New-ScheduledTaskSettingsSet @settingsCommon

try {
    Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger `
        -Settings $settings -Force -ErrorAction Stop | Out-Null
    Register-ScheduledTask -TaskName $catch -Action $catchAction -Trigger @($wake, $unlock, $logon) `
        -Settings $catchSettings -Force -ErrorAction Stop | Out-Null
} catch {
    Write-Error "Task registration FAILED: $($_.Exception.Message)"
    exit 1
}
foreach ($n in @($task, $catch)) {
    if (-not (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue)) {
        Write-Error "Task '$n' not found after registration"; exit 1
    }
}
$i = Get-ScheduledTaskInfo -TaskName $task
Write-Host "Task '$task' installed: every $EveryHours h, next run $($i.NextRunTime), skips if a good run started under $fresh h ago"
Write-Host "Task '$catch' installed: on wake, unlock or logon; runs only if the last good run is $EveryHours h or older"
Write-Host "Logs: $repo\logs\pipeline_<date>.log   Check: -Status   Remove: -Uninstall"
