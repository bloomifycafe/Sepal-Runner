# Run a command while holding the shared GPU lock (atomic mkdir of runs\gpu.lock.d), waiting for it if somebody else has it.
#
#   powershell -File tools\runner_oracle\with_gpu_lock.ps1 -Agent rt-oracle -Job "golden served" -Command "<program> <args...>" [-RequireIdle]
#
# The lock directory holds one file (owner.txt: agent, start time, job). It is removed when the command ends, however it ends. A lock older
# than 45 minutes with no GPU compute process in nvidia-smi is stale and is taken over. With -RequireIdle the command only starts when
# nvidia-smi lists no compute process (a timing run must not share the GPU); it waits for that as well.
param(
    [Parameter(Mandatory = $true)][string]$Agent,
    [Parameter(Mandatory = $true)][string]$Job,
    [Parameter(Mandatory = $true)][string]$Command,
    [switch]$RequireIdle,
    [int]$PollSeconds = 3,
    [int]$GiveUpMinutes = 240
)
$lock = "E:\MYCOMPANY\BlumSOC\Sepal\runs\gpu.lock.d"
function Compute-Apps { @(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>$null | Where-Object { $_ -and $_ -notmatch "No devices" }) }
$deadline = (Get-Date).AddMinutes($GiveUpMinutes)
$held = $false
while (-not $held) {
    if (Test-Path $lock) {
        $age = (Get-Date) - (Get-Item $lock).LastWriteTime
        if ($age.TotalMinutes -gt 45 -and (Compute-Apps).Count -eq 0) {
            Write-Host "stale lock ($([int]$age.TotalMinutes) min, no compute process): removing"
            Remove-Item -Recurse -Force $lock
        }
    }
    try { New-Item -ItemType Directory -Path $lock -ErrorAction Stop | Out-Null; $held = $true } catch { }
    if (-not $held) {
        if ((Get-Date) -gt $deadline) { Write-Host "gave up waiting for the GPU lock"; exit 3 }
        Start-Sleep -Seconds $PollSeconds
    }
}
"$Agent $(Get-Date -Format s) job $Job" | Set-Content (Join-Path $lock "owner.txt")
try {
    if ($RequireIdle) {
        $until = (Get-Date).AddMinutes(30)
        while ((Compute-Apps).Count -gt 0 -and (Get-Date) -lt $until) { Start-Sleep -Seconds 5 }
        $foreign = Compute-Apps
        if ($foreign.Count -gt 0) { Write-Host "foreign GPU compute process while holding the lock: $($foreign -join '; ')" }
    }
    Write-Host "lock taken $(Get-Date -Format s); running: $Command"
    Invoke-Expression $Command
    $code = $LASTEXITCODE
}
finally {
    Remove-Item -Recurse -Force $lock -ErrorAction SilentlyContinue
    Write-Host "lock released $(Get-Date -Format s)"
}
exit $code
