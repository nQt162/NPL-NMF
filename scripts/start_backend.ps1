[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8001
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$backendRoot = Join-Path $projectRoot 'backend'
$python = Join-Path $backendRoot '.venv\Scripts\python.exe'
$uvicorn = Join-Path $backendRoot '.venv\Scripts\uvicorn.exe'
$launcher = Join-Path $backendRoot 'run_backend.py'

function Test-CommandPath {
    param([string]$CommandLine, [string]$Path)
    return $CommandLine -match ('(?i)(?:^|[\s"])' + [regex]::Escape($Path) + '(?=[\s"]|$)')
}

function Test-BackendRoot {
    param($Process)
    if ($Process.Name -notmatch '^(python(w)?|uvicorn)\.exe$') { return $false }
    $command = [string]$Process.CommandLine
    $usesLauncher = Test-CommandPath $command $launcher
    $usesUvicorn = (
        ($command -match '(?i)(?:^|\s)-m\s+uvicorn(?:\s|$)' -or (Test-CommandPath $command $uvicorn)) -and
        $command -match '(?i)(?:^|[\s"])app\.main:app(?=[\s"]|$)'
    )
    $usesProjectPython = (
        [string]$Process.ExecutablePath -ieq $python -or
        [string]$Process.ExecutablePath -ieq $uvicorn -or
        (Test-CommandPath $command $python) -or
        (Test-CommandPath $command $uvicorn)
    )
    if (-not ($usesLauncher -or ($usesUvicorn -and $usesProjectPython))) { return $false }

    $portOption = [regex]::Match($command, '(?i)(?:^|\s)--port(?:\s+|=)"?(\d+)(?:"|\s|$)')
    if ($portOption.Success) { return [int]$portOption.Groups[1].Value -eq $Port }
    $defaultPort = if ($usesLauncher) { 8001 } else { 8000 }
    return $defaultPort -eq $Port
}

function Test-PortAvailable {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    try {
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
        return $true
    }
    catch [System.Net.Sockets.SocketException] {
        if ($_.Exception.NativeErrorCode -eq 10048) { return $false }
        if ($_.Exception.NativeErrorCode -eq 10013) {
            throw "Windows denied access to port $Port (10013). Check reserved ports or security settings."
        }
        throw
    }
    finally { $listener.Stop() }
}

try {
    foreach ($path in @($python, $launcher, (Join-Path $backendRoot 'app\main.py'))) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Required file missing: $path. Install the backend environment as described in README.md."
        }
    }

    Write-Host "Checking project backend on 127.0.0.1:$Port ..."
    try { $processes = @(Get-CimInstance Win32_Process) }
    catch { throw 'Cannot inspect process ownership. No processes were stopped. Check Windows permissions.' }

    $processById = @{}
    $ownedIds = @{}
    foreach ($process in $processes) {
        $processById[[int]$process.ProcessId] = $process
        if (Test-BackendRoot $process) { $ownedIds[[int]$process.ProcessId] = $true }
    }

    # Windows venv launchers and Uvicorn reloaders can have a separate Python child.
    do {
        $added = $false
        foreach ($process in $processes) {
            $id = [int]$process.ProcessId
            $parentId = [int]$process.ParentProcessId
            if ($ownedIds.ContainsKey($id) -or -not $ownedIds.ContainsKey($parentId)) { continue }
            if ($process.CreationDate -lt $processById[$parentId].CreationDate) { continue }
            $ownedIds[$id] = $true
            $added = $true
        }
    } while ($added)

    # Stop parents first so a reloader cannot replace a worker during cleanup.
    $targets = @($processes | Where-Object { $ownedIds.ContainsKey([int]$_.ProcessId) } | Sort-Object CreationDate)
    foreach ($target in $targets) {
        $live = Get-CimInstance Win32_Process -Filter "ProcessId = $($target.ProcessId)"
        if ($null -eq $live -or $live.CreationDate -ne $target.CreationDate) { continue }
        Write-Host "Stopping old project backend PID $($target.ProcessId) ..."
        try { Stop-Process -Id $target.ProcessId -ErrorAction Stop }
        catch {
            if (Get-Process -Id $target.ProcessId -ErrorAction SilentlyContinue) { throw }
        }
    }

    $available = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if (Test-PortAvailable) { $available = $true; break }
        Start-Sleep -Milliseconds 250
    }
    if (-not $available) {
        throw "Port $Port is still occupied by another or unverified process. It was NOT stopped. Inspect it with: netstat -ano | findstr :$Port"
    }

    Write-Host "Starting backend from $backendRoot"
    Write-Host "Health: http://127.0.0.1:$Port/health"
    Write-Host 'Keep this terminal open. Press Ctrl+C to stop. Restarting cancels unfinished ablation jobs.'
    & $python -u $launcher --port $Port
    exit $LASTEXITCODE
}
catch {
    Write-Host "Backend startup failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
