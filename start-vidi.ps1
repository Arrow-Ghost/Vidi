<#
    Start Vidi.

    Written because the servers kept "going down". Two different causes:

      1. They were started from a tool session, so they died with it.
         This script launches them DETACHED - they keep running after the
         window that started them is gone.

      2. The backend and frontend were sometimes started in different modes
         (one TLS, one plain). The page then loads and hangs on "connecting"
         forever, which is indistinguishable from the backend being down.
         This script starts BOTH the same way, always.

    It also regenerates the certificate when your LAN address has changed,
    which happens every time DHCP hands out a new lease and silently breaks
    phone access.

    Usage:   .\start-vidi.ps1          start both
             .\start-vidi.ps1 -Stop    stop both
             .\start-vidi.ps1 -Status  show what is running
#>
param(
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Python = Join-Path $Root 'backend\.venv\Scripts\python.exe'
$CertDir = Join-Path $Root 'certs'
$KeyFile = Join-Path $CertDir 'dev-key.pem'
$CertFile = Join-Path $CertDir 'dev-cert.pem'
$BackendPort = 8000
$FrontendPort = 4321

function Get-LanIp {
    $probe = New-Object System.Net.Sockets.Socket('InterNetwork', 'Dgram', 'Udp')
    try {
        $probe.Connect('8.8.8.8', 80)
        return $probe.LocalEndPoint.Address.IPAddressToString
    } finally { $probe.Dispose() }
}

function Get-PortOwner([int]$Port) {
    $c = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { return $c.OwningProcess }
    return $null
}

function Stop-Servers {
    foreach ($port in @($BackendPort, $FrontendPort)) {
        $owner = Get-PortOwner $port
        if ($owner) {
            Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
            Write-Host "  stopped port $port (PID $owner)"
        } else {
            Write-Host "  port $port already free"
        }
    }
}

function Show-Status {
    $ip = Get-LanIp
    foreach ($p in @(@{n='backend';port=$BackendPort}, @{n='frontend';port=$FrontendPort})) {
        $owner = Get-PortOwner $p.port
        if ($owner) { Write-Host ("  {0,-9} RUNNING  port {1}  PID {2}" -f $p.n, $p.port, $owner) }
        else { Write-Host ("  {0,-9} DOWN     port {1}" -f $p.n, $p.port) }
    }
    Write-Host ""
    Write-Host "  Desktop : https://localhost:$FrontendPort"
    Write-Host "  Phone   : https://${ip}:$FrontendPort"
}

if ($Status) { Show-Status; exit 0 }
if ($Stop)   { Write-Host "Stopping Vidi..."; Stop-Servers; exit 0 }

# ---------------------------------------------------------------- certificate
$ip = Get-LanIp
$needCert = $true
if ((Test-Path $CertFile) -and (Test-Path $KeyFile)) {
    # Does the existing certificate still cover the address we have TODAY?
    $covered = & $Python -c @"
import ssl, sys
try:
    d = ssl._ssl._test_decode_cert(r'$CertFile')
    alts = [v for k, v in d.get('subjectAltName', ()) if k == 'IP Address']
    sys.stdout.write('yes' if '$ip' in alts else 'no')
except Exception:
    sys.stdout.write('no')
"@
    if ($covered -eq 'yes') { $needCert = $false }
}

if ($needCert) {
    Write-Host "LAN address is $ip and the certificate does not cover it - regenerating."
    & $Python (Join-Path $Root 'scripts\setup_mobile.py') | Out-Null
    Write-Host "  certificate updated"
} else {
    Write-Host "Certificate already covers $ip"
}

# ---------------------------------------------------------------- restart
Write-Host "Stopping anything already running..."
Stop-Servers
Start-Sleep -Seconds 2

Write-Host "Starting backend (HTTPS)..."
Start-Process -FilePath $Python `
    -ArgumentList @(
        '-m', 'uvicorn', 'app.main:app',
        '--host', '0.0.0.0', '--port', "$BackendPort",
        '--ssl-keyfile', $KeyFile, '--ssl-certfile', $CertFile,
        '--log-level', 'warning'
    ) `
    -WorkingDirectory (Join-Path $Root 'backend') `
    -WindowStyle Hidden

Write-Host "Starting frontend (HTTPS)..."
Start-Process -FilePath 'cmd.exe' `
    -ArgumentList @('/c', 'npx astro dev --host --port ' + $FrontendPort) `
    -WorkingDirectory (Join-Path $Root 'frontend') `
    -WindowStyle Hidden

# ---------------------------------------------------------------- wait
Write-Host "Waiting for both to come up..."
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Seconds 2
    if ((Get-PortOwner $BackendPort) -and (Get-PortOwner $FrontendPort)) { $ready = $true; break }
}

Write-Host ""
if ($ready) {
    Write-Host "Vidi is running." -ForegroundColor Green
} else {
    Write-Host "Timed out waiting. Run '.\start-vidi.ps1 -Status' to check." -ForegroundColor Yellow
}
Write-Host ""
Show-Status
Write-Host ""
Write-Host "On the phone, trust the backend certificate FIRST:"
Write-Host "  1. https://${ip}:$BackendPort/health   -> Advanced -> Proceed"
Write-Host "  2. https://${ip}:$FrontendPort          -> Advanced -> Proceed"
Write-Host ""
Write-Host "Both are HTTPS. An http:// address will not work - browsers block"
Write-Host "the camera on insecure origins."
