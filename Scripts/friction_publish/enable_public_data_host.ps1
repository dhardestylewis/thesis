<#
.SYNOPSIS
Make the austin-friction-public R2 bucket publicly readable so worldcastr.com/rezoning
can fetch its data, then verify a browser-shaped range request works.

.DESCRIPTION
Two ways to expose the bucket:

  r2.dev         no arguments. Cloudflare generates a pub-<hash>.r2.dev hostname.
                 Rate limited; Cloudflare advises against it for production traffic.
  custom domain  pass -ZoneId. Connects data.worldcastr.com, which the page is
                 already coded for, so nothing else has to change.

Find the zone id at dash.cloudflare.com, pick worldcastr.com, Overview tab, the
API section on the lower right shows Zone ID.

.EXAMPLE
  .\enable_public_data_host.ps1
  .\enable_public_data_host.ps1 -ZoneId <worldcastr.com zone id>
#>
param(
    [string]$ZoneId,
    [string]$Bucket = 'austin-friction-public',
    [string]$Domain = 'data.worldcastr.com',
    [string]$AccountId = '465fb49b30af5adbba4bf08bcf12b5ce'
)

$ErrorActionPreference = 'Stop'
$env:CLOUDFLARE_ACCOUNT_ID = $AccountId
# An API token in the environment takes precedence over the OAuth login and the
# one in .env.local has no R2 scope, so clear it and force the interactive login.
Remove-Item Env:CLOUDFLARE_API_TOKEN -ErrorAction SilentlyContinue

function Invoke-Wrangler { npx -y wrangler@latest @args }

Write-Host "Checking Cloudflare login..." -ForegroundColor Cyan
$who = Invoke-Wrangler whoami 2>&1 | Out-String
if ($who -notmatch 'You are logged in') {
    Write-Host "Not logged in. A browser tab will open; click Allow." -ForegroundColor Yellow
    Invoke-Wrangler login
    $who = Invoke-Wrangler whoami 2>&1 | Out-String
    if ($who -notmatch 'You are logged in') { throw "Login did not complete." }
}
Write-Host "Logged in." -ForegroundColor Green

if ($ZoneId) {
    Write-Host "Connecting $Domain to $Bucket..." -ForegroundColor Cyan
    Invoke-Wrangler r2 bucket domain add $Bucket --domain $Domain --zone-id $ZoneId --min-tls 1.2 -y
    $host_name = $Domain
} else {
    Write-Host "Enabling the r2.dev public URL on $Bucket..." -ForegroundColor Cyan
    Invoke-Wrangler r2 bucket dev-url enable $Bucket -y
    $status = Invoke-Wrangler r2 bucket dev-url get $Bucket 2>&1 | Out-String
    Write-Host $status
    if ($status -match '([a-z0-9-]+\.r2\.dev)') { $host_name = $Matches[1] } else { $host_name = $null }
}

if (-not $host_name) {
    Write-Host "Could not determine the public hostname. Copy it from the command output above." -ForegroundColor Yellow
    exit 1
}

# DNS and certificate issuance for a new custom domain take a few minutes.
Write-Host "`nVerifying https://$host_name ..." -ForegroundColor Cyan
$url = "https://$host_name/austin_friction_grid.f32"
$ok = $false
foreach ($attempt in 1..10) {
    try {
        # The exact request the page makes for the default 45 ft scenario
        $slice = 271567 * 3 * 4
        $start = (45 - 5) * $slice
        $req = [System.Net.HttpWebRequest]::Create($url)
        $req.Headers.Add('Range', "bytes=$start-$($start + $slice - 1)")
        $req.Headers.Add('Origin', 'https://worldcastr.com')
        $req.Timeout = 30000
        $res = $req.GetResponse()
        $len = $res.Headers['Content-Length']
        Write-Host "  status $([int]$res.StatusCode) $($res.StatusDescription)"
        Write-Host "  content-range: $($res.Headers['Content-Range'])"
        Write-Host "  bytes: $len (expected $slice)"
        Write-Host "  CORS allow-origin: $($res.Headers['Access-Control-Allow-Origin'])"
        $res.Close()
        if ([int]$len -eq $slice) { $ok = $true }
        break
    } catch {
        Write-Host "  attempt $attempt not ready yet: $($_.Exception.Message)"
        Start-Sleep -Seconds 20
    }
}

Write-Host ""
if ($ok) {
    Write-Host "Public data host is live: https://$host_name" -ForegroundColor Green
} else {
    Write-Host "Public access was configured but the range check did not pass yet." -ForegroundColor Yellow
    Write-Host "A new custom domain can take a few minutes to issue its certificate. Re-run to recheck." -ForegroundColor Yellow
}
Write-Host "Tell Claude this hostname: $host_name"
if ($host_name -ne 'data.worldcastr.com') {
    Write-Host "It differs from the baked-in data.worldcastr.com, so DATA_BASE in the page needs updating before PR #807 is merged." -ForegroundColor Yellow
}
