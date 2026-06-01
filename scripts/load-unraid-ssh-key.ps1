<#
.SYNOPSIS
  Load the Unraid SSH key into ssh-agent and verify Tower connectivity.

.DESCRIPTION
  Run once per Windows boot (or when Cursor/scripts report "Permission denied").
  You will be prompted for your key passphrase interactively.

.EXAMPLE
  .\scripts\load-unraid-ssh-key.ps1
#>
[CmdletBinding()]
param(
    [string] $KeyPath = "$env:USERPROFILE\.ssh\id_ed25519_unraid",
    [string] $HostAlias = "unraid"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $KeyPath)) {
    throw "SSH key not found: $KeyPath"
}

$agent = Get-Service ssh-agent -ErrorAction SilentlyContinue
if (-not $agent) {
    throw "OpenSSH Authentication Agent (ssh-agent) is not installed."
}
if ($agent.Status -ne "Running") {
    Write-Host "Starting ssh-agent..." -ForegroundColor Yellow
    Start-Service ssh-agent
}

$loaded = ssh-add -l 2>&1 | Out-String
if ($loaded -match "The agent has no identities") {
    Write-Host "Loading key (enter passphrase when prompted):" -ForegroundColor Cyan
    Write-Host "  $KeyPath" -ForegroundColor Yellow
    & ssh-add $KeyPath
    if ($LASTEXITCODE -ne 0) {
        throw "ssh-add failed with exit code $LASTEXITCODE"
    }
}
elseif ($loaded -match "Could not open a connection to your authentication agent") {
    throw "ssh-agent is not reachable. Run scripts/setup-unraid-ssh-agent.ps1 as Administrator."
}
else {
    Write-Host "Key already loaded in ssh-agent:" -ForegroundColor Green
    ssh-add -l
}

Write-Host ""
Write-Host "Testing SSH to $HostAlias..." -ForegroundColor Cyan
& ssh -o BatchMode=yes -o ConnectTimeout=10 $HostAlias "echo SSH OK && hostname"
if ($LASTEXITCODE -ne 0) {
    throw "SSH test failed. Check ~/.ssh/config and that your public key is on Tower."
}

Write-Host ""
Write-Host "SSH is ready. Example:" -ForegroundColor Green
Write-Host "  ssh $HostAlias `"docker ps --filter name=frigate_buffer`"" -ForegroundColor Yellow
