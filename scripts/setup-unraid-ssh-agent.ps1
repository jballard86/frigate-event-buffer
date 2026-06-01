#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Enable Windows OpenSSH Agent so your Unraid key passphrase is entered once per boot.

.DESCRIPTION
  Right-click PowerShell -> Run as administrator, then:
    Set-ExecutionPolicy -Scope Process Bypass -Force
    & "C:\Users\jeffb\OneDrive\Documents\Code\frigate-event-buffer\scripts\setup-unraid-ssh-agent.ps1"

  After this one-time setup, load your key in a normal PowerShell window:
    .\scripts\load-unraid-ssh-key.ps1
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Service -Name ssh-agent -StartupType Automatic
Start-Service ssh-agent

$keyPath = Join-Path $env:USERPROFILE ".ssh\id_ed25519_unraid"
$configPath = Join-Path $env:USERPROFILE ".ssh\config"

Write-Host "ssh-agent is running (StartupType=Automatic)." -ForegroundColor Green

if (-not (Test-Path $keyPath)) {
    Write-Host "WARNING: Key not found at $keyPath" -ForegroundColor Red
}
else {
    Write-Host "Found key: $keyPath" -ForegroundColor Green
}

if (Test-Path $configPath) {
    $config = Get-Content $configPath -Raw
    if ($config -notmatch "AddKeysToAgent\s+yes") {
        Write-Host ""
        Write-Host "Tip: add 'AddKeysToAgent yes' under Host unraid in $configPath" -ForegroundColor Yellow
        Write-Host "     so the key is added to the agent on first successful use." -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Next, in a normal (non-admin) PowerShell window from the repo root:" -ForegroundColor Cyan
Write-Host "  .\scripts\load-unraid-ssh-key.ps1" -ForegroundColor Yellow
Write-Host ""
Write-Host "Then test:" -ForegroundColor Cyan
Write-Host '  ssh unraid "docker ps --filter name=frigate_buffer"' -ForegroundColor Yellow
