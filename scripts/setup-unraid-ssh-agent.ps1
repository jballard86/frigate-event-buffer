#Requires -RunAsAdministrator
<#
.SYNOPSIS
  Enable Windows OpenSSH Agent so your Unraid key passphrase is entered once per boot.

.DESCRIPTION
  Right-click PowerShell -> Run as administrator, then:
    Set-ExecutionPolicy -Scope Process Bypass -Force
    & "C:\Users\jeffb\OneDrive\Documents\Code\frigate-event-buffer\scripts\setup-unraid-ssh-agent.ps1"
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Service -Name ssh-agent -StartupType Automatic
Start-Service ssh-agent

Write-Host "ssh-agent is running (StartupType=Automatic)." -ForegroundColor Green
Write-Host ""
Write-Host "Next, in a normal (non-admin) PowerShell window, load your key once:" -ForegroundColor Cyan
Write-Host '  ssh-add $env:USERPROFILE\.ssh\id_ed25519_unraid' -ForegroundColor Yellow
Write-Host ""
Write-Host "Then test:" -ForegroundColor Cyan
Write-Host '  ssh unraid "docker ps --filter name=frigate_buffer"' -ForegroundColor Yellow
