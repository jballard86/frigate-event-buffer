<#
.SYNOPSIS
  Run Docker Compose actions on Unraid via SSH (Host alias: unraid).

.EXAMPLE
  .\scripts\unraid-docker.ps1 -Action status
  .\scripts\unraid-docker.ps1 -Action restart
  .\scripts\unraid-docker.ps1 -Action rebuild
  .\scripts\unraid-docker.ps1 -Action logs
#>
[CmdletBinding()]
param(
    [ValidateSet("status", "restart", "rebuild", "logs", "backup-config")]
    [string] $Action = "status",

    [string] $HostAlias = "unraid",

    [string] $Service = "frigate_buffer",

    # Git repo root (Dockerfile + src/). Storage/config live in frigate_buffer (underscore).
    [string] $RepoPath = "/mnt/user/appdata/frigate-buffer",

    [string] $AppdataPath = "/mnt/user/appdata/frigate_buffer"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Test-UnraidSshReady {
    $agentList = ssh-add -l 2>&1 | Out-String
    if ($agentList -match "The agent has no identities|Could not open a connection") {
        Write-Host "SSH key is not loaded in ssh-agent." -ForegroundColor Red
        Write-Host "Run this in PowerShell (enter your key passphrase once):" -ForegroundColor Yellow
        Write-Host "  .\scripts\load-unraid-ssh-key.ps1" -ForegroundColor Yellow
        throw "Unraid SSH is not ready."
    }
}

function Invoke-UnraidSsh {
    param([Parameter(Mandatory = $true)][string] $RemoteCommand)
    & ssh $HostAlias $RemoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "ssh $HostAlias failed with exit code $LASTEXITCODE"
    }
}

Test-UnraidSshReady

switch ($Action) {
    "status" {
        Invoke-UnraidSsh "docker ps -a --filter name=$Service --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"
    }
    "restart" {
        Invoke-UnraidSsh "docker restart $Service"
        Invoke-UnraidSsh "docker ps --filter name=$Service --format 'table {{.Names}}\t{{.Status}}'"
    }
    "rebuild" {
        Invoke-UnraidSsh "cd $RepoPath && git pull && docker compose up -d --build $Service"
        Invoke-UnraidSsh "docker ps --filter name=$Service --format 'table {{.Names}}\t{{.Status}}'"
    }
    "logs" {
        Invoke-UnraidSsh "docker logs -f --tail 100 $Service"
    }
    "backup-config" {
        Invoke-UnraidSsh "cp -a $AppdataPath/config.yaml $AppdataPath/config.yaml.backup && test -f $AppdataPath/.env && cp -a $AppdataPath/.env $AppdataPath/.env.backup || true && ls -la $AppdataPath/config.yaml $AppdataPath/config.yaml.backup $AppdataPath/.env.backup 2>/dev/null || ls -la $AppdataPath/config.yaml $AppdataPath/config.yaml.backup"
    }
}
