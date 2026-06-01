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
    [ValidateSet("status", "restart", "rebuild", "logs")]
    [string] $Action = "status",

    [string] $HostAlias = "unraid",

    [string] $Service = "frigate_buffer",

    [string] $RepoPath = "/mnt/user/appdata/frigate_buffer"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-UnraidSsh {
    param([Parameter(Mandatory = $true)][string] $RemoteCommand)
    & ssh $HostAlias $RemoteCommand
    if ($LASTEXITCODE -ne 0) {
        throw "ssh $HostAlias failed with exit code $LASTEXITCODE"
    }
}

switch ($Action) {
    "status" {
        Invoke-UnraidSsh "docker ps -a --filter name=$Service --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'"
    }
    "restart" {
        Invoke-UnraidSsh "docker restart $Service"
        Invoke-UnraidSsh "docker ps --filter name=$Service --format 'table {{.Names}}\t{{.Status}}'"
    }
    "rebuild" {
        Invoke-UnraidSsh "cd $RepoPath && docker compose up -d --build $Service"
        Invoke-UnraidSsh "docker ps --filter name=$Service --format 'table {{.Names}}\t{{.Status}}'"
    }
    "logs" {
        Invoke-UnraidSsh "docker logs -f --tail 100 $Service"
    }
}
