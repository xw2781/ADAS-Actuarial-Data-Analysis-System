<#
.SYNOPSIS
Reorganize an ArcRho Server deploy folder so core stays source-only.

.DESCRIPTION
Moves generated build output, virtual environments, component runtime instance
state, config, shortcuts, and Python cache files out of core. The top-level
requests folder is intentionally left unchanged.

Default deploy root:
  E:\ArcRho Server

Examples:
  .\tools\reorganize_deploy_layout.ps1 -DryRun
  .\tools\reorganize_deploy_layout.ps1 -DeployRoot "E:\ArcRho Server"
#>

[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$DeployRoot = "E:\ArcRho Server",

  [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Components = @("arcrho_admin", "arcrho_engine", "arcrho_launcher", "arcrho_orchestrator")

function Assert-PathInside {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Child,

    [Parameter(Mandatory = $true)]
    [string]$Parent
  )

  $childPath = [System.IO.Path]::GetFullPath($Child).TrimEnd('\')
  $parentPath = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\')
  if (
    -not $childPath.Equals($parentPath, [System.StringComparison]::OrdinalIgnoreCase) -and
    -not $childPath.StartsWith($parentPath + "\", [System.StringComparison]::OrdinalIgnoreCase)
  ) {
    throw "Refusing to operate outside deploy root. Child='$childPath', Parent='$parentPath'."
  }
}

function Invoke-LoggedAction {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Message,

    [Parameter(Mandatory = $true)]
    [scriptblock]$Action
  )

  if ($DryRun) {
    Write-Host "[dry-run] $Message"
  } else {
    Write-Host $Message
    & $Action
  }
}

function Ensure-Directory {
  param([Parameter(Mandatory = $true)][string]$Path)

  if (-not (Test-Path -LiteralPath $Path)) {
    Invoke-LoggedAction "Create directory: $Path" {
      New-Item -ItemType Directory -Path $Path -Force | Out-Null
    }
  }
}

function Move-PathIfPresent {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Source,

    [Parameter(Mandatory = $true)]
    [string]$Destination,

    [Parameter(Mandatory = $true)]
    [string]$DeployRootPath
  )

  if (-not (Test-Path -LiteralPath $Source)) {
    return
  }

  Assert-PathInside -Child $Source -Parent $DeployRootPath
  Assert-PathInside -Child $Destination -Parent $DeployRootPath

  if (Test-Path -LiteralPath $Destination) {
    $sourceItem = Get-Item -LiteralPath $Source -Force
    if ($sourceItem.PSIsContainer) {
      $sourceFiles = @(
        Get-ChildItem -LiteralPath $Source -Force -Recurse -File -ErrorAction SilentlyContinue |
          Select-Object -First 1
      )
      if ($sourceFiles.Count -eq 0) {
        Invoke-LoggedAction "Remove source tree with no files already present at destination: $Source" {
          Remove-Item -LiteralPath $Source -Recurse -Force
        }
        return
      }
    }

    throw "Destination already exists and source is not empty. Resolve manually before moving. Source='$Source', Destination='$Destination'."
  }

  $destinationParent = Split-Path -Parent $Destination
  Ensure-Directory -Path $destinationParent
  Invoke-LoggedAction "Move: $Source -> $Destination" {
    Move-Item -LiteralPath $Source -Destination $Destination
  }
}

function Remove-PathIfPresent {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [Parameter(Mandatory = $true)]
    [string]$DeployRootPath
  )

  if (-not (Test-Path -LiteralPath $Path)) {
    return
  }

  Assert-PathInside -Child $Path -Parent $DeployRootPath
  Invoke-LoggedAction "Remove generated item: $Path" {
    Remove-Item -LiteralPath $Path -Recurse -Force
  }
}

$deployRootPath = (Resolve-Path -LiteralPath $DeployRoot).ProviderPath
$coreRoot = Join-Path $deployRootPath "core"
if (-not (Test-Path -LiteralPath $coreRoot -PathType Container)) {
  throw "Deploy core folder was not found: $coreRoot"
}

Write-Host "Deploy root: $deployRootPath"
Write-Host "Leaving requests unchanged: $(Join-Path $deployRootPath 'requests')"

$configRoot = Join-Path $deployRootPath "config"
$runtimeRoot = Join-Path $deployRootPath "runtime"
$buildsRoot = Join-Path $deployRootPath "builds"
$venvsRoot = Join-Path $deployRootPath "venvs"
$shortcutsRoot = Join-Path $deployRootPath "shortcuts"

foreach ($path in @($configRoot, $runtimeRoot, (Join-Path $runtimeRoot "instances"), $buildsRoot, $venvsRoot, $shortcutsRoot)) {
  Ensure-Directory -Path $path
}

Move-PathIfPresent `
  -Source (Join-Path $coreRoot "config.json") `
  -Destination (Join-Path $configRoot "config.json") `
  -DeployRootPath $deployRootPath

foreach ($shortcut in @(Get-ChildItem -LiteralPath $deployRootPath -Filter "*.lnk" -Force -File -ErrorAction SilentlyContinue)) {
  Move-PathIfPresent -Source $shortcut.FullName -Destination (Join-Path $shortcutsRoot $shortcut.Name) -DeployRootPath $deployRootPath
}

foreach ($shortcut in @(Get-ChildItem -LiteralPath $coreRoot -Filter "*.lnk" -Force -File -ErrorAction SilentlyContinue)) {
  Move-PathIfPresent -Source $shortcut.FullName -Destination (Join-Path $shortcutsRoot $shortcut.Name) -DeployRootPath $deployRootPath
}

foreach ($component in $Components) {
  $componentRoot = Join-Path $coreRoot $component
  if (-not (Test-Path -LiteralPath $componentRoot -PathType Container)) {
    continue
  }

  Move-PathIfPresent -Source (Join-Path $componentRoot ".venv") -Destination (Join-Path $venvsRoot $component) -DeployRootPath $deployRootPath
  Move-PathIfPresent -Source (Join-Path $componentRoot "build") -Destination (Join-Path $buildsRoot "$component\build") -DeployRootPath $deployRootPath
  Move-PathIfPresent -Source (Join-Path $componentRoot "dist") -Destination (Join-Path $buildsRoot "$component\dist") -DeployRootPath $deployRootPath
  Move-PathIfPresent -Source (Join-Path $componentRoot "spec") -Destination (Join-Path $buildsRoot "$component\spec") -DeployRootPath $deployRootPath
  Move-PathIfPresent -Source (Join-Path $componentRoot "instances") -Destination (Join-Path $runtimeRoot "instances\$component") -DeployRootPath $deployRootPath
}

foreach ($cache in @(Get-ChildItem -LiteralPath $coreRoot -Recurse -Force -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue)) {
  Remove-PathIfPresent -Path $cache.FullName -DeployRootPath $deployRootPath
}

foreach ($thumbs in @(Get-ChildItem -LiteralPath $deployRootPath -Recurse -Force -File -Filter "Thumbs.db" -ErrorAction SilentlyContinue)) {
  Remove-PathIfPresent -Path $thumbs.FullName -DeployRootPath $deployRootPath
}

Write-Host "Deploy layout reorganization complete."
