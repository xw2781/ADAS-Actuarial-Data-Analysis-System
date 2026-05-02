<#
.SYNOPSIS
Mirror the actively tested ArcRho core source into data-engine/src.

.DESCRIPTION
Copies the three data-engine component source folders from an active ArcRho
Server core tree into the repository layout. Generated/runtime artifacts are
excluded so commits stay focused on source code.

Default source:
  E:\ArcRho Server\core

Default destination:
  <repo>\data-engine\src

Examples:
  .\tools\sync_data_engine_core.ps1
  .\tools\sync_data_engine_core.ps1 -DryRun
  .\tools\sync_data_engine_core.ps1 -CoreSourceRoot "E:\ArcRho Server\core"
#>

[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$CoreSourceRoot = "E:\ArcRho Server\core",

  [string]$DestinationRoot = "",

  [switch]$DryRun,

  [switch]$KeepLegacyFolders
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ComponentMappings = @(
  @{ Source = "arcrho_engine"; Destination = "arcrho_engine" },
  @{ Source = "arcrho_launcher"; Destination = "arcrho_launcher" },
  @{ Source = "arcrho_orchestrator"; Destination = "arcrho_orchestrator" }
)

$LegacyFolders = @("agent", "master", "shell", "installer")
$ExcludedDirectories = @(".venv", "build", "dist", "instances", "__pycache__", "spec")
$ExcludedFileNames = @("Thumbs.db", "Desktop.ini", ".DS_Store")
$ExcludedFileExtensions = @(".pyc", ".pyo", ".pyd", ".exe", ".log")

function Resolve-RepoRoot {
  $scriptRoot = Split-Path -Parent $PSCommandPath
  return (Resolve-Path (Join-Path $scriptRoot "..")).ProviderPath
}

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
    throw "Refusing to write outside destination root. Child='$childPath', Parent='$parentPath'."
  }
}

function Get-RelativePath {
  param(
    [Parameter(Mandatory = $true)]
    [string]$Path,

    [Parameter(Mandatory = $true)]
    [string]$BasePath
  )

  $fullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
  $fullBase = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\')
  if ($fullPath.Equals($fullBase, [System.StringComparison]::OrdinalIgnoreCase)) {
    return ""
  }
  if (-not $fullPath.StartsWith($fullBase + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Path '$fullPath' is not under '$fullBase'."
  }
  return $fullPath.Substring($fullBase.Length).TrimStart('\')
}

function Test-ExcludedRelativePath {
  param([string]$RelativePath)

  if (-not $RelativePath) {
    return $false
  }

  $parts = @($RelativePath -split '[\\/]') | Where-Object { $_ }
  foreach ($part in $parts) {
    foreach ($excludedDir in $ExcludedDirectories) {
      if ($part.Equals($excludedDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        return $true
      }
    }
  }

  $leaf = Split-Path -Leaf $RelativePath
  foreach ($excludedName in $ExcludedFileNames) {
    if ($leaf.Equals($excludedName, [System.StringComparison]::OrdinalIgnoreCase)) {
      return $true
    }
  }

  $extension = [System.IO.Path]::GetExtension($leaf)
  foreach ($excludedExtension in $ExcludedFileExtensions) {
    if ($extension.Equals($excludedExtension, [System.StringComparison]::OrdinalIgnoreCase)) {
      return $true
    }
  }

  return $false
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

function Copy-FileIfChanged {
  param(
    [Parameter(Mandatory = $true)]
    [string]$SourceFile,

    [Parameter(Mandatory = $true)]
    [string]$DestinationFile
  )

  $destinationParent = Split-Path -Parent $DestinationFile
  if (-not (Test-Path -LiteralPath $destinationParent)) {
    Invoke-LoggedAction "Create directory: $destinationParent" {
      New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    }
  }

  $shouldCopy = $true
  if (Test-Path -LiteralPath $DestinationFile) {
    $sourceHash = (Get-FileHash -LiteralPath $SourceFile -Algorithm SHA256).Hash
    $destinationHash = (Get-FileHash -LiteralPath $DestinationFile -Algorithm SHA256).Hash
    $shouldCopy = ($sourceHash -ne $destinationHash)
  }

  if ($shouldCopy) {
    Invoke-LoggedAction "Copy file: $SourceFile -> $DestinationFile" {
      Copy-Item -LiteralPath $SourceFile -Destination $DestinationFile -Force
    }
  }
}

function Sync-Directory {
  param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDirectory,

    [Parameter(Mandatory = $true)]
    [string]$DestinationDirectory,

    [Parameter(Mandatory = $true)]
    [string]$DestinationRoot
  )

  Assert-PathInside -Child $DestinationDirectory -Parent $DestinationRoot

  if (-not (Test-Path -LiteralPath $SourceDirectory -PathType Container)) {
    throw "Source component directory does not exist: $SourceDirectory"
  }

  if (-not (Test-Path -LiteralPath $DestinationDirectory)) {
    Invoke-LoggedAction "Create directory: $DestinationDirectory" {
      New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
    }
  }

  $expected = New-Object 'System.Collections.Generic.HashSet[string]' ([System.StringComparer]::OrdinalIgnoreCase)
  [void]$expected.Add("")

  $sourceItems = @(
    Get-ChildItem -LiteralPath $SourceDirectory -Recurse -Force |
      Where-Object {
        $relativePath = Get-RelativePath -Path $_.FullName -BasePath $SourceDirectory
        -not (Test-ExcludedRelativePath -RelativePath $relativePath)
      }
  )

  foreach ($item in $sourceItems) {
    $relativePath = Get-RelativePath -Path $item.FullName -BasePath $SourceDirectory
    [void]$expected.Add($relativePath)
  }

  $destinationItems = @(
    Get-ChildItem -LiteralPath $DestinationDirectory -Recurse -Force -ErrorAction SilentlyContinue |
      Sort-Object { $_.FullName.Length } -Descending
  )

  foreach ($item in $destinationItems) {
    $relativePath = Get-RelativePath -Path $item.FullName -BasePath $DestinationDirectory
    if (-not $expected.Contains($relativePath)) {
      Assert-PathInside -Child $item.FullName -Parent $DestinationRoot
      Invoke-LoggedAction "Remove destination-only item: $($item.FullName)" {
        Remove-Item -LiteralPath $item.FullName -Recurse -Force
      }
    }
  }

  foreach ($directory in ($sourceItems | Where-Object { $_.PSIsContainer })) {
    $relativePath = Get-RelativePath -Path $directory.FullName -BasePath $SourceDirectory
    $destinationPath = Join-Path $DestinationDirectory $relativePath
    if (-not (Test-Path -LiteralPath $destinationPath)) {
      Invoke-LoggedAction "Create directory: $destinationPath" {
        New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
      }
    }
  }

  foreach ($file in ($sourceItems | Where-Object { -not $_.PSIsContainer })) {
    $relativePath = Get-RelativePath -Path $file.FullName -BasePath $SourceDirectory
    $destinationPath = Join-Path $DestinationDirectory $relativePath
    Copy-FileIfChanged -SourceFile $file.FullName -DestinationFile $destinationPath
  }
}

$repoRoot = Resolve-RepoRoot
if (-not $DestinationRoot) {
  $DestinationRoot = Join-Path $repoRoot "data-engine\src"
}

$sourceRootPath = (Resolve-Path -LiteralPath $CoreSourceRoot).ProviderPath
$destinationRootPath = [System.IO.Path]::GetFullPath($DestinationRoot)

if (-not (Test-Path -LiteralPath $destinationRootPath)) {
  Invoke-LoggedAction "Create directory: $destinationRootPath" {
    New-Item -ItemType Directory -Path $destinationRootPath -Force | Out-Null
  }
}

Assert-PathInside -Child $destinationRootPath -Parent (Join-Path $repoRoot "data-engine")

Write-Host "Source core: $sourceRootPath"
Write-Host "Destination: $destinationRootPath"

$sharedUtils = Join-Path $sourceRootPath "utils.py"
if (Test-Path -LiteralPath $sharedUtils -PathType Leaf) {
  Copy-FileIfChanged -SourceFile $sharedUtils -DestinationFile (Join-Path $destinationRootPath "utils.py")
}

foreach ($mapping in $ComponentMappings) {
  $sourceDirectory = Join-Path $sourceRootPath $mapping.Source
  $destinationDirectory = Join-Path $destinationRootPath $mapping.Destination
  Sync-Directory -SourceDirectory $sourceDirectory -DestinationDirectory $destinationDirectory -DestinationRoot $destinationRootPath
}

if (-not $KeepLegacyFolders) {
  foreach ($legacyFolder in $LegacyFolders) {
    $legacyPath = Join-Path $destinationRootPath $legacyFolder
    if (Test-Path -LiteralPath $legacyPath) {
      Assert-PathInside -Child $legacyPath -Parent $destinationRootPath
      Invoke-LoggedAction "Remove legacy folder: $legacyPath" {
        Remove-Item -LiteralPath $legacyPath -Recurse -Force
      }
    }
  }
}

Write-Host "Data-engine source sync complete."
