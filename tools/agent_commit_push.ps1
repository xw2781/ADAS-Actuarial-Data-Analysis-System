<#
.SYNOPSIS
Commit and push ArcRho frontend changes with an explicit agent-written message.

.DESCRIPTION
This script is intended for coding agents. It keeps the commit/push workflow
repeatable while still requiring the agent to write a fresh commit message for
the current conversation.

Default behavior:
- Finds the Git repository root.
- Verifies the remote points at xw2781/ArcRho.
- Stages all changes with git add -A.
- Commits with the supplied message.
- Pushes the current branch to origin.

Examples:
  .\tools\agent_commit_push.ps1 -Message "Reorganize frontend UI entrypoints"
  .\tools\agent_commit_push.ps1 -Message "Update docs index" -NoPush
  .\tools\agent_commit_push.ps1 -Message "Review commit scope" -DryRun
#>

[CmdletBinding(PositionalBinding = $false)]
param(
  [Parameter(Mandatory = $true)]
  [ValidateNotNullOrEmpty()]
  [string]$Message,

  [string]$Remote = "origin",

  [string]$Branch = "",

  [ValidateSet("all", "tracked", "none")]
  [string]$StageMode = "all",

  [switch]$NoPush,

  [switch]$DryRun,

  [switch]$AllowEmpty,

  [switch]$SkipRemoteCheck,

  [switch]$SetOriginIfMissing,

  [string]$ExpectedRemoteSlug = "xw2781/ArcRho",

  [string[]]$Pathspec = @(".")
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Pathspec = @(
  $Pathspec |
    ForEach-Object { $_ -split "," } |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ }
)
if ($Pathspec.Count -eq 0) {
  $Pathspec = @(".")
}

function Invoke-Git {
  param(
    [Parameter(Mandatory = $true)]
    [string[]]$GitArgs,

    [switch]$AllowFailure,

    [switch]$Capture
  )

  if ($Capture) {
    $output = & git @GitArgs 2>&1
    $code = $LASTEXITCODE
    if ($code -ne 0 -and -not $AllowFailure) {
      throw "git $($GitArgs -join ' ') failed with exit code $code.`n$output"
    }
    return [pscustomobject]@{
      Code = $code
      Output = @($output)
    }
  }

  & git @GitArgs
  $code = $LASTEXITCODE
  if ($code -ne 0 -and -not $AllowFailure) {
    throw "git $($GitArgs -join ' ') failed with exit code $code."
  }
  return [pscustomobject]@{
    Code = $code
    Output = @()
  }
}

function Normalize-RemoteUrl {
  param([string]$Url)

  return ($Url.Trim() `
    -replace "^git@github\.com:", "https://github.com/" `
    -replace "\.git$", "" `
    -replace "/$", "")
}

function Assert-GitAvailable {
  $cmd = Get-Command git -ErrorAction SilentlyContinue
  if (-not $cmd) {
    throw "git was not found on PATH."
  }
}

Assert-GitAvailable

$repoProbe = Invoke-Git -GitArgs @("rev-parse", "--show-toplevel") -Capture
$repoRoot = ($repoProbe.Output | Select-Object -First 1).ToString().Trim()
if (-not $repoRoot) {
  throw "Could not determine Git repository root."
}

Set-Location -LiteralPath $repoRoot

$gitDir = (Invoke-Git -GitArgs @("rev-parse", "--git-dir") -Capture).Output[0].ToString().Trim()
if (Test-Path -LiteralPath (Join-Path $gitDir "MERGE_HEAD")) {
  throw "A merge is in progress. Resolve it before committing."
}
if (
  (Test-Path -LiteralPath (Join-Path $gitDir "rebase-merge")) -or
  (Test-Path -LiteralPath (Join-Path $gitDir "rebase-apply"))
) {
  throw "A rebase is in progress. Resolve it before committing."
}

$conflicts = (Invoke-Git -GitArgs @("diff", "--name-only", "--diff-filter=U") -Capture).Output
if ($conflicts.Count -gt 0) {
  throw "Unresolved merge conflicts are present:`n$($conflicts -join "`n")"
}

$currentBranch = ((Invoke-Git -GitArgs @("branch", "--show-current") -Capture).Output | Select-Object -First 1).ToString().Trim()
if (-not $currentBranch) {
  throw "Detached HEAD state detected. Check out a branch before committing."
}
if ($Branch -and $Branch -ne $currentBranch) {
  throw "Current branch is '$currentBranch', but -Branch requested '$Branch'."
}

$remoteProbe = Invoke-Git -GitArgs @("remote", "get-url", $Remote) -Capture -AllowFailure
if ($remoteProbe.Code -ne 0) {
  if ($SetOriginIfMissing -and $Remote -eq "origin") {
    $originUrl = "https://github.com/$ExpectedRemoteSlug.git"
    if ($DryRun) {
      Write-Host "[dry-run] Would add remote origin $originUrl"
    } else {
      Invoke-Git -GitArgs @("remote", "add", "origin", $originUrl) | Out-Null
    }
    $remoteUrl = $originUrl
  } else {
    throw "Remote '$Remote' is not configured. Add it or rerun with -SetOriginIfMissing."
  }
} else {
  $remoteUrl = ($remoteProbe.Output | Select-Object -First 1).ToString().Trim()
}

if (-not $SkipRemoteCheck) {
  $normalizedRemote = Normalize-RemoteUrl -Url $remoteUrl
  $expected = "https://github.com/$ExpectedRemoteSlug"
  if ($normalizedRemote -ne $expected) {
    throw "Remote '$Remote' points to '$remoteUrl', expected '$expected' or git@github.com:$ExpectedRemoteSlug.git. Use -SkipRemoteCheck only if this is intentional."
  }
}

Write-Host "Repository: $repoRoot"
Write-Host "Branch:     $currentBranch"
Write-Host "Remote:     $Remote ($remoteUrl)"
Write-Host ""
Write-Host "Pre-stage status:"
Invoke-Git -GitArgs @("status", "--short") | Out-Null
Write-Host ""

switch ($StageMode) {
  "all" {
    $args = @("add", "-A", "--") + $Pathspec
    if ($DryRun) {
      Write-Host "[dry-run] Would run: git $($args -join ' ')"
    } else {
      Invoke-Git -GitArgs $args | Out-Null
    }
  }
  "tracked" {
    $args = @("add", "-u", "--") + $Pathspec
    if ($DryRun) {
      Write-Host "[dry-run] Would run: git $($args -join ' ')"
    } else {
      Invoke-Git -GitArgs $args | Out-Null
    }
  }
  "none" {
    Write-Host "StageMode=none: using currently staged changes only."
  }
}

if ($DryRun -and $StageMode -ne "none") {
  $statusArgs = @("status", "--porcelain", "--") + $Pathspec
  $candidateChanges = (Invoke-Git -GitArgs $statusArgs -Capture).Output
  if ($candidateChanges.Count -eq 0 -and -not $AllowEmpty) {
    throw "No changes to commit. Use -AllowEmpty only when an empty commit is intentional."
  }

  Write-Host ""
  Write-Host "Candidate status:"
  $candidateChanges | ForEach-Object { Write-Host $_ }
  Write-Host ""
  Write-Host "Candidate tracked diff summary:"
  $diffStatArgs = @("diff", "--stat", "--") + $Pathspec
  Invoke-Git -GitArgs $diffStatArgs | Out-Null
  Write-Host ""
  Write-Host "Commit message:"
  Write-Host $Message
  Write-Host ""
  Write-Host "[dry-run] No staging, commit, or push was performed."
  exit 0
}

$diffCheck = Invoke-Git -GitArgs @("diff", "--cached", "--quiet") -AllowFailure
$hasStagedChanges = ($diffCheck.Code -eq 1)
if (-not $hasStagedChanges -and -not $AllowEmpty) {
  throw "No staged changes to commit. Use -AllowEmpty only when an empty commit is intentional."
}

Write-Host ""
Write-Host "Staged summary:"
Invoke-Git -GitArgs @("diff", "--cached", "--stat") | Out-Null
Write-Host ""
Write-Host "Commit message:"
Write-Host $Message
Write-Host ""

if ($DryRun) {
  Write-Host "[dry-run] No commit or push was performed."
  exit 0
}

if ($AllowEmpty -and -not $hasStagedChanges) {
  Invoke-Git -GitArgs @("commit", "--allow-empty", "-m", $Message) | Out-Null
} else {
  Invoke-Git -GitArgs @("commit", "-m", $Message) | Out-Null
}

Write-Host ""
Write-Host "Created commit:"
Invoke-Git -GitArgs @("log", "-1", "--oneline") | Out-Null

if ($NoPush) {
  Write-Host ""
  Write-Host "NoPush set: commit was not pushed."
  exit 0
}

Write-Host ""
Write-Host "Pushing $currentBranch to $Remote..."
Invoke-Git -GitArgs @("push", "-u", $Remote, $currentBranch) | Out-Null

Write-Host ""
Write-Host "Post-push status:"
Invoke-Git -GitArgs @("status", "--short", "--branch") | Out-Null
