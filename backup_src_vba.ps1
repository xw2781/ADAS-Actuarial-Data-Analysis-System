param(
    [string]$Message = "",
    [switch]$Push
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$targetPath = "excel/src_vba"

# Stage only the VBA folder (including deletions).
git add -- $targetPath

# If nothing is staged for this path, exit cleanly.
git diff --cached --quiet -- $targetPath
if ($LASTEXITCODE -eq 0) {
    Write-Host "No changes detected in $targetPath. Nothing to commit."
    exit 0
}

if ([string]::IsNullOrWhiteSpace($Message)) {
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Message = "backup excel src_vba $stamp"
}

git commit -m $Message
if ($LASTEXITCODE -ne 0) {
    throw "Commit failed."
}

if ($Push) {
    $branch = (git rev-parse --abbrev-ref HEAD).Trim()
    git push origin $branch
    if ($LASTEXITCODE -ne 0) {
        throw "Push failed."
    }
}

Write-Host "Backup completed for $targetPath."
