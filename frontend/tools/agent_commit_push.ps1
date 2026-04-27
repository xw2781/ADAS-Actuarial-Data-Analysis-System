<#
.SYNOPSIS
Compatibility wrapper for the ArcRho root commit/push helper.
#>

$rootScript = Resolve-Path (Join-Path $PSScriptRoot "..\..\tools\agent_commit_push.ps1")
& $rootScript @args
exit $LASTEXITCODE
