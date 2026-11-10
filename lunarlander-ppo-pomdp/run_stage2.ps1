#Requires -Version 5.1
[CmdletBinding()]
param(
    [string[]]$Groups,
    [string]$Batch,
    [string]$BaselineBatch = 'baseline-v1',
    [switch]$Status,
    [switch]$DryRun,
    [switch]$AnalyzeOnly,
    [switch]$Smoke,
    [ValidateSet('cpu', 'cuda')][string]$Device
)

$arguments = @{} + $PSBoundParameters
$arguments['Stage'] = 'Stage2'
& (Join-Path $PSScriptRoot 'run_experiments.ps1') @arguments
exit $LASTEXITCODE
