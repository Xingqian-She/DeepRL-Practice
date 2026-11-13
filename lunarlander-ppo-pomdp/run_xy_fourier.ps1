#Requires -Version 5.1
[CmdletBinding()]
param(
    [string[]]$Groups,
    [string]$Batch,
    [switch]$Status,
    [switch]$DryRun,
    [switch]$AnalyzeOnly,
    [switch]$Smoke,
    [ValidateSet('cpu', 'cuda')][string]$Device
)

$arguments = @{} + $PSBoundParameters
$arguments['Config'] = Join-Path $PSScriptRoot 'xy_fourier_experiments.json'
if (-not $Batch) {
    $arguments['Batch'] = if ($Smoke) { 'xy-fourier-smoke-v1' } else { 'xy-fourier-10m-v1' }
}
& (Join-Path $PSScriptRoot 'run_experiments.ps1') @arguments
exit $LASTEXITCODE
