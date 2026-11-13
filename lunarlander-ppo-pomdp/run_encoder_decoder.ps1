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
$arguments['Config'] = Join-Path $PSScriptRoot 'encoder_decoder_experiments.json'
if (-not $Batch) {
    $arguments['Batch'] = if ($Smoke) { 'encoder-decoder-smoke-v1' } else { 'encoder-decoder-10m-v1' }
}
& (Join-Path $PSScriptRoot 'run_experiments.ps1') @arguments
exit $LASTEXITCODE
