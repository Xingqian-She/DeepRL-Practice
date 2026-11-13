#Requires -Version 5.1
[CmdletBinding()]
param(
    [string[]]$Groups,
    [string]$Batch,
    [string]$Config,
    [switch]$Status,
    [switch]$DryRun,
    [switch]$AnalyzeOnly,
    [switch]$Smoke,
    [ValidateSet('cpu', 'cuda')][string]$Device,
    [ValidateSet('Stage1', 'Stage2')][string]$Stage = 'Stage1',
    [string]$BaselineBatch = 'baseline-v1'
)

$ErrorActionPreference = 'Stop'
$originalConsoleEncoding = [Console]::OutputEncoding
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$repoRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
if (-not $Config) { $Config = Join-Path $PSScriptRoot 'experiments.json' }
$configPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Config)
$readOnly = $Status -or $DryRun
$inspectOnly = $readOnly -or $AnalyzeOnly
$modeCount = [int]$Status.IsPresent + [int]$DryRun.IsPresent + [int]$AnalyzeOnly.IsPresent
if ($modeCount -gt 1) { throw 'Status、DryRun 和 AnalyzeOnly 不能同时使用。' }
if (-not $Batch) {
    if ($Stage -eq 'Stage2') {
        if ($Smoke) { $Batch = 'course-smoke-v1' } else { $Batch = 'course-10m-v1' }
    } elseif ($Smoke) { $Batch = 'smoke-v1' } else { $Batch = 'baseline-v1' }
}
if ($Stage -eq 'Stage2' -and $PSBoundParameters.ContainsKey('Config')) { throw 'Stage2 uses the frozen baseline configuration; Config is not supported.' }
if ($Stage -eq 'Stage2' -and $BaselineBatch -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$') { throw 'Invalid BaselineBatch identifier.' }
if ($Batch -notmatch '^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$') { throw 'Batch 请使用英文字母、数字、下划线和连字符，长度不超过 80。' }
if (-not $configPath.StartsWith($repoRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Config 必须位于当前仓库内，以便挂载到容器。'
}

function Invoke-DockerCapture {
    param([string[]]$Arguments)
    # Windows PowerShell turns native stderr into ErrorRecords. Keep the exit code authoritative.
    $savedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $lines = @(& $script:dockerExecutable @Arguments 2>&1 | ForEach-Object { "$_" })
        $code = $LASTEXITCODE
        return [PSCustomObject]@{ Code = $code; Lines = $lines }
    } finally { $ErrorActionPreference = $savedPreference }
}

function Assert-DockerSuccess {
    param([string[]]$Arguments)
    $result = Invoke-DockerCapture -Arguments $Arguments
    foreach ($line in $result.Lines) { Write-Host $line }
    if ($result.Code -ne 0) { throw "Docker 命令失败（退出码 $($result.Code)）。" }
}

function Receive-ExperimentOutput {
    param($Job)
    foreach ($item in @(Receive-Job -Job $Job)) {
        if ($null -ne $item -and $null -ne $item.PSObject.Properties['ExperimentExitCode']) {
            $script:experimentExitCode = [int]$item.ExperimentExitCode
        } else { Write-Host "$item" }
    }
}

$job = $null
$sessionDirectory = $null
$containerName = $null
$sessionId = [Guid]::NewGuid().ToString('N')
$experimentExitCode = 1
try {
    $dockerExecutable = (Get-Command docker -ErrorAction Stop).Source
    $version = Invoke-DockerCapture -Arguments @('version', '--format', '{{.Server.Version}}')
    if ($version.Code -ne 0) {
        $desktopPath = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
        if (-not (Test-Path -LiteralPath $desktopPath)) { throw '请先安装 Docker Desktop，并启用 WSL2 和 Linux 容器。' }
        Write-Host '正在启动 Docker Desktop，等待 Linux 引擎就绪……'
        Start-Process -FilePath $desktopPath -WindowStyle Hidden
        $deadline = [DateTime]::UtcNow.AddSeconds(120)
        do {
            Start-Sleep -Seconds 2
            $version = Invoke-DockerCapture -Arguments @('version', '--format', '{{.Server.Version}}')
        } until ($version.Code -eq 0 -or [DateTime]::UtcNow -ge $deadline)
        if ($version.Code -ne 0) { throw 'Docker 引擎未就绪，请检查 Docker Desktop 的 Linux/WSL2 设置后重新执行。' }
    }
    $composeVersion = Invoke-DockerCapture -Arguments @('compose', 'version')
    if ($composeVersion.Code -ne 0) { throw '当前 Docker 缺少 Compose，请更新 Docker Desktop。' }
    $compose = @('compose', '--project-directory', $repoRoot, '-f', (Join-Path $repoRoot 'compose.yaml'))
    $needsGpu = -not $inspectOnly -and ((-not $Smoke -and $Device -ne 'cpu') -or $Device -eq 'cuda')
    if ($needsGpu) { $compose += @('-f', (Join-Path $repoRoot 'compose.gpu.yaml')) }
    $image = 'deeprl-practice:py311-cu128'
    $imageInfo = Invoke-DockerCapture -Arguments @('image', 'inspect', $image, '--format', '{{.Id}}')
    $rebuild = $imageInfo.Code -ne 0
    if (-not $rebuild) {
        $probeCode = "import hashlib; from pathlib import Path; print(hashlib.sha256(Path('/opt/deeprl/requirements.lock').read_bytes().replace(b'\r\n', b'\n')).hexdigest())"
        $probe = Invoke-DockerCapture -Arguments @('run', '--rm', '--entrypoint', 'python', $image, '-B', '-c', $probeCode)
        $lockHasher = [Security.Cryptography.SHA256]::Create()
        $lockContent = [Text.Encoding]::UTF8.GetString([IO.File]::ReadAllBytes((Join-Path $repoRoot 'requirements.lock'))).Replace("`r`n", "`n")
        try {
            $expectedHash = ([BitConverter]::ToString($lockHasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($lockContent)))).Replace('-', '').ToLowerInvariant()
        } finally { $lockHasher.Dispose() }
        $rebuild = $probe.Code -ne 0 -or $probe.Lines[-1] -ne $expectedHash
    }
    if ($rebuild) {
        if ($inspectOnly) { throw '项目镜像缺失或依赖锁不匹配。请先不带 Status/DryRun/AnalyzeOnly 执行入口，以准备环境。' }
        Write-Host '正在构建项目镜像（首次运行可能需要下载依赖）……'
        Assert-DockerSuccess -Arguments ($compose + @('build', 'runner'))
    }

    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        $identityBytes = [Text.Encoding]::UTF8.GetBytes($repoRoot.ToLowerInvariant() + '|' + $Batch)
        $identity = ([BitConverter]::ToString($sha.ComputeHash($identityBytes))).Replace('-', '').ToLowerInvariant().Substring(0, 16)
    } finally { $sha.Dispose() }
    $containerName = 'deeprl-lunarlander-' + $identity
    if ($inspectOnly) { $containerName += '-view-' + $sessionId.Substring(0, 8) }
    $existing = Invoke-DockerCapture -Arguments @('inspect', $containerName, '--format', '{{.State.Running}} {{json .Config.Labels}}')
    if ($existing.Code -eq 0) {
        $existingParts = $existing.Lines[-1].Split([char[]]@(' '), 2)
        $existingLabels = $existingParts[1] | ConvertFrom-Json
        if ($existingParts[0] -ne 'false' -or $existingLabels.'deeprl.lunarlander.owner' -ne $identity) {
            throw '该批次的实验容器已运行，或容器名称被其他任务占用。请等待原任务结束；可使用 -Status 查看记录。'
        }
        Assert-DockerSuccess -Arguments @('rm', $containerName)
    }

    $containerConfig = '/workspace/' + $configPath.Substring($repoRoot.Length + 1).Replace('\', '/')
    if ($Stage -eq 'Stage2') {
        $pythonArguments = @('python', '-B', '-m', 'stage2', '--baseline-batch', $BaselineBatch, '--batch', $Batch)
    } else {
        $pythonArguments = @('python', '-B', 'run_experiments.py', '--config', $containerConfig, '--batch', $Batch)
    }
    if ($Groups) {
        $pythonArguments += '--groups'
        foreach ($groupValue in $Groups) { $pythonArguments += $groupValue.Split(',') }
    }
    if ($Status) { $pythonArguments += '--status' }
    if ($DryRun) { $pythonArguments += '--dry-run' }
    if ($AnalyzeOnly) { $pythonArguments += '--analyze-only' }
    if ($Smoke) { $pythonArguments += '--smoke' }
    if ($Device) { $pythonArguments += @('--device', $Device) }
    if (-not $inspectOnly) {
        $sessionDirectory = Join-Path $PSScriptRoot ('runs\experiments\.sessions\' + $sessionId)
        [IO.Directory]::CreateDirectory($sessionDirectory) | Out-Null
        $heartbeatPath = Join-Path $sessionDirectory 'heartbeat'
        [IO.File]::WriteAllText($heartbeatPath, [Guid]::NewGuid().ToString('N'))
        $containerHeartbeat = '/workspace/lunarlander-ppo-pomdp/runs/experiments/.sessions/' + $sessionId + '/heartbeat'
        $pythonArguments += @('--heartbeat-file', $containerHeartbeat)
    }
    $runArguments = $compose + @('run', '--rm', '--no-deps', '-T', '--name', $containerName,
        '--label', ('deeprl.lunarlander.owner=' + $identity), '--label', ('deeprl.lunarlander.session=' + $sessionId),
        '-e', 'PYTHONPATH=/workspace/shared/src', '-e', 'TORCHINDUCTOR_COMPILE_THREADS=2',
        '-w', '/workspace/lunarlander-ppo-pomdp', 'runner') + $pythonArguments
    if ($inspectOnly) { Write-Host "实验批次：$Batch。仅查看或分析，不执行训练和评估任务。" }
    else { Write-Host "实验批次：$Batch。前台运行；Ctrl+C 停止，再次执行相同命令可检查并补跑。" }
    $job = Start-Job -ScriptBlock {
        param($Executable, $DockerArguments)
        [Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
        $OutputEncoding = [Console]::OutputEncoding
        & $Executable @DockerArguments 2>&1 | ForEach-Object { "$_" }
        [PSCustomObject]@{ ExperimentExitCode = $LASTEXITCODE }
    } -ArgumentList $dockerExecutable, (,$runArguments)
    $lastHeartbeat = [DateTime]::UtcNow
    while ($job.State -in @('NotStarted', 'Running')) {
        Receive-ExperimentOutput -Job $job
        if ($sessionDirectory -and ([DateTime]::UtcNow - $lastHeartbeat).TotalSeconds -ge 5) {
            [IO.File]::WriteAllText($heartbeatPath, [Guid]::NewGuid().ToString('N'))
            $lastHeartbeat = [DateTime]::UtcNow
        }
        Start-Sleep -Milliseconds 200
    }
    Receive-ExperimentOutput -Job $job
    if ($job.State -eq 'Failed') { throw "Docker 客户端任务失败：$($job.ChildJobs[0].JobStateInfo.Reason)" }
    if (-not $readOnly) {
        Write-Host ('报告位置：' + (Join-Path $PSScriptRoot "runs\experiments\$Batch\summary\overview.md"))
    }
} catch {
    Write-Host ("启动失败：" + $_.Exception.Message) -ForegroundColor Red
    $experimentExitCode = 1
} finally {
    if ($containerName -and $dockerExecutable) {
        $owned = Invoke-DockerCapture -Arguments @('inspect', $containerName, '--format', '{{json .Config.Labels}}')
        $ownedLabels = $null
        if ($owned.Code -eq 0) { $ownedLabels = $owned.Lines[-1] | ConvertFrom-Json }
        if ($ownedLabels -and $ownedLabels.'deeprl.lunarlander.session' -eq $sessionId) {
            # SIGTERM reaches the coordinator, which interrupts and reaps its active child first.
            $stopped = Invoke-DockerCapture -Arguments @('stop', '--time', '20', $containerName)
        }
    }
    if ($job) {
        if ($job.State -in @('NotStarted', 'Running')) { Stop-Job -Job $job }
        Remove-Job -Job $job -Force
    }
    if ($sessionDirectory) {
        if (Test-Path -LiteralPath $heartbeatPath) { Remove-Item -LiteralPath $heartbeatPath -Force }
        Remove-Item -LiteralPath $sessionDirectory -Force
    }
    [Console]::OutputEncoding = $originalConsoleEncoding
}
exit $experimentExitCode
