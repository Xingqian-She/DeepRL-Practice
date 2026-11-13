# Stage2：固定 XY Fourier 输入实验

设计日期：2026-11-12（北京时间）。这是实验设计与验收说明，尚无正式收敛结果。

## 固定编码与研究问题

检验位置的确定性多尺度特征能否改善原 Stage2 的部分观测控制表现。原始坐标并未丢失精度；
本实验改变的是网络对小位移的响应和学习条件，不增加可观测信息。不能由位置历史必然推得唯一倾角，
也尚未证明 Transformer 是原结果的失败原因。

原输入为 `masked_readings_8, availability_8, previous_action_one_hot_4`，完整保留这 20 维。
在其后依次追加 x 的四档 sin/cos、y 的两档 sin/cos，每档先 sin 后 cos：

```text
x: π, 2π, 4π, 8π       8 维
y: π, 2π               4 维
最终输入：20 + 8 + 4 = 32 维
```

频率是每单位**环境归一化坐标**的角频率，单位为 rad/unit；不是时间 Hz。
直接使用原坐标，不裁剪、不重新归一化、不缩放特征幅值。编码无可学习参数，也不消费随机数。
两个独立输入投影各增加 `12×256` 个权重，总参数从 6,438,677 增至 **6,444,821**（约增加 0.095%）。
单个坐标微扰的局部特征变化范数分别为 `π√85≈28.96` 和 `π√5≈7.02` 倍坐标微扰；
这不等于最终网络梯度或性能获得相同倍数提升。

## 频段依据与边界

Gymnasium 1.2.2、box2d-py 2.3.5、无风、离散动作，每步物理时间 1/50 秒。
环境 x/y 位置尺度分别为 10 和 20/3 米，速度读数的缩放也不同；
忽略碰撞、转动及约束修正时，约有 `Δx≈vx/100`、`Δy≈0.0225×vy`。
不能直接把 25 Hz 用作空间编码的上限，也不能将 observation_space 声明范围当作严格物理速度界。

使用种子 **30000–30255**，每种策略 256 回合，每回合最多 1000 步；不使用周期种子 10000 或留出种子 20000。
只记录真实相邻位置，不包含 reset 跳变，不按回报挑选频率。

| 校准策略 | 转移数 | 最大 \|Δx\| | 最大 \|Δy\| |
|---|---:|---:|---:|
| 随机动作 | 24,910 | 0.020108 | 0.046066 |
| Gymnasium 启发式 | 66,664 | 0.015776 | 0.025153 |
| 持续主引擎 | 32,368 | 0.050648 | 0.196751 |
| 合计 | **123,942** | **0.050648** | **0.196751** |

设计准则为最高档单步相位 `ω_max |Δu| ≤ π/2`。在基频固定为 π、逐档倍频时：

| 轴 | 选定最高档 | 校准最大相位 / π | 满足 π/2 的位移上界 | 达到 π 的位移 |
|---|---:|---:|---:|---:|
| x | 8π | 0.40519 | 0.0625 | 0.125 |
| y | 2π | 0.39350 | 0.25 | 0.5 |

再提高一档，压力校准分别为 `0.81037π`、`0.78700π`，不再满足余量标准。
因此 x 四档、y 两档是**在这套基频、倍频及经验余量约束下的最大档数**，并非理论或性能最优。
这里的相位准则是经验性的局部绕回监测，不是对非平稳轨迹满足带限采样定理的证明。
保留原坐标使周期特征重复不会导致完整输入的位置歧义，但不能保证网络必然学会使用原坐标消歧。
新策略可能进入校准未覆盖的状态，因此监测仅记录越界，不中止训练或调整编码。

校准复现脚本：[calibrate_xy_fourier.py](scripts/calibrate_xy_fourier.py)。
完整数值：[calibration.json](xy-fourier-design/calibration.json)。

参考：[环境固定版本源码](https://github.com/Farama-Foundation/Gymnasium/blob/v1.2.2/gymnasium/envs/box2d/lunar_lander.py)、
[NeRF](https://www.matthewtancik.com/nerf)、[Fourier Features](https://bmild.github.io/fourfeat/)。
MLP 上的已有结果不构成本任务及 Transformer 上的收益保证。

## 运行协议

配置从 `course-10m-v1/frozen-recipe.json` 的 FO04 参数和课程复制，仅启用 `xy_fourier`。
`decoder_layers=4` 是现有配置新增的默认字段，在 separate 模式下不参与网络。
Actor/Critic 独立骨干、D256、4 层、4 heads、每层窗口 32，上一动作保留，残差门初值 1e-4。
PPO 为 8 环境×256 步、序列 64、每 minibatch 4 序列、最多 4 epochs、学习率 3e-4 线性降至零。
其余参数见 [xy_fourier_experiments.json](../xy_fourier_experiments.json)。

- `XY_PO01`：直接 p=1，从头训练。
- `XY_PO03`：成绩课程；均分 ≥200 且 ≥200 比例 ≥80%，连续两次通过才推进 0.1，失败重置计数，不回退。
- 每组种子 42、43、44，共 6 次训练；每次请求 10,000,000 步，实际 9,998,336 步、4,882 次更新。
- 周期 sample 评估：每 100 次更新及最后更新，20 回合、8 环境、种子 10000–10019。
- 最终 latest.pt：p=0/0.5/1 × sample/argmax，每项 100 回合、8 环境、种子 20000–20099，共 36 个评估任务。
- FP32、math SDPA、TF32/AMP 关闭，compile auto；沿用原奖励、无风、最长回合 1000 步。

在仓库根目录 PowerShell 启动：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_xy_fourier.ps1
```

正式批次 `xy-fourier-10m-v1`；`-DryRun` 查看 42 个任务，`-Status` 查看状态，`-AnalyzeOnly` 重算汇总。
`-Smoke` 使用独立小模型和批次 `xy-fourier-smoke-v1`，不用于收敛结论。
`-Groups XY_PO01` 可只运行一组；同批次稍后补齐另一组。
Ctrl+C 停止，重跑同一命令会跳过校验通过的任务；被中断的训练以新 attempt 从头训练，
这不是恢复优化器和环境状态。源码或已有任务配置改变时必须使用新的 `-Batch` 名称。

新增入口复用通用调度器，不走旧 Stage2 的历史源码相等检查。历史文件不改写、不重跑基线。
`summary/xy-comparison.md` 和 JSON 按已登记 attempt 校验历史来源、配置和文件哈希，
分别保留两批次源码指纹，报告每个条件三个训练种子的均值和样本标准差；不将它们合并为同源严格对照。
主要关注 p=1/sample 的均分与 ≥200 比例，同时检查 argmax、其他缺失概率、课程推进及 p=1 预算。
留出种子已参与历史分析，本次为探索性实验。

## 相位日志与兼容

开启时，`training.csv` 每个 rollout、`evaluation.csv` 每次周期评估及独立 `evaluation.json` 记录：

- `xy_phase_transitions`：有效转移数；`xy_phase_skipped_terminal_transitions`：缺少末帧而跳过的终止转移数。
- `xy_{x,y}_phase_step_max_over_pi`：最高档实际相位增量的最大值，单位为 π。
- `xy_{x,y}_phase_ge_half_pi_count`、`xy_{x,y}_phase_ge_pi_count`：达到对应阈值的转移次数。

使用未取模的 `ω|Δu|`；最高档约束也覆盖全部较低档。SAME_STEP 终止使用 `final_obs`，
reset 不计位移；连续转移跨 rollout 计数，评估复用器每次调用重新统计。
训练时 buffer 和历史仍保存 16 维传感器观测及动作，不缓存展开特征、不加入真速度或倾角。
旧 checkpoint 缺失 `xy_fourier` 时默认关闭，维持原 20 维权重名；新 checkpoint 的 input_spec
记录固定频率、排列、维数及编码版本，载入时严格校验。开关为训练结构参数，评估从 checkpoint 自动恢复。
