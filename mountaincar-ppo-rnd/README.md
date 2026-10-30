# MountainCar PPO + RND

使用双价值头 PPO：外在价值评估环境奖励，内在价值评估 RND 好奇心奖励。
已增加固定种子、状态范围、优势尺度和独立评估诊断。
2026-10-30 的 Docker 对照发现：原配置仅降低 Predictor 学习率仍会失败；
保留残差网络，加入策略输入缩放与外在奖励尺度调整后，多次训练出现持续成功。
具体预算、种子和评估结果见 [诊断实验报告](experiments/diagnosis/REPORT.md)。
这些结果只对应已测试配置，不代表任意随机种子都保证收敛。

## 复现已经验证的配置

在仓库根目录用 PowerShell 执行；输出放到独立目录：

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm runner python mountaincar-ppo-rnd/train.py --normalize-policy-input --ext-reward-scale 0.01 --rnd-lr 4e-5 --no-compile --seed 0 --run-dir mountaincar-ppo-rnd/runs/balanced-seed0
```

保留原来的残差网络和 Target/Predictor 结构。两个尺度调整分别是：

```text
PPO 输入：position' = (position + 0.3) / 0.9，velocity' = velocity / 0.07
训练外在奖励：-1 → -0.01；回合统计及独立评估仍记录原始奖励 -1
```

输入缩放使用 MountainCar 观测空间的固定边界，保存在模型 buffer 中。
它与 RND 自己的运行均值/方差独立。外在奖励缩放会同时改变外在 critic
目标的单位和两路优势的相对影响，因此不是对整个混合奖励统一换单位。

`--no-compile` 关闭编译，不改变算法；GPU 编译路径另外经过短程闭环验证。
可用 `--device cpu --no-compile` 复现 CPU 对照；浮点数和设备差异会影响轨迹。
裸命令保留未缩放输入、原始外在奖励的配置，方便复现失败基线。
`--architecture mlp` 提供 64×64 Tanh 网络作为结构对照，不是推荐配置的必要部分。
`--seed` 默认 0；`--total-timesteps` 默认 500000，按完整 rollout 运行 499712 步。

训练后，对 checkpoint 进行 100 回合独立评估：

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm runner python mountaincar-ppo-rnd/evaluate_checkpoint.py mountaincar-ppo-rnd/runs/balanced-seed0/checkpoints/ppo_rnd_model.pth --episodes 100 --seed 20000
```

评估分别输出 argmax 策略与采样策略的成功率、原始平均回报，并保存 JSON。
训练中的定期评估使用另一组种子 10000；独立评估不消耗训练用的 PyTorch RNG。
新 checkpoint 包含 `model_config`，评估脚本也兼容原始 checkpoint。

每次训练的 `config.json` 保存运行参数；`evaluations.jsonl` 保存定期评估。
同一个 `--run-dir` 再运行会覆盖日志及模型，请用不同目录比较实验。

新增 CSV/曲线指标：

| 指标 | 用途 |
| --- | --- |
| `position_min/max`、`speed_abs_max` | 本轮到达的状态范围；不是累计覆盖率 |
| `adv_ext_weighted_std`、`adv_int_weighted_std` | 两路优势乘以各自系数后、共同标准化前的标准差 |
| `adv_int_ext_std_ratio` | 两路优势的尺度比；不能直接解释成策略梯度贡献比例 |
| `adv_ext/int_mean`、`value_ext_mean` | 辅助识别 critic 偏差及近似恒定价值 |
| `action_*_fraction` | 本轮三种动作的比例 |
| `agent_grad_norm` | 裁剪前的 Agent 总梯度范数 |

损失现为整轮所有 minibatch 的均值，旧日志的损失是最后一个 minibatch 的值，
两者不宜逐点对比。RND 奖励和优势尺度曲线使用对数形式，避免早期尖峰掩盖后期变化。

## 先从哪里读代码

训练入口仍然是 `python mountaincar-ppo-rnd/train.py`；对照参数可通过命令行设置。
重构按职责拆分，不需要先理解所有文件。

| 文件 | 负责什么 | 建议先看 |
| --- | --- | --- |
| `train.py` | 创建组件，按顺序组织训练 | `_train()` 中的四个阶段 |
| `rollout.py` | 环境交互、观测预热、还原真实后继状态 | `collect_rollout()` |
| `rnd.py` | 状态归一化、内在奖励、奖励尺度统计 | `compute_intrinsic_reward()`、`scale_intrinsic_rewards()` |
| `buffer.py` | 存储 rollout，分别计算两路 GAE 和价值目标 | 两个 `compute_*_returns_and_advantages()` |
| `ppo_rnd.py` | 合并优势、更新 PPO 和 RND | `update()` → `_update_agent()` / `_update_predictor()` |
| `model.py` | Actor、双 Critic、RND Target/Predictor 的网络定义 | 两个模型的 `forward()` |
| `reporting.py` | 回合统计、终端进度、CSV 和曲线 | 需要看日志时再读 |
| `checkpoint.py` | 保存模型和配套归一化统计量 | 需要使用模型文件时再读 |
| `diagnostics.py` | 记录采样范围、优势尺度，独立评估 | `rollout_metrics()`、`evaluate()` |

建议沿着一次更新的数据流阅读：

```text
collect_rollout()             采样：保存 s_t、真实 s_{t+1}、两路奖励和价值
        ↓
scale_intrinsic_rewards()    缩放内在奖励，保留外在奖励
        ↓
buffer.compute_*()           两路 GAE：奖励 → 优势与价值目标
        ↓
trainer.update()             混合优势 → PPO；后继状态 → RND Predictor
        ↓
obs_rms.update()             本轮训练结束，更新下一轮的观测统计量
```

`collect_rollout()` 返回下一次采样的状态，其余训练数据写入 `buffer`。
`scale_intrinsic_rewards()` 原地缩放 `buffer.rewards_int`，返回值仅用于日志。
`trainer.update()` 更新网络并重置 buffer 写指针，不清除观测数据。
统计量的更新顺序由 `train.py` 明确控制，采样和更新 Predictor 期间都不改 `obs_rms`。

网络更新的公式集中在两个方法中：

- `_update_agent()`：PPO 裁剪策略损失 + 两个价值损失 − 熵奖励。
- `_update_predictor()`：Predictor 与固定 Target 的特征均方误差。

为了兼容已有调用，`ppo_rnd.py` 仍导出 `RolloutBuffer`、`RunningMeanStd`、
`RewardForwardFilter`、`normalize_rnd_obs`；它们的实现位于各自职责文件。
原有 checkpoint 字段、输出目录及训练命令保持兼容。

## 1. 一条 transition 中，两种模型看什么

环境交互为 `(s_t, a_t, r_t, s_{t+1})`：

| 用途 | 使用的观测 |
| --- | --- |
| PPO 策略、当前价值 | `buffer.obs`，即 `s_t` |
| RND 奖励、predictor 更新 | `buffer.rnd_obs`，即真实 `s_{t+1}` |
| 外在价值 bootstrap | 真实 `s_{t+1}`；真正终止时为 0 |
| 内在价值 bootstrap | 下一次采取动作的状态；回合结束时是 reset 状态 |

`SAME_STEP` 自动重置会将返回的观测替换为 reset 状态，因此要从
`infos["final_obs"]` 还原 RND 使用的真实后继状态。
RND 的训练目标始终是 target 特征；不会用缩放后的内在奖励训练 predictor。

一个 rollout 内固定 `obs_rms`，执行顺序是：

1. 用固定的观测统计量计算所有 RND 奖励，并保存真实后继状态。
2. 缩放内在奖励，计算两路 GAE 和价值目标。
3. 用同一批后继状态、同一组观测统计量更新 predictor，并更新 PPO。
4. 最后用本轮后继状态更新 `obs_rms`，供下一轮使用。

奖励计算和 predictor 更新共用 `normalize_rnd_obs()`，避免两处公式不一致。
`buffer.reset()` 仅重置写指针，数据在下一轮采样时才会被覆盖。

## 2. 为什么统计折扣累计奖励的方差

先计算原始 RND 奖励：

```text
r_int[t] = mean((predictor(normalize(s[t+1])) - target(normalize(s[t+1])))²)
```

对每个环境分别维护一个前向滤波器 `RewardForwardFilter`：

```text
F[t] = gamma_int * F[t-1] + r_int[t]
     = r_int[t] + gamma_int*r_int[t-1] + gamma_int²*r_int[t-2] + ...
```

这里积累的是过去奖励，越久远权重越小。它与 critic 要预测的未来回报不同：

```text
G[t] = r_int[t] + gamma_int*r_int[t+1] + gamma_int²*r_int[t+2] + ...
```

例如 `gamma_int=0.5`、奖励依次为 `[1, 2, 3]`，初始 `F=0`：

| 时间 | 当前奖励 | 前向滤波 F | 假设第三步之后无奖励的未来回报 G |
| --- | ---: | ---: | ---: |
| 1 | 1 | 1 | 2.75 |
| 2 | 2 | 2.5 | 3.5 |
| 3 | 3 | 4.25 | 3 |

所以，`F` 不是逐状态的价值预测，也不直接用作 critic 目标。
它提供一个容易在线计算的、考虑时间累计效应的奖励尺度。
在奖励平稳且各时刻不相关的理想情况下，过去与未来的无限折扣和具有相同方差；
实际训练的奖励会变化，因此这里是尺度估计，不能保证等于未来回报方差。

实现中先用所有环境的 `F` 样本更新 `int_rew_rms`，再缩放原始单步奖励：

```text
sigma = sqrt(running_variance(F))
scaled_r_int[t] = r_int[t] / (sigma + 1e-8)
```

注意三点：

- 分母来自累计量 `F`，分子仍是单步奖励 `r_int`；不能把 `F/sigma` 当作单步奖励。
- 不减均值。这里调整奖励尺度，保留原始好奇心奖励非负的性质。
- 每个环境有独立的 `F`，但共享运行方差；`F` 跨 rollout、跨回合保留。
  本实现把内在奖励视为持续任务，因此 reset 不清零；外在 GAE 仍按回合切断。

这与原始 RND 的奖励尺度处理一致。critic 的目标仍由缩放后的奖励和 bootstrap
通过 GAE 得到，不是 `F`。奖励尺度处理也不能保证好奇心权重自动衰减；
如果需要明确降低探索强度，应另行调度 `INT_COEF`。

## 3. 优势怎样合并

```text
A = EXT_COEF * A_ext + INT_COEF * A_int
A_policy = (A - mean(A)) / (std(A) + 1e-8)
```

整个 rollout 只计算一次总优势标准化，所有 minibatch 和 epoch 共用。
不单独标准化 `A_int` 和 `A_ext`，因而保留两路优势的相对尺度。
例如内在优势整体缩小 100 倍时，它相对于外在优势的影响也会下降。
价值目标 `returns_ext` 和 `returns_int` 不经过优势标准化。

原始 RND 代码直接加权合并两路优势，并没有再标准化总优势。
这里的总优势标准化是额外的 PPO 稳定化选择，因此本项目不是逐项复刻原始实现。
奖励尺度处理与优势标准化作用于不同阶段，前者还会影响 critic 的学习目标。

## 4. 如何判断训练是否有效

执行 `python mountaincar-ppo-rnd/train.py`，输出位于本目录的 `runs/`：

- `training.csv`：每次更新的训练指标、内在奖励尺度、最近 50 个已完成回合的表现。
- `episodes.csv`：每个完成回合的外在奖励总和、步数、成功和终止/截断标志。
- `training.png`：训练指标、回合表现及内在奖励尺度曲线。
- `checkpoints/ppo_rnd_model.pth`：模型权重以及配套的归一化统计量、奖励滤波状态。
  尚未提供完整断点续训功能，文件不包含优化器和环境状态。

`success_rate` 是最近 50 个完成回合中到达终点的比例；不是所有历史回合的成功率，
也不是独立评估集上的成功率。尚无完成回合时，表现指标记为 NaN。
MountainCar 每步奖励为 -1，200 步超时通常得到 -200；先看成功率是否上升，
再看成功回合能否更快结束。仅看 predictor loss 下降不能说明车学会上山。

## 5. 验证

在仓库根目录运行：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -B -m unittest discover -s mountaincar-ppo-rnd -p test_ppo_rnd.py -v
```

测试覆盖独立评估及随机数隔离、模型缩放恢复、多环境奖励滤波、优势相对尺度、异步回合统计、真实环境终止/截断、
奖励缩放、两路 GAE 边界和 1024 步训练闭环。训练测试会核对 RND 样本与统计量一致性、
target 冻结、predictor 更新，以及 CSV、图片和 checkpoint 输出。
测试临时绕过 `torch.compile` 并在 CPU 上执行，不代表编译后端或完整收敛验证。

参考：[OpenAI 原始 RND 实现](https://github.com/openai/random-network-distillation/blob/master/ppo_agent.py)。
