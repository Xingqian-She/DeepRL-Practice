# MountainCar DQN + SimHash 填空教程

目标：亲手实现「状态哈希 → 访问计数 → 内在奖励 → DQN 学习」的数据流。
环境为 `MountainCar-v0`，观测 `[position, velocity]`，动作 0/1/2。
本项目按填空教程组织；当前第 1–4 课已经完成，原来的 TODO 注释保留作学习提示。
四项自检已通过。seed0 的三组实验与限制见 [评估诊断报告](analysis/REPORT.md)。
已完成 [三个训练 seed 与 reset 配对对照](analysis/reset-generalization-20261031/REPORT.md)。
当前超参数仍是教学起点，多 seed 结果显示明显波动，尚未稳定解决任务。

## 文件和学习顺序

```text
SimHash/
├── README.md             课程路线、约定和命令
├── config.py             已提供：集中放置超参数
├── model.py              已提供：residual Q 网络
├── exploration.py        第 1 课：特征、SimHash、计数奖励（3 处填空）
├── replay.py             第 2 课：经验回放（2 处填空）
├── dqn.py                第 3 课：动作选择、Bellman loss（2 处填空）
├── train.py              第 4 课：环境交互与回合管理（2 处填空）
├── evaluate.py           已提供：独立的贪心策略评估
├── checks/
│   └── check.py          按课自检，不含参考答案
└── notes/
    └── experiments.md    预测、实验对照和结果记录，由你填写
```

搜索 `TODO` 即可找到填空。每处都有输入输出、实现步骤或验收提示。
推荐一次只做一课；先读函数注释，自己写实现，再运行该课检查。

## 第 0 课：先认清网络的角色

`model.py` 参考 `../../mountaincar-ppo-rnd/model.py` 的 residual 分支：
输入投影、RMSNorm、SiLU、无 bias 的残差 FFN、零初始化可学习 alpha，
并保留 128 宽度、2 个主干 block 和 1 个 head block。
这里使用一个输出 3 个 Q 值的 head；Q 值可以为负，无需 softmax。
固定观测缩放保存在 buffer，随 `state_dict` 一起保存。
代码在本目录独立定义，不跨子项目导入，也不会修改原 PPO-RND。

思考：如果 alpha 初始为 0，首次反向传播时，FFN 参数和 alpha 的梯度分别是什么？

## 第 1 课：哈希计数不是神经网络训练

实现 `exploration.py` 的 TODO 1a–1c：

```text
next_obs [2] → 固定缩放特征 φ(s) [d]
             → A @ φ(s) [k] → >=0 → tuple 哈希 key
             → N[key] += 1 → bonus = 1 / sqrt(N[key])
```

`A` 一次随机初始化后固定；字典只保存访问过的桶，无需申请 2^k 大小的数组。
两个状态落入同一桶，就共享次数。这是哈希聚合计数，不是真实连续状态的精确计数。
本练习对**后继状态**计数，先加次数再计算奖励：第一次 bonus=1。
计数跨回合累计；reset 初始状态不另计数；回放采样不增加次数。

二维上有一个值得亲自验证的限制：对任何正数 c，
`sign(A @ x) == sign(A @ (c*x))`，因此纯 SimHash 无法区分同方向的不同距离。
在二维中 k 条过原点的直线至多切出 2k 个开放扇区，不能把 k=32 理解为
必然有 2^32 个可用空间区域；恰在边界的状态还取决于符号约定。

默认 `append_constant=False` 保持简单的二维随机超平面版本。
完成后再比较 `--append-constant`：将特征变成 `[scaled_position, scaled_velocity, 1]`，
相当于在原二维空间加入固定随机偏置。它改变了特征和分桶方式，是额外实验，
不应当直接称为论文配置的复现。Q 网络输入始终是原始二维观测。

## 第 2 课：存下采集当时的奖励

实现 `replay.py` 的 TODO 2a–2b。每条 transition 保存：

```text
(obs, action, reward_train, next_obs, terminated)
reward_train = ext_reward_scale * reward_ext + beta * bonus(next_obs)
```

本教程将 bonus 在采集时固定下来，回放时不根据最新计数重算。
这会保留较早样本较大的探索奖励，是这里明确选择的实现约定。
原始环境回报另行记录，不能拿混合奖励判断任务是否解决。

## 第 3 课：DQN 学习什么

实现 `dqn.py` 的 TODO 3a–3b。普通 DQN 的单步目标为：

```text
q_sa = Q_policy(obs, action)
y = reward_train + gamma * (1 - terminated) * max_a Q_target(next_obs, a)
loss = Huber(q_sa, y)
```

两边形状都应是 `[B]`；target 分支不参与反传。
这里使用 target 自己取最大值，是普通 DQN。
每 1000 个环境步复制 policy 参数到 target，使用硬更新。

`terminated` 和 `truncated` 有不同用途：

| 信号 | 是否结束本回合并 reset | TD target 是否保留 bootstrap |
| --- | --- | --- |
| 到达目标 `terminated=True` | 是 | 否 |
| 到达时间限制 `truncated=True` 且未 terminated | 是 | 是 |

本练习把 TimeLimit 看作采样截断。如果换成有限时域任务定义，还需重新考虑该约定。

## 第 4 课：把数据流接起来

实现 `train.py` 的 TODO 4a–4b。单步顺序是：

```text
epsilon-greedy → env.step → next_obs 的计数与 bonus
→ 混合奖励入池 → 条件满足时更新 DQN → 定期同步 target
→ 推进 obs → 若回合结束则记录并 reset
```

`episodes.csv` 每行记录一个完整回合；最后一个未完成回合不会写成完整回合。
`bonus_sum` 指未乘 beta 的 bonus 总和，`unique_buckets` 指训练至今累计访问桶数。
输出目录默认使用 UTC 时间戳、seed 和随机后缀，避免覆盖；也可以手动指定一个新目录。
训练结束保存 `q_network.pt` 供评估使用，暂不支持断点续训。

## 运行和检查

从仓库根目录在 PowerShell 执行，复用仓库现有 Docker 环境：

```powershell
# 第 0 课已配好，可以先运行
docker compose run --rm -w /workspace/mountaincar-dqn-count/SimHash runner python checks/check.py model

# 每完成一课检查一次；未填空时会打印对应 TODO 并以状态码 2 退出
docker compose run --rm -w /workspace/mountaincar-dqn-count/SimHash runner python checks/check.py hash
docker compose run --rm -w /workspace/mountaincar-dqn-count/SimHash runner python checks/check.py replay
docker compose run --rm -w /workspace/mountaincar-dqn-count/SimHash runner python checks/check.py dqn

# 全部填完：先短跑 2500 步，覆盖 warmup 后的更新和 target 同步
docker compose run --rm -w /workspace/mountaincar-dqn-count/SimHash runner python train.py --steps 2500 --beta 0

# 正式对照：相同预算和 seed，先基线，再加 bonus
docker compose run --rm -w /workspace/mountaincar-dqn-count/SimHash runner python train.py --beta 0 --run-dir runs/baseline-seed0
docker compose run --rm -w /workspace/mountaincar-dqn-count/SimHash runner python train.py --beta 0.1 --run-dir runs/simhash-seed0
docker compose run --rm -w /workspace/mountaincar-dqn-count/SimHash runner python evaluate.py runs/baseline-seed0/q_network.pt --episodes 20
docker compose run --rm -w /workspace/mountaincar-dqn-count/SimHash runner python evaluate.py runs/simhash-seed0/q_network.pt --episodes 20
```

本地已安装仓库依赖时，也可进入本目录运行 `python checks/check.py model` 等命令。
默认 CPU；需要 GPU 时按根 README 叠加 `compose.gpu.yaml` 并传 `--device cuda`。
评估使用另一组 seed，只按原始环境奖励和是否到达目标报告结果，不使用 bonus。
自检通过说明对应局部行为正确，并不保证训练收敛。

## 已有实验的分析

见 [三组 seed0 模型的评估诊断](analysis/REPORT.md)：包含成功率与平均回报的分解、
逐回合配对比较、失败轨迹、哈希分区图，以及另一组 500 个评估起点的复核。
新增 `analyze.py` 可重新分析已有 checkpoint，原始数据保存在 `analysis/` 下。
训练循环已修复每回合重复设 seed 的问题；旧 checkpoint 保留，新训练请使用新 run 目录。

后续已用 GPU Docker 完成 9 次随机起点主训练和 3 次固定起点 seed0 配对重训，
见 [新报告](analysis/reset-generalization-20261031/REPORT.md) 与 [实验脚本说明](experiments/README.md)。
固定起点重训完整复现旧模型；放开起点后，10 万步预算下仍有明显训练 seed 波动。

## 参考

- [Tang 等：#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1611.04717)，本地材料见 [theory PDF](../theory/1611.04717v3.pdf)。论文用哈希聚合状态并通过计数提供探索奖励；本教程是 MountainCar + DQN 的简化练习。
- residual 网络来源：[本仓库 PPO-RND model.py](../../mountaincar-ppo-rnd/model.py)。
