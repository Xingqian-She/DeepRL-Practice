# Reset 泛化训练对照

本轮主实验按三种配置 × 三个训练 seed 组织，并补充三种配置的固定起点 seed0 配对重训。
共 12 次正式训练。每次训练固定 100000 环境步，
使用仓库的 GPU Docker 镜像和原 residual Q 网络。
只在首次 reset 时设训练 seed，之后让环境的随机数发生器继续生成起点。

| 配置名 | beta | append_constant |
| --- | --- | --- |
| baseline | 0 | false |
| simhash | 0.1 | false |
| append | 0.1 | true |

三个训练 seed 是 0、1、2。每个最终 checkpoint 用相同的评估 seed：
10000–10499 与 20000–20499，共 1000 回合，贪心动作、原始奖励。
还会在相同环境下重新评估原来的三个 seed0 模型。

## 目录

```text
SimHash/
├── experiments/
│   ├── reset_study.py                    可复用的调度与分析脚本
│   ├── fixed_start_control.py            三个 seed0 固定起点配对重训
│   ├── finish_reset_study.py             等待主实验完成后自动运行补充对照与分析
│   └── README.md
├── runs/
│   ├── baseline-seed0/                   旧模型，保留
│   ├── simhash-seed0/                    旧模型，保留
│   ├── simhash-seed0-appendconstant/     旧模型，保留
│   ├── reset-study-smoke/                三组 GPU 短跑，仅验证流程
│   └── reset-generalization-20261031/
│       ├── manifest.json                环境、配置、命令、源码和旧模型哈希
│       ├── status.json                  调度进度
│       ├── source/                      训练开始前冻结的 Python 源码
│       ├── logs/                        每次训练的终端日志
│       ├── fixed-controls/              补充对照的代码、配置、日志、起点和模型
│       └── runs/<配置>-seed<种子>/
│           ├── config.json
│           ├── episodes.csv             原始回报、成功、bonus、桶数
│           ├── resets.csv               每回合实际起点及 seed 参数
│           ├── complete.json            用时、起点统计、GPU 信息和模型哈希
│           └── q_network.pt
└── analysis/reset-generalization-20261031/
    ├── REPORT.md                        结果解释与限制
    ├── summary.json                     完整统计与来源
    ├── per_run.csv                      每个训练 seed 的最终结果
    ├── by_variant.csv                   按配置汇总训练 seed 均值与标准差
    ├── per_episode.csv                  全部评估回合
    └── *.png                            结果与训练曲线
```

未移动或覆盖旧实验。训练原始文件保存在 runs 内，分析结果放在 analysis 内。
所有命令的目标目录必须尚不存在，避免不小心覆盖前一次实验。

## 复现

从仓库根目录，在 PowerShell 中运行。重新运行请换一个 study/output 名称。

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm -T -w /workspace/mountaincar-dqn-count/SimHash runner python -u experiments/reset_study.py run --study runs/reset-generalization-20261031 --steps 100000 --seeds 0 1 2 --jobs 3

# 上面的 9 次训练全部结束后再执行
docker compose -f compose.yaml -f compose.gpu.yaml run --rm -T -w /workspace/mountaincar-dqn-count/SimHash runner python -u experiments/reset_study.py analyze --study runs/reset-generalization-20261031 --output analysis/reset-generalization-20261031
```

若要连同三个固定起点配对对照一起做，可以在主实验运行时启动以下命令；
它会等待九次主训练完成，再占用 GPU 补跑，最后自动分析。此时不再单独执行上面的 analyze。

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml run --rm -T -w /workspace/mountaincar-dqn-count/SimHash runner python -u experiments/finish_reset_study.py --study runs/reset-generalization-20261031 --output analysis/reset-generalization-20261031
```

每个 worker 独立进程、独立 RNG；训练设备强制为 CUDA，不允许自动回退 CPU。
最多同时跑三个 worker，CPU 数学库各限制一个线程。训练过程不启用 torch.compile，
也不调整网络、epsilon、学习率、奖励缩放等算法设置。
GPU 运算没有强制逐位确定性，因此相同 seed 也不应被解释为跨环境的逐位复现保证。

完成后的结果见 [本轮报告](../analysis/reset-generalization-20261031/REPORT.md)。
`refine_study_plot.py <analysis目录>` 可仅从已保存 CSV 重画起点图，
减少横轴刻度并记录绘图脚本来源；它不训练模型，也不改变评估数据。

## 怎样解读

统计时先对每个训练 seed 求成功率和平均回报，再计算三个训练 seed 之间的
均值和样本标准差。不要把 3000 个评估回合当成 3000 次独立训练。
成功回合的平均步数应与成功率一起看，不能只挑能成功的回合判断模型优劣。

补充对照使用同一份冻结的训练源码，通过环境 wrapper 在每回合 reset 时强制设置 seed0，
模拟旧代码的固定起点行为，并记录实际生效的 seed。它与主实验 seed0 的区别只有 reset 方式。
还会逐张量检查补充对照与旧 checkpoint 是否相同。
另外两个新训练 seed 用于检查 varied-start 配置的稳定性；固定起点组只重训了 seed0，
因此不能据此估计 reset 方式对多个训练 seed 的平均因果效应。
