# CartPole PPO

使用独立 Actor/Critic 和 GAE 训练 `CartPole-v1`。默认 4 个并行环境，
每个环境每次采样 128 步，目标总步数 500000（只执行完整 rollout）。
完整训练的收敛效果待验证。

在已安装根目录依赖的开发容器中运行：

```bash
cd /workspace/cartpole-ppo
python train.py
```

训练完成后，在支持 WSLg 的开发容器终端中观看智能体动画：

```bash
cd /workspace/cartpole-ppo
python evaluate.py
```

默认加载 `runs/checkpoints/ppo_model.pth`，使用 CPU 玩 5 局，每局打印得分。
评估选择策略中概率最高的动作；可在 `evaluate.py` 顶部修改 `NUM_EPISODES`。
在终端按 `Ctrl+C` 可提前停止。

产物始终保存在本项目的 `runs/`，不受启动时工作目录影响：

```text
runs/
├── training.csv
├── training.png
└── checkpoints/
    └── ppo_model.pth
```

按本项目当前需求使用固定文件名，重复训练会覆盖同名产物，不创建时间戳目录。
CSV 和图片每 10 次 update 及最后一次 update 更新一次，包含本次训练每次
update 的数据。模型权重在训练正常结束时覆盖保存。

曲线横轴为累计环境步数；四个指标为 policy loss、value loss、entropy 和
clip fraction。前三项沿用 trainer 返回的最后一个 minibatch 指标，clip fraction
为本次 update 所有 minibatch 的平均值。这些曲线不代表 episode reward。

`train.py` 中的 `save_training_logs(history, run_dir)` 负责保存 CSV 和图片：
`history` 是日志字典列表，`run_dir` 是输出目录。绘图使用 Agg 后端，不弹窗。

运行边界与日志保存检查：

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```
