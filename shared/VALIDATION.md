# 实际验证记录

日期：2026-11-02。基于仓库 HEAD `90fb83b`，本次仅新增公共组件并更新根目录入口，未修改原有算法代码。

## 环境与正确性

- Docker 项目镜像 `deeprl-practice:py311-cu128`，Python 3.11.14，PyTorch 2.9.1+cu128，CUDA 12.8。
- GPU：NVIDIA GeForce RTX 4090 Laptop GPU，支持 BF16。
- 本地 editable 安装：`pip install --no-index --no-deps --no-build-isolation -e ./shared`，未下载新依赖。

| 检查 | 实际结果 |
|---|---|
| CPU 行为测试 | 16 项通过，3.293 秒 |
| CUDA 测试 | 3 项通过，12.910 秒；包含 BF16/FP16、延迟反向传播、真实 Inductor fullgraph 训练 |
| 环形缓存与整序列一致性 | 23 步、W=4、不同 episode 重置、QK/末尾 Norm 开关组合通过；门控 0.7，避免近恒等映射掩盖错误 |
| 当前步梯度 | 输入梯度与整前缀末步参考一致；历史无跨步梯度；后续写入、回绕、重置后仍可反向传播 |
| 序列隔离 | 因果、episode、padding、自定义 bool/float/head mask、全屏蔽行均通过 |
| 示例 | CPU 运行通过；流式/并行结果一致，最终位置 `[20,13,9]`，当前输入梯度 norm 约 0.95927 |
| 独立 wheel 安装 | 构建并安装 wheel 后，从 `/tmp` 导入 `site-packages/rl_shared` 并执行 `(1,2,16)` 前向通过；`pip check` 无依赖冲突 |
| 静态检查 | 11 个 Python 文件 AST 解析、组件文档相对链接、新增文件行尾空白、`git diff --check` 均通过 |
| 原 MountainCar PPO-RND 回归 | 10 项通过，4.579 秒 |
| 原 CartPole PPO 回归 | 5 项通过、1 项既有错误，详见下文 |

完整命令见 [README](README.md)。编译测试包含 packed episode、padding 和 T>W 的窗口掩码，不只是导入或空调用。

## 性能实测

默认基准：B=32、完整前缀 T=512、D=128、2 层、4 heads、W=256、FFN=384、alpha=1e-4；FP32 参数配合 CUDA BF16 autocast。每项先 warmup 5 次，再测量 30 次；逐次 CUDA 同步，计入 Python 调度与算子发射开销，不含初始化、编译、warmup 和 profiler。训练/反向项目包含 zero_grad，不含 optimizer。

| 路径 | eager 中位 ms | eager P95 ms | eager 峰值 allocated MiB | 编译并行路径后中位 ms |
|---|---:|---:|---:|---:|
| 完整训练 forward + backward | 7.603 | 8.870 | 337.40 | 3.761 |
| 完整前缀重算，仅保留末步输出 | 2.789 | 3.740 | 126.50 | 1.067 |
| KV Step，关闭梯度 | 2.331 | 2.442 | 34.90 | 2.299 |
| KV Step，开启梯度，仅 forward | 2.761 | 3.024 | 43.31 | 2.794 |
| KV Step，开启梯度，forward + backward | 6.280 | 7.436 | 49.45 | 6.522 |
| 每步局部重置，关闭梯度 | 2.353 | 2.448 | 34.90 | 2.390 |
| 每步局部重置，开启梯度并 backward | 6.223 | 6.717 | 49.45 | 7.040 |

`--compile` 仅编译完整训练与前缀路径，Step 仍是相同 eager 实现；表中 Step 两列的差异是独立运行波动。局部重置每次作用于固定 4/32 个槽位（ceil(10% × 32)，实际 12.5%）。

- eager 训练吞吐约 215 万 token/s，编译后约 436 万 token/s；这次观测到约 2.02 倍。
- 无梯度缓存 Step 吞吐约 13,729 个环境步/s，相对 eager 前缀重算约 1.20 倍；K/V 与元数据常驻共 8,391,168 bytes（约 8.003 MiB）。
- 当前小模型中，**编译后的完整前缀重算比 eager Step 更快**；缓存显存优势与计算量优势不保证在每种尺寸下都转化为延迟优势。这里只编译了并行路径，不能将该对比理解为缓存方案的最佳性能上限。
- Step 开启梯度需要独立 K/V 快照；这次仅 forward 的增量峰值显存由约 1.00 MiB 增至 9.41 MiB，代价已计入测试。
- 当前显式窗口/缓存掩码观察到 `aten::_scaled_dot_product_efficient_attention`；反向相应使用 efficient attention backward。未强制 FlashAttention，也未安装第三方 attention 库。
- 数据为一次本机运行，受温度、功耗及系统负载影响；合成损失与性能基准均不代表任何 RL 任务的学习成绩。

原始报告保存在 [eager JSON](benchmarks/results/2026-11-02-eager.json) 和 [compiled JSON](benchmarks/results/2026-11-02-compiled.json)。重跑输出默认可放在被 Git 忽略的 `shared/runs/`。

## 既有 CartPole 测试问题

`test_short_training_with_repeated_timeouts` 在 `cartpole-ppo/test_rollout.py:136` 报错：

```text
KeyError: '_orig_mod.actor.net.0.weight'
```

测试拿 compiled trainer 的 `state_dict()` 键与原始 agent 保存的 checkpoint 键直接比较，二者存在 `_orig_mod.` 前缀差异。已将未修改 HEAD 的整个 `cartpole-ppo` 子目录通过 `git archive` 导出到容器临时目录，单独运行同一测试集，复现相同错误（5 通过、1 错误，4.203 秒）。本次没有导入 rl_shared，也没有修改该算法或测试；保留此既有问题供后续单独处理。
