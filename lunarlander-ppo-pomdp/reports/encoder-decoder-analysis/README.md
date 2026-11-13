# Encoder–Decoder 报告的复现数据

主报告：[ED_PO01 / ED_PO03 探索实验分析](../ED_PO01-ED_PO03-encoder-decoder-analysis.md)。

从仓库根目录运行 `python -B lunarlander-ppo-pomdp/reports/scripts/analyze_encoder_decoder.py`，依赖 NumPy、Pandas、Matplotlib，以及同目录已有的 `analyze_stage2.py`。不导入训练代码，不反序列化 checkpoint，不启动训练或评估，只写本目录的生成文件和 `../assets/ed-*.png` 六张图。重运行会覆盖生成文件，不覆盖人工报告和本说明。

数据按 `state.json` 当前 attempt 选择：新批次 `encoder-decoder-10m-v1` 的 ED_PO01 / ED_PO03，以及历史 `course-10m-v1` 的 PO01 / PO03。两批次各保留自己的源码指纹。原有烟测、其他组和旧 attempt 不参与比较。

| 文件 | 内容 |
|---|---|
| `audit.json` / `artifact-audit.csv` | 当前 attempt 的登记文件 SHA-256、来源、配置、环境及逐回合统计核验；权重仅校验哈希 |
| `training-seeds.csv` / `training-groups.csv` | 12 次训练的单种子统计、跨种子均值与样本 SD |
| `heldout-seeds.csv` / `heldout-groups.csv` | 72 次最终评估及 24 个条件的汇总 |
| `heldout-episodes.csv` | 7,200 条最终评估回合，含离线诊断用的真实终止状态 |
| `periodic-evaluations.csv` | 588 个周期评估点，使用各自当时课程难度 |
| `seed-differences.csv` | 同训练种子编号的新减旧差值，不表示初始化严格匹配 |
| `difference-groups.csv` | 差值均值、样本 SD、正差值数量及逐一去掉一个训练种子的均值范围 |
| `transitions.csv` | 60 次成绩课程转换、切换前的门槛均分及后续首次评估 |
| `performance-stages.csv` | 训练日志重算的各难度实际步数，已与元数据比对 |
| `diagnostics.csv` | 1,188 条固定批次探针；新模型特有列在历史运行中为空 |
| `tables.md` | 脚本直接生成的最终评估表 |

均值和比率先按训练种子计算，再等权汇总三个训练种子。跨种子 SD 使用 `ddof=1`；单次评估中的 `std_return` 使用 `ddof=0`。比率为 0–1；不是百分数。条件均值（例如某类终止回合的平均回报）若某种子没有该类回合，则该单元格为空，聚合时只包含存在该类回合的种子。

`p1_full_rollout_steps` 只统计开始时 p=1 的完整 rollout。`p1_entry_step` 为首个 p=1 rollout 开始前累计步数；`p1_entry_lr` 为该次更新的学习率。`p1_lr_mass_fraction` 为最终难度下的名义学习率求和占比，不是实际参数改变量。

`elapsed_hours` 沿用训练元数据计时；`attempt_wall_hours` 从状态清单起止时间计算，两者边界不同。历史日志未记录 `cache_bytes`，这里留空，不进行推测补值。`shared_negative_cosine_fraction` 与梯度范数比来自固定探针，不等同于完整 rollout 或优化器实际更新贡献。

脚本检查每次运行与其自身批次源码指纹一致，并记录当前源码是否仍匹配新批次；未来训练代码变动时，仍可复算旧产物，不把历史源码指纹强行要求为当前值。脚本针对本报告的固定批次、预算和评估规模，不是任意实验的通用汇总器。
