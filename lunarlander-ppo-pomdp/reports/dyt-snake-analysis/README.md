# DyT + Snake 报告的复现数据

主报告：[DyT + Snake Encoder–Decoder 实验报告](../DYT_SNAKE_ED_PO01-DYT_SNAKE_ED_PO03-analysis.md)。

从仓库根目录运行 `python -B lunarlander-ppo-pomdp/reports/scripts/analyze_dyt_snake.py`，依赖 NumPy、Pandas、Matplotlib 及同目录的 `analyze_stage2.py`。不导入训练模块、不加载权重推理、不启动训练或评估；只写本目录的生成文件与 `../assets/dyt-snake-*.png` 六张图。重运行不覆盖人工报告或本说明。

数据按各批次 `state.json` 当前 attempt 选择：新批次 `encoder-decoder-dyt-snake-10m-v1` 的两个正式组，对照 `encoder-decoder-10m-v1` 的 ED_PO01 / ED_PO03。各批次使用自己的源码指纹，排除烟测及非当前 attempt。登记配置补齐旧默认项后，仅 norm_type / ffn_type 不同；不因此声称源码相同或独立组件因果已经得到隔离。

| 文件 | 内容 |
|---|---|
| `audit.json` / `artifact-audit.csv` | 306 个登记文件 SHA-256、来源、环境、配方和逐回合统计核验；权重仅校验哈希 |
| `training-seeds.csv` / `training-groups.csv` | 12 次训练的单种子统计、跨种子均值与样本 SD |
| `heldout-seeds.csv` / `heldout-groups.csv` | 72 次最终评估及 24 个条件的汇总 |
| `heldout-episodes.csv` | 7,200 条最终评估回合，真实终止状态仅供离线分析 |
| `periodic-evaluations.csv` | 588 个周期评估点，各自处于当时课程难度 |
| `seed-differences.csv` / `difference-groups.csv` | 同编号训练种子新减旧差值，以及去掉任一种子的平均差值范围 |
| `performance-stages.csv` | 六次课程训练在 11 个阶段的实际环境步数，与元数据比对 |
| `stage-evaluations.csv` | 有周期评估的阶段：预算、评估次数、单次达标数、末期表现、进入步数与学习率 |
| `transitions.csv` | 56 次课程转换及转换前的两个门槛评估、后续第一次评估 |
| `diagnostics.csv` | 1,188 个固定批次梯度探针；不能作为独立训练样本 |
| `tables.md` | 脚本直接生成的最终评估表 |

指标先在单个训练种子内计算，再等权聚合三个训练种子；跨种子 SD 使用 `ddof=1`，单次评估回报 SD 使用 `ddof=0`。比率列为 0–1。某种子不存在某类终止回合时，该类别条件均值留空，聚合只包含有观测的种子。

`p1_full_rollout_steps` 仅统计开始时 p=1 的完整 rollout。`p1_entry_step` 为首个此类 rollout 前的累计步数，`p1_entry_lr` 为其更新学习率。没有进入 p=1 时，训练步数为 0，进入步数、学习率及 p=1 周期峰值留空；不得把空值解释为 step=0。上述预算与训练元数据交叉核验。`p1_lr_mass_fraction` 是名义学习率求和占比，不是实际更新量。

`qualified_count` 是单次同时满足均分 ≥200、≥200 比例 ≥80% 的评估次数，课程晋级还需要连续两次。`last5_*` 只取该阶段最后至多五个周期点，不跨难度汇总。最后停留阶段不同，不能直接据此排名模型。

`late_*` 来自最后 500 次更新；`grad_norm_median` 为逐更新日志值的中位数，`late_grad_norm_median` 限定末 500 次更新。日志 grad_norm 是裁剪前范数。共享梯度比、余弦与 cross-attention 梯度来自固定探针，价值目标已包括 vf_coef，不代表优化器实际更新份额。没有保存 DyT 饱和率或 Snake 频率轨迹，脚本不推测这些值。

`elapsed_hours` 来自训练进程 perf_counter，`attempt_wall_hours` 来自外部起止时间戳；记录存在不一致，分别保留，不混加或当作精确 GPU 小时。`collect_hours` 与 `ppo_hours` 只是相应日志字段之和。`peak_allocated_mib` 为 PyTorch 报告的峰值 allocated，不是整机显存占用。

当前源码是否匹配新批次只记录、不强制；未来代码变化后仍能复算保存产物。脚本针对本报告固定预算、批次与评估规模，不是通用实验汇总器。
