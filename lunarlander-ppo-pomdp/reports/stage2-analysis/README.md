# 第二阶段报告的复现数据

主报告：[PO01–PO03 第二阶段分析](../PO01-PO03-stage2-analysis.md)。

从仓库根目录运行 `python -B lunarlander-ppo-pomdp/reports/scripts/analyze_stage2.py` 可重算 CSV、核验摘要和六张 `reports/assets/po-stage2-*.png`。依赖 NumPy、Pandas、Matplotlib；不需要训练用的 CUDA 或 Box2D 环境。脚本只写报告目录，未改动 `runs/`。

- `audit.json` 和 `artifact-audit.csv`：状态、来源、已登记产物 SHA-256 及逐回合统计复核。冻结第一阶段选型证据也在哈希核验范围内；数值分析只纳入 FO04 和 PO01–PO03。复用的相同文件只计一次。
- `training-seeds.csv` / `training-groups.csv`：12 次训练的逐种子统计及跨种子均值、样本 SD。`p1_full_rollout_steps` 只计 `sensor_failure_probability_start == 1` 的完整 rollout；`p1_lr_mass_fraction` 是每轮名义学习率求和的占比，不能换算为真实参数更新量。
- `heldout-seeds.csv` / `heldout-groups.csv`：72 次留出评估及 24 个条件的汇总。`std_return` 是单次评估内回报的总体 SD；带 `_std` 的跨种子聚合列是三个训练种子之间的样本 SD。比率为 0–1，角度沿用原环境弧度值。
- `heldout-episodes.csv`：7,200 条回合记录。`training_seed` 为训练种子，`seed` 为环境回合种子。`final_*` 是终止时完整真实状态，仅用于离线诊断。
- `periodic-evaluations.csv`：588 个周期评估点，使用当时难度，不能跨不同 `p` 直接排序。
- `transitions.csv`：PO03 的 30 次推进。`gate_*` 是切换前两次评估；`first_after_*` 是后续 100 次更新后的首次下一难度评估，二者使用不同权重和不同轨迹。
- `performance-stages.csv`：按 `curriculum_stage` 从训练日志重算的阶段步数，已与元数据逐项比对。
- `fixed-cliff.csv`：PO02 接近课程终点的周期评估；`policy-differences.csv`：相同训练种子与 `p` 下的 sample−argmax 指标差。
- `tables.md`：脚本直接生成的留出结果表。

所有跨种子统计先在每条训练运行内求指标，再等权汇总三个训练种子。逐回合合并文件及经验分布仅用于描述，不能将回合数作为独立训练重复数。条件均值（例如超时回合平均回报）在某个种子没有对应回合时为空，分组的此类均值只包含存在该类回合的种子。

重运行覆盖本目录的生成文件和六张图，不自动改写人工撰写的主报告或本说明。脚本会对正式实验规模、固定参数与当前源码指纹做严格检查，适用于本报告对应的批次，不是任意实验的通用汇总器。
