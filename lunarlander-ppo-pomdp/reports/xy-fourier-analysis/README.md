# XY Fourier 报告的复现数据

主报告：[XY_PO01 / XY_PO03 固定 Fourier 输入实验报告](../XY_PO01-XY_PO03-fourier-analysis.md)。

从仓库根目录运行 `python -B lunarlander-ppo-pomdp/reports/scripts/analyze_xy_fourier.py`，依赖 NumPy、Pandas、Matplotlib，以及同目录已有的 `analyze_stage2.py`。不导入训练代码、不反序列化 checkpoint、不启动训练或评估，只生成本目录的 CSV / JSON / tables.md 以及 `../assets/xy-*.png` 六张图。重运行会覆盖生成文件，不覆盖人工报告和本说明。

数据按 `state.json` 当前 attempt 选择：新批次 `xy-fourier-10m-v1` 的 XY_PO01 / XY_PO03，以及历史 `course-10m-v1` 的 PO01 / PO03。两批次各保留自己的源码指纹，历史批次其他组、烟测和非当前 attempt 不参与分析。

| 文件 | 内容 |
|---|---|
| `audit.json` / `artifact-audit.csv` | 当前 attempt 的文件 SHA-256、来源、配置、环境和逐回合统计核验；权重只校验哈希 |
| `training-seeds.csv` / `training-groups.csv` | 12 次训练的单种子统计及跨种子均值 / 样本 SD |
| `heldout-seeds.csv` / `heldout-groups.csv` | 72 次最终评估、24 个条件的汇总 |
| `heldout-episodes.csv` | 7,200 条最终评估回合，真实终止状态只用于离线分析 |
| `periodic-evaluations.csv` | 588 个周期评估点，各自使用当时课程难度 |
| `seed-differences.csv` / `difference-groups.csv` | 同种子编号新减旧差值、样本 SD、正差值数量及逐一去掉一个种子的均值范围 |
| `transitions.csv` / `performance-stages.csv` | 60 次课程转换、门槛前后评估，以及训练日志重算的阶段预算 |
| `diagnostics.csv` | 1,188 条固定批次梯度探针，本报告不将探针当作独立训练样本 |
| `phase-seeds.csv` | 新模型的训练与周期日志按每次训练聚合，最终评估逐任务保留，共 48 行 |
| `phase-groups.csv` | 新模型按组和记录范围聚合的相位计数与最大值，共 6 行 |
| `tables.md` | 脚本生成的最终评估表 |

各指标先在单个训练种子内计算，再等权聚合三个种子；跨种子 SD 使用 `ddof=1`，单次评估 `std_return` 使用 `ddof=0`。比率列为 0–1。某种子不存在某类终止回合时，该类别的条件均值为空，聚合仅包含有观测的种子。

`p1_full_rollout_steps` 只统计开始时 p=1 的完整 rollout；`p1_entry_step` 为首个此类 rollout 前累计步数；`p1_entry_lr` 为该更新学习率。`p1_lr_mass_fraction` 是名义学习率求和占比，不是实际参数更新量。新批次这些字段与元数据交叉核验；历史字段由训练日志推导。

相位计数求和，最大相位取最大值；不把最大值相加。训练相位转移数加跳过数等于预算，评估相位转移数加跳过数等于回合长度之和。日志没有逐步坐标，脚本不能独立重建每次相位增量；它核验已保存相位字段、哈希与计数关系。

`elapsed_hours` 来自训练元数据，`attempt_wall_hours` 来自任务起止时间，计时边界不同。历史未记录的 `cache_bytes` 留空。训练环境与离线分析环境分别保存在 `audit.json`，不要混淆 NumPy 等版本。

脚本逐运行检查其自身批次源码指纹，另记录当前源码是否匹配新批次。未来源码变化不妨碍复算已保存数据。相同种子编号不代表严格匹配的模型初始化；两个批次的源码也不同。脚本针对固定批次、预算和评估规模，不能直接用作任意批次的通用汇总器。
