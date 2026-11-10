# 2026-11-06 实施验证

本页只记录本次实际完成的实现测试、短程运行和性能基准。**未启动 FO/PO 正式长训练，没有新的着陆收敛结论。** 实验执行手册见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 环境与数据清理

- Linux 项目容器，镜像 `deeprl-practice:py311-cu128`，Python 3.11.14。
- PyTorch 2.9.1+cu128、CUDA 12.8、Gymnasium 1.2.2、NumPy 2.4.6、box2d-py 2.3.5。
- NVIDIA GeForce RTX 4090 Laptop GPU；训练与评估均 FP32/math SDPA，TF32/AMP 关闭。
- `pip check` 通过，见 [依赖检查](runs/validation/pip-check.log)。
- 已确认没有活动训练进程后，删除本项目旧 runs、archives、results，共 2,259,381,793 字节，约 **2.10 GiB**。旧数据未复制归档。保留其他项目及原有改动。
- 本次新的验证产物集中在 `runs/validation/`，不属于正式实验数据，不进入多种子收敛结论。

## 测试与命令核验

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 本项目完整测试 | **62 项通过**，53.306 秒，无跳过 | [project-tests.log](runs/validation/project-tests.log) |
| 共享组件完整测试 | **24 项通过**，69.509 秒，无跳过 | [shared-tests.log](runs/validation/shared-tests.log) |
| Bash 语法 | 手册 12 个代码块全部通过 | [命令核验 JSON](runs/validation/handbook-check.json) |
| 手册循环模拟 | 42 条训练、252 条评估参数/路径通过；未执行正式训练或评估 | [核验脚本](runs/validation/check_handbook.py) |
| 真实产物汇总 | 4 条训练、4 条独立评估；8 组有效、无不可读数据，未产生课程候选 | [summary.json](runs/validation/summary/summary.json) |

测试包含调度首/末/单轮、原始 GAE 与优势开关、冻结旧价值及裁剪损失梯度、拒绝批次的覆盖率、诊断不改变参数/梯度/Adam/RNG、策略专用推理与缓存一致性、旧 checkpoint、异步终止/截断、真实 Box2D、可解码视频、CUDA 和真实 Inductor 前向/反向。

评估覆盖 sample/argmax、p=0/.5/1、回合数小于并发数、不能整除、不同长度、恰好 200 分、模型权重刷新和异常资源关闭。确定性测试环境使用固定动作分布，验证不同并发数的回合种子和采样轨迹；真实网络跨 batch/设备仅能保证浮点容差，不承诺逐步轨迹相同。

汇总测试覆盖缺失种子/条件、重复运行、混合配置/预算/源码、失败和损坏数据、留出评估不参与选型，以及“残留评估记录不能替代缺失训练条件”。无效组保留问题明细，不计算合并成绩。

## 真实训练与独立评估烟测

以下四条运行各 **64 环境步、2 次 PPO 更新**，D32/L2/H2/context4、最大回合 16 步；同时开启线性学习率、线性熵系数、价值裁剪 .2、关闭优势标准化、每轮诊断和评估：

| 运行 | 执行方式 | 状态 |
|---|---|---|
| [CPU shared](runs/validation/smoke-cpu-shared/metadata.json) | eager | completed |
| [CPU separate](runs/validation/smoke-cpu-separate/metadata.json) | eager | completed |
| [CUDA shared](runs/validation/smoke-cuda-shared/metadata.json) | compile on，实际启用 | completed |
| [CUDA separate](runs/validation/smoke-cuda-separate/metadata.json) | compile on，实际启用 | completed |

四个 checkpoint 又各通过 7 回合、3 环境、seed=20000、sample、p=.5 的真实 Box2D CLI 评估，验证尾部和来源关联。结果目录分别为 `runs/validation/heldout-{cpu,cuda}-{shared,separate}`。回合上限继承上述烟测的 16 步，因此这些分数不用于标准任务比较。

## 评估端到端加速

在上述 Linux/CUDA 环境中，固定未训练 D256/L4/H4/context32、seed42 权重，真实 LunarLander-v3、1000 步回合上限、p=0、sample、评估 seed10000。对照旧串行评估与新 8 环境批量策略推理，各重复三次并交替先后顺序；表中为三次端到端耗时中位数。

| 骨干 | 回合数 | 旧串行 | 新批量 | 耗时加速 |
|---|---:|---:|---:|---:|
| shared | 20 | 8.433 s | 1.224 s | **6.89×** |
| shared | 100 | 41.729 s | 7.960 s | **5.24×** |
| separate | 20 | 15.187 s | 1.206 s | **12.59×** |
| separate | 100 | 75.580 s | 8.121 s | **9.31×** |

批量评估首轮/后续中位数分别为 shared 20 回合 1.224/1.207 秒、100 回合 7.960/7.985 秒；separate 20 回合 1.190/1.210 秒、100 回合 8.159/8.106 秒。首轮计时包含策略副本创建，后续复用。

旧采样使用全局 Torch RNG，新采样使用回合独立 NumPy RNG，轨迹长度不同：20 回合分别为 1854/1849 步，100 回合为 9184/10067 步。按实际 steps/s 比较的加速分别约 **6.87×、5.75×、12.56×、10.20×**。这里没有减少评估回合、缩短回合上限或改变 200 分门槛。

原始证据：[完整 JSON](runs/validation/evaluation-benchmark.json)、[逐次日志](runs/validation/evaluation-benchmark.log)。复现命令：

```bash
python -B benchmark_evaluation.py --device cuda --episodes 20 100 --repeats 3 --num-envs 8 \
  --output runs/new-evaluation-benchmark.json
```

该基准只支持执行速度结论。学会后的模型可能运行更长回合，CPU 物理开销和尾部空槽占比也会变化，不能直接把这些倍数保证到所有 checkpoint。

## 诊断开销与限制

另外使用 CPU D32/L2/H2/context4、非零门控初值 .7 的确定性测试环境，对比关闭诊断与每 50 轮诊断，100 次更新，各重复三次并交替顺序。开启时实际在第 1、50、100 轮采集，共三次。

| 骨干 | 关闭诊断 PPO 耗时中位数 | 开启诊断 | 直接探针耗时占比中位数 |
|---|---:|---:|---:|
| shared | 2.528 s | 2.553 s | 0.56% |
| separate | 4.049 s | 4.036 s | 0.43% |

独立模式的负耗时差属于计时波动，不表示诊断会加速。该小模型结果不能代替默认 D256/CUDA 测量；实际训练逐轮记录 `diagnostic_seconds`，可据此调整间隔。证据：[JSON](runs/validation/diagnostic-overhead.json)、[脚本](runs/validation/check_diagnostics.py)、[日志](runs/validation/diagnostic-overhead.log)。

尚未验证：FO/PO 多种子长训练、5000 万步课程收敛、所有组件组合、等总参数量架构对照和 WSLg 实际弹窗。checkpoint 不提供完整断点续训。评估协议已变为 `batched-policy-v2`，不承诺与旧全局 RNG 的结果逐步相同。

## 统一实验入口验收（2026-11-06）

已增加 Windows PowerShell 启动器、JSON 实验组配置和顺序任务调度器。默认仅 FO01、seed=42/43/44、各请求 1000 万步；使用最终权重执行 18 次留出评估。**本次仍未启动正式 1000 万步训练，也未创建默认 `baseline-v1` 批次。**

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 最终完整项目回归 | **78 项通过**，50.896 秒，无跳过；含原有 62 项及 16 项调度器测试 | [project-tests.log](runs/entrypoint-validation/project-tests.log) |
| Windows 默认预览 | 3 次训练、18 次评估；多组选项 FO01,FO02 为 6 次训练、36 次评估；不启动训练 | [默认配置](experiments.json)、上述测试日志 |
| CPU 完整入口烟测 | 3 条 64 步训练、18 次每项 2 回合评估，21 个任务全部 completed；再次执行全部跳过 | [overview.md](runs/experiments/entrypoint-smoke-v1/summary/overview.md) |
| GPU 完整入口烟测 | 原生 `powershell -NoProfile -ExecutionPolicy Bypass -File ... -Smoke -Device cuda`；21 个任务全部 completed，7 个统计组有效，无不可读结果 | [summary.json](runs/experiments/entrypoint-smoke-cuda-v2/summary/summary.json) |
| SIGINT 中断及恢复 | 向专用容器发送 SIGINT，模拟 Ctrl+C 交付至调度器；训练记录 interrupted，原命令重跑完成 | [生命周期结果](runs/entrypoint-validation/lifecycle-results.json) |
| 关闭启动窗口及恢复 | 强制结束专用 Windows 启动进程；心跳失联停止容器内任务，记录原因，补跑后完成 | [批次状态](runs/experiments/entrypoint-close-v2/state.json) |
| 强制终止容器及恢复 | SIGKILL 后残留 running 状态被新调度器识别；保留第一次尝试，第二次尝试完成 | [恢复报告](runs/experiments/entrypoint-kill-v2/summary/overview.md) |
| 重复启动、只读与分析 | 同批次第二次启动被拒绝且不影响原任务；Status 与 AnalyzeOnly 通过；配置路径含空格通过 | [生命周期核验脚本](runs/entrypoint-validation/check_lifecycle.ps1) |

生命周期测试使用单种子 D32/L2/H2/context4、CPU、2048/4096 步和 2 回合评估，均为独立测试批次。每种中断方式均验证只有两次训练 attempt、一次最终评估 attempt；历史尝试保留，不进入当前成绩汇总。验证后没有残留的专用实验容器，原有两个交互式项目容器保持运行。

调度器测试另覆盖：中断训练和仅评估失败的分别补跑、训练重跑使六项依赖评估失效、损坏 checkpoint（含尚未提交内容哈希的情况）、损坏评估结果、子进程返回成功却缺少产物、完成产物早于状态提交、创建目录后状态尚未提交就中断、追加新实验组、参数/源码/环境冲突、文件锁互斥、路径越界、心跳停止与子进程回收，以及未达标仍正常生成统计。

原有手工运行 `runs/fo-10m-v1/FO01/seed42/train` 保持 interrupted、757760 步、370 次更新，未删除或覆盖。测试批次带有实际执行时的源码指纹；后续源码改变须建立新批次，不改写既有来源字段。

本次未实测 Docker Desktop 从完全关闭状态启动、缺失镜像的联网冷构建；当前机器已有在线 Linux 引擎和匹配镜像，复用与依赖锁检查已验证。物理键盘 Ctrl+C 使用 SIGINT 链路与信号处理测试覆盖，未使用桌面按键自动化。

## 第二阶段一键入口验收（2026-11-09）

新增 `run_stage2.ps1` 与独立 `stage2/` 包，复用现有调度器和 Docker/心跳启动机制。正式默认从 `baseline-v1` 只读核验七组训练、重新选出 FO04，再核验其 18 次留出评估；第二阶段新增 9 次训练、54 次评估，FO04 的 3 条训练和 18 条评估仅引用。

本次只运行预览、测试和独立短程验收批次，**未创建正式 `course-10m-v1`，未启动正式 1000 万步 PO 训练，也未修改第一阶段冻结配置或权重。**

| 检查 | 实际结果 | 证据 |
|---|---|---|
| 项目完整回归 | **96 项通过，无失败、无跳过**，207.43 秒；含原有 78 项及新增 18 项第二阶段测试 | [测试日志](runs/stage2-validation/project-tests.log)、[结果 JSON](runs/stage2-validation/project-tests.json) |
| Windows 只读预览 | 第二阶段默认 9 次训练/54 次评估，PO01,PO02 为 6/36；第一阶段默认仍为 3/18 | [只读核验](runs/stage2-validation/readonly-checks.json) |
| 最终版 CPU 一键烟测 | 9 条 64 步训练、54 次每项 2 回合评估，**63/63 completed**，21 个统计组有效 | [CPU 概况](runs/experiments/stage2-cpu-validation-v2/summary/overview.md)、[成绩与课程](runs/experiments/stage2-cpu-validation-v2/summary/summary.json) |
| 最终版 GPU 一键烟测 | Windows 入口 `-Smoke -Device cuda`，**63/63 completed**，21 个统计组有效；实际使用 RTX 4090 Laptop GPU | [GPU 概况](runs/experiments/stage2-cuda-validation-v2/summary/overview.md)、[成绩与课程](runs/experiments/stage2-cuda-validation-v2/summary/summary.json) |
| 同批次重复启动 | Windows 实测明确拒绝第二个启动器，原 GPU 烟测继续完成，所有训练和评估均只有 attempt 1 | [重复启动日志](runs/stage2-validation/stage2-cuda-validation-v2-duplicate.log) |
| 再次执行 | 最终 CPU、GPU 批次均跳过全部 **63** 个任务；所有任务仍只有 attempt 1，无新增训练或评估 | [CPU 复跑](runs/stage2-validation/stage2-cpu-validation-v2-repeat.log)、[GPU 复跑](runs/stage2-validation/stage2-cuda-validation-v2-repeat.log) |
| AnalyzeOnly 与 Status | 最终 CPU、GPU 批次均成功；分析无新任务，状态查看不依赖训练设备初始化 | [最终核验 JSON](runs/stage2-validation/final-checks.json) |
| 来源与数据保全 | 第一阶段 **525** 项已登记产物哈希全部匹配，状态文件和训练源码保持原样；两条最终烟测另核验 **450** 项产物哈希 | [最终核验 JSON](runs/stage2-validation/final-checks.json)、[核验脚本](runs/stage2-validation/final_audit.py) |

两条最终烟测都使用 D32/L2/H2/context4、2 环境、最大回合 16 步、64 步预算、两次 PPO 更新、关闭编译。短回合不用于标准任务收敛结论；正式 FO04 不混入烟测成绩。真实 PO03 三个种子均记录为进程 completed、最终 p=0、未进入 p=1、最终任务未达标，保留 64 步阶段计数，未将预算完成误报为课程完成。

新增测试使用临时目录中的小模型、可加载 checkpoint 与受控成绩，验证完整七组训练后才能选型、只按周期成绩选择 FO04、缺失或损坏产物拦截、源码/环境不匹配、快照固定、参考 attempt 改变后拒绝混用、第一阶段只读复用、三组参数差异、仅评估失败不重训、训练中断以新 attempt 重跑、权重损坏后仅重训对应种子及六项评估、追加课程组、互斥锁、只读预览、仅分析和脱离参考目录查看已保存状态。正式汇总的受控数据检查确认 63 条新运行 + 21 条 FO04 引用，共 84 条运行、28 个有效组。

验收中修复了旧 PowerShell 启动器在容器已存在分支中 `String.Split` 的重载歧义，显式使用字符数组分隔符。修复后使用新的 `*-validation-v2` 批次重新完成 CPU/GPU 全链路，未改写旧验收批次的来源指纹。

验收结束后，专用测试容器均已退出，无遗留训练或评估进程。第一阶段训练源码指纹保持 `47786bd7215381980b7c74452a38ab74173a1fa3ed34b188166ef49ad271f179`；协调代码指纹单独保存在各第二阶段快照及最终核验 JSON 中。
