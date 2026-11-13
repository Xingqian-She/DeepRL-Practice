# LunarLander PPO-POMDP

用 PPO + GAE 和时序 Transformer，研究策略如何适应传感器独立失灵。支持共享、独立和因果 Encoder–Decoder 策略价值骨干，学习率衰减、优势标准化、价值裁剪和熵系数调度均可对照。周期评估使用策略专用副本与批量推理，默认 8 个环境、20 回合。

**第一阶段 FO01–FO07 已完成，按既定规则选择 FO04。** 该配置采用独立 Actor/Critic 骨干与线性学习率；各组成绩、训练稳定性和传感器缺失泛化结果见[第一阶段报告](reports/FO01-FO07-stage1-analysis.md)。

**第二阶段 PO01–PO03 已完成，9 次训练和 54 次留出评估全部结束。** 两种课程在 `p=0.5` 下的 sample 留出均分约为 250 分，但在 `p=1` 下仍未达标，直接部分观测训练的最终留出成绩更高。成绩课程三个种子均到达 `p=1`，切换后却出现明显失分；课程模型在该条件下的 argmax 成绩也明显低于 sample。完整对照、六张分析图、复现脚本和后续验证建议见[第二阶段报告](reports/PO01-PO03-stage2-analysis.md)。运行完成、课程推进和任务达标分别报告。

运行规划和可复制命令见 [EXPERIMENTS.md](EXPERIMENTS.md)，实现测试、烟测与性能基准的历史记录见 [VALIDATION.md](VALIDATION.md)。

## Windows 统一实验入口

从 Windows PowerShell 在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_experiments.ps1
```

默认运行 **FO01、seed=42/43/44、每个请求 1000 万步**，然后用各自最终 `latest.pt` 完成 p=0/.5/1 × sample/argmax、每项 100 回合的 18 次留出评估。默认要求 CUDA；任务顺序执行，不自动切换 CPU，不自动启动课程或 5000 万步实验。脚本检查 Docker Desktop、镜像依赖锁、Python、GPU、依赖和 Box2D；宿主机不需要安装训练用 Python 包。首次缺少镜像时会构建，已有匹配镜像直接复用。

下列参数直接加在上述命令末尾：

| 参数 | 用途 |
|---|---|
| `-DryRun` | 预览默认 3 次训练、18 次评估，不启动训练或写实验状态 |
| `-Smoke` | 在独立 `smoke-v1` 批次用 CPU、小模型、64 步训练和每项 2 回合评估检查完整链路 |
| `-Smoke -Device cuda -Batch smoke-cuda-v1` | 同样的短程链路，使用 GPU；烟测始终关闭编译 |
| `-Status` | 查看批次最后保存的任务状态，不启动训练 |
| `-AnalyzeOnly` | 重新检查已有产物并生成统计，不补跑任务 |
| `-Groups FO01,FO02` | 选择多个实验组；完整有效的 FO01 任务会跳过 |
| `-Batch baseline-v2` | 使用新的独立批次 |
| `-Config .\lunarlander-ppo-pomdp\experiments.json` | 使用仓库内的指定配置文件 |

`-Batch`、`-Smoke`、`-Groups` 等选择参数在补跑时保持一致。烟测默认批次为 `smoke-v1`，正式运行默认批次为 `baseline-v1`；`-Status` 和 `-AnalyzeOnly` 也可指定对应批次。检查/分析模式需要已有匹配镜像，不会自动构建镜像。

执行过程在窗口显示，也会逐任务保存控制台日志。Ctrl+C 停止当前任务和后续队列；关闭窗口后，容器通过心跳失联检测停止任务（约 30 秒检测，随后清理）。再次执行相同命令会核对实际预算、checkpoint 和评估来源，跳过完整任务，只将未完成或产物损坏的任务从头重跑。每次启动遇到第一个失败任务就停止，不自动循环重试。

产物保存在 `runs/experiments/<批次>/`：

```text
state.json                            任务清单、状态、参数/源码/环境与 attempt 历史
tasks/<任务>/attempt-0001/console.log  本次尝试的完整控制台日志
tasks/<任务>/attempt-0001/run/         原有训练或评估产物
summary/overview.md                   先看此文件：任务完成情况、实际步数和耗时
summary/summary.md                    各条件的成绩、达标比例与问题
summary/runs.csv                      逐训练种子的结果
summary/groups.csv                    跨训练种子均值和样本标准差
summary/tasks.csv                     任务状态和目录
summary/summary.json                  原始结构化汇总
```

旧 attempt 保留，汇总只读取各任务当前选定的 attempt。仅评估中断不会重新训练；训练重跑后，其旧留出结果失效并重新评估。原有手工运行目录（包括 `runs/fo-10m-v1`）保留，不自动导入或删除。checkpoint 不提供完整断点续训。

运行完成与成绩达标分别报告；未学会着陆也会正常产出统计。留出回合不能当作独立训练种子。入口不自动冻结课程配置，`selection.json` 只描述本批次已登记组的候选，不替代手册要求的七组对照。

### 添加实验组

配置集中在 [experiments.json](experiments.json)，预置 FO01–FO07；`training` 是公共参数，组内 `training` 递归覆盖差异项。例如，在 `groups` 对象中追加：

```json
"FO08": {
  "description": "较低初始学习率",
  "training": {"learning_rate": 0.0001}
}
```

随后使用 `-Groups FO08`。只增加新组可追加到原批次；改变公共参数、既有任务参数、预算、种子、留出设置、源码或运行环境时，使用 `-Batch 新名称`，避免混入不可比较的数据。默认实验组由 `default_groups` 指定。

容器内也可执行 `python -B run_experiments.py --dry-run` 或 `--smoke --batch my-smoke`；Windows 启动器负责的窗口心跳只在通过 PowerShell 启动时启用。

## 第二阶段一键入口

第一阶段 `baseline-v1` 完成后，在仓库根目录执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_stage2.ps1
```

默认建立独立批次 **`course-10m-v1`**，只读核验七组训练和 FO04 的参考评估，按原定周期门槛重新确认 FO04，然后自动保存第二阶段的冻结配置。无需手工复制 `selection.json`，也不修改第一阶段状态、汇总或权重。

新训练均从随机初始化开始，沿用 FO04 的独立骨干、线性学习率和其余训练参数，仅改变观测方案：

| 条件 | 训练观测 | 次数 |
|---|---|---:|
| FO04 参考 | 引用已有 3 条全观测训练、18 次留出评估 | 不重跑 |
| PO01 | 全程 p=1 | 3 |
| PO02 | 固定课程：前 10% p=0，中间 70% 线性退化，最后 20% p=1 | 3 |
| PO03 | 成绩课程：每次 p 增加 .1，连续两次均分 ≥200、≥200 分比例 ≥80% 才推进 | 3 |

默认 **9 次训练，各请求 1000 万步**（实际各 9,998,336 步），seed=42/43/44；之后使用最终 `latest.pt`，完成 p=0/.5/1 × sample/argmax、seed=20000、8 环境、每项 100 回合的 **54 次新评估**。共 63 个新增任务；FO04 的 21 条引用另行计数。默认要求与第一阶段相同的 CUDA 环境，不自动运行 5000 万步实验。

这些选项直接加在第二阶段命令末尾：

| 参数 | 行为 |
|---|---|
| `-DryRun` | 复核第一阶段、预览任务，不创建第二阶段批次或冻结文件 |
| `-Status` | 只读查看已保存状态；即使参考目录暂不可用，也能查看 |
| `-AnalyzeOnly` | 复核来源和已有产物、重建汇总，不补跑任务 |
| `-Groups PO01,PO02` | 仅登记所选课程组；后续可追加 PO03，完整任务会跳过 |
| `-Batch course-10m-v2` | 指定新的第二阶段批次 |
| `-BaselineBatch baseline-v2` | 指定符合相同七组对照要求的第一阶段批次 |
| `-Smoke` | 在 `course-smoke-v1` 用 CPU、D32 小模型、64 步训练、2 回合评估验证 63 个任务 |
| `-Smoke -Device cuda -Batch course-smoke-cuda-v1` | 使用 GPU 完成相同烟测，关闭编译 |

例如先做烟测，再查看状态或重新分析：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_stage2.ps1 -Smoke
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_stage2.ps1 -Smoke -Status
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_stage2.ps1 -Smoke -AnalyzeOnly
```

正式产物位于 `runs/experiments/course-10m-v1/`。先看 `summary/overview.md` 的新增任务完成状态，再看 `summary/summary.md` 的 FO04/PO01/PO02/PO03 统一条件成绩；`summary/curriculum.csv` 和 `summary/summary.json` 另存最终 p、是否进入 p=1、是否达标、阶段步数和转换记录。固定课程的阶段步数按配置及已完成采样步数推导，成绩课程使用训练器记录的阶段步数。

正式配置及参考 attempt、run_id、产物哈希保存在 `frozen-recipe.json`；烟测使用 `smoke-recipe.json`，不会把正式 FO04 成绩混入烟测统计。`summary/selection.json` 仅标识第一阶段冻结的 FO04，不代表已选出第二阶段最佳课程。不会用不同结束 p 的周期成绩给课程排名。

Ctrl+C、关闭窗口、失败停止、完整产物跳过和 attempt 补跑沿用第一阶段行为：再次执行相同命令即可，未完成训练从头重跑，单项评估失败不重训。参考产物、冻结参数或第二阶段协调代码变化后须使用新批次；训练源码或运行环境与第一阶段不匹配时，需先建立匹配的第一阶段参考。运行和分析期间对第一阶段持有共享只读锁，避免其权重被另一个受管理任务改写。

第二阶段不接受 `-Config`，参数来自校验后的 FO04。容器内可在本项目目录执行 `python -B -m stage2 --dry-run`；训练源码指纹规则保持原样，第二阶段包与 PowerShell 入口单独记录协调代码指纹。

## Encoder–Decoder 独立探索实验

从仓库根目录的 Windows PowerShell 执行：

```powershell
# 预览：6 次训练 + 36 项最终权重评估，不启动正式训练
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_encoder_decoder.ps1 -DryRun

# 小模型、每次 64 环境步、2 次 PPO 更新、每项 2 回合评估
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_encoder_decoder.ps1 -Smoke

# 手动启动正式探索实验：每个训练种子请求 1000 万步
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_encoder_decoder.ps1
```

默认正式批次为 `encoder-decoder-10m-v1`，烟测批次为 `encoder-decoder-smoke-v1`；支持 `-Groups ED_PO01,ED_PO03`、`-Batch`、`-Status`、`-AnalyzeOnly`、`-Device`，沿用统一入口的中断、失败停止和完整任务跳过机制。配置为 [encoder_decoder_experiments.json](encoder_decoder_experiments.json)。容器内对应 `python -B run_experiments.py --config encoder_decoder_experiments.json --batch encoder-decoder-10m-v1`，需要烟测时显式追加 `--smoke` 并换烟测批次名。

新训练从随机初始化开始，保留冻结 FO04 的训练超参数与 FP32/math SDPA、TF32/AMP 关闭的口径；模型改为 D256、Encoder 4 层、Decoder 4 层、4 heads、窗口 32。`ED_PO01` 直接 p=1，`ED_PO03` 使用原成绩课程；各 seed=42/43/44，实际各 9,998,336 环境步。最终 `latest.pt` 按 p=0/.5/1 × sample/argmax、seed=20000、8 环境、100 回合评估。seed=20000 已用于历史分析，这是一轮探索性比较。

两套独立的 20→D `Linear → Tanh` 投影分别输入 Encoder 和 Decoder。Critic 读取 Encoder 的 `H`；每个 Decoder 层依次执行因果自注意力、以 `H` 为 memory 的因果交叉注意力、FFN，Actor 读取 Decoder 输出。交叉读取 `H_t` 合法，训练时保留策略损失回传 Encoder 的计算图。`--num-layers` 控制 Encoder，`--decoder-layers` 控制 Decoder，后者默认 4。

两路最大原始输入感受野分别为 125 和 249，PPO 保守预热前缀为 249，完整学习序列最长 313。预热排除直接损失但保留梯度。三组 KV 共用环形时钟与异步 reset，更新后用实际传感器输入重放重建；bootstrap 只运行价值所需的 Encoder。评估副本保留两套投影、Encoder、Decoder 和 Actor。

`diagnostics.csv` 增加 Encoder 两种目标的梯度范数、交叉注意力策略梯度和交叉残差门控范围；它们来自固定批次探针。`training.csv` 记录缓存字节数，`metadata.json` 记录参数与预热长度。`summary/curriculum.csv` 汇总最终 p、阶段步数、转换、首次 p=1 训练位置与学习率、p=1 完整 rollout 步数以及成本。最后一次评估才进入 p=1 时，进入最终阶段与实际在该阶段训练的步数分别报告。

**对比范围：**容量、策略感受野、价值/策略梯度共享均发生变化，成绩只能用于评估整套架构是否值得继续研究；Decoder 自身也有历史通路，梯度连通不等于已经有效使用 Encoder memory。原 PO01/PO03 的[报告](reports/PO01-PO03-stage2-analysis.md)保留为独立历史参考，新旧结果按各自来源解释。修改后的源码使用新批次，原 Stage2 的同源码参考校验继续有效。

烟测为 D32、2+2 层、窗口 4，回合上限 19，使轨迹跨过 16 步 rollout 更新边界并触发截断；只验证工程链路。生产形状 CUDA 验证及原始日志见 [VALIDATION.md](VALIDATION.md)。

## 单次训练与评估接口

在已安装项目依赖的 Linux 容器中，从本项目目录执行：

```bash
python -m pip install --no-index --no-deps --no-build-isolation -e ../shared
python -B train.py --device cpu --compile off --num-envs 2 --num-steps 16 \
  --total-timesteps 64 --hidden-dim 32 --num-layers 2 --num-heads 2 \
  --context-length 4 --sequence-length 6 --sequences-per-minibatch 3 \
  --max-episode-steps 16 --eval-interval 1 --eval-episodes 2 \
  --run-dir runs/quick-smoke --experiment-id SMOKE
```

这是实现烟测，缩短回合不用于判断是否学会着陆。正式的 1000 万步、多种子对照按实验手册执行。

```bash
python -B evaluate.py --checkpoint runs/quick-smoke/checkpoints/latest.pt \
  --device cpu --episodes 10 --num-envs 8 --seed 20000 --policy sample \
  --sensor-failure-probability 1 --run-dir runs/quick-smoke-heldout --experiment-id SMOKE
python -B train.py --help
python -B evaluate.py --help
```

`--run-dir` 是准确目录，存在时拒绝覆盖；`--output-dir` 是父目录，会创建带时间和 UUID 的唯一子目录，两者互斥。单次示例再次执行时应换一个新目录。新容器启动方式见[仓库说明](../README.md)。

## 任务与输入

环境为 `LunarLander-v3`：离散 4 动作、无风、原始奖励，最大回合默认 1000 步。动作 0/1/2/3 分别为无动作、左侧推、主引擎、右侧推。

原始状态依次为 `x, y, vx, vy, angle, angular_velocity, left_contact, right_contact`。位置和腿接触始终可用；速度、角度、角速度四通道，在每个环境、每个时间步独立以概率 p 缺失，下步可恢复。

模型输入始终为 20 维：

```text
[8 维读数 × availability, 8 维 availability, 4 维上一动作 one-hot]
```

有效提示为 1，缺失为 0；缺失读数置零。真正为零和失灵可区分。回合首步上一动作为全零，执行动作后才在下一条观测中加入动作 one-hot。Actor/Critic 看到相同信息，没有真实完整状态、奖励或课程进度等特权输入。`--no-previous-action` 保留维数并将动作通道置零。

Transformer 读取观测和动作历史。rollout 与历史保存实际看到的传感器输入，PPO 重算、预热和缓存重建不重新抽样缺失模式。真正终止不 bootstrap；时间截断使用旧回合 final observation 估值并切断 GAE，不使用新回合 reset observation。

## 训练组件

| 参数 | 默认值 | 对照含义 |
|---|---|---|
| `--backbone-mode` | `shared` | `separate` 为独立骨干；`encoder_decoder` 为两路投影、价值 Encoder、策略 Decoder |
| `--learning-rate` | `3e-4` | 初始 Adam 学习率 |
| `--lr-schedule` | `constant` | `linear` 线性插值到 `--learning-rate-final`，默认终值 0 |
| `--normalize-advantage` | 开启 | `--no-normalize-advantage` 使用原始 GAE 优势 |
| `--value-clip-coef` | `0`，关闭 | 正值开启相对旧价值的损失裁剪；FO06 使用 0.2 |
| `--ent-coef` | `0.01` | 初始熵系数 |
| `--ent-coef-schedule` | `constant` | `linear` 插值到 `--ent-coef-final`，默认终值 0 |
| `--target-kl` | `0.02` | 更新前 minibatch KL 大于其 1.5 倍即拒绝并停止本轮；0 关闭 |
| `--diagnostic-interval` | `50` | 首轮、每 50 轮和末轮采集重诊断；0 完全关闭 |

固定默认模型为 D256、4 层、4 heads、每层窗口 32、残差门控初值 1e-4。保守历史预热长度为 `1 + layers × (context_length - 1)`。默认 8 环境 ×256 步，学习序列 64，每 minibatch 4 条序列、4 epochs；gamma=.99、GAE lambda=.95、policy clip=.2、value coefficient=.5、梯度裁剪 .5、Adam eps=1e-5，无权重衰减或奖励归一化。

只执行完整 rollout，实际步数为 `floor(requested / rollout_size) × rollout_size`。命令行默认请求 500 万步，实验手册显式指定 1000/5000 万步。线性调度按实际更新数推进，首轮取初值、末轮取终值；只有一轮时取初值。KL 拒绝不改变进度，因此最后一轮线性学习率可为 0，optimizer step 次数仍按真实调用记录。

优势标准化作用于整个 rollout 的策略优势，不改变原始 GAE 或价值目标。价值损失采用 `0.5 × mean(max((V-R)^2, (V_old+clip(V-V_old)-R)^2))`；关闭时为半 MSE。0.2 是原始价值单位的待验证设置，不保证改善训练。

独立骨干初值从策略分支复制，参数存储独立，不额外消耗初始化 RNG。同宽度默认参数总数：共享 3,219,981，独立 6,438,677；这是同宽度而非等总参数量对照。仍使用一个 Adam 和全局梯度裁剪；不能将结果直接归因于彻底隔离所有优化联系。

## 观测方案与课程

| 设置 | 行为 |
|---|---|
| `--sensor-mode full` | 全程 p=0 |
| `--sensor-mode partial` | 全程 p=1 |
| `--sensor-mode curriculum --curriculum-schedule fixed` | 默认按步数退化 |
| `--sensor-mode curriculum --curriculum-schedule performance` | 根据独立评估逐级推进 |

固定课程在前 10% 实际步数保持 p=0，接下来 70% 线性增加，最后 20% 保持 p=1；比例由 `--curriculum-start-fraction` 和 `--curriculum-end-fraction` 控制。

成绩课程从 p=0 开始，默认连续两次评估同时满足均分 ≥200 和回报 ≥200 的比例 ≥80%，才增加 p=.1；失败清空连续计数，不回退，预算结束不强制推进。`--eval-interval 0` 关闭周期/结束评估，但成绩课程不允许关闭。

## 批量独立评估

训练默认每 **100 次 PPO 更新**评估 **20 回合**，不是每 100 个训练回合。可使用 `--eval-interval 50` 加密观察。默认 `--eval-num-envs 8`，独立 CLI 对应 `--num-envs 8`。视频与窗口模式实际使用 1 个环境。

评估器复用独立的 eager 策略副本和环境池，只运行 Encoder、策略 Transformer、Actor；独立骨干不创建价值网络或价值缓存。每个向量步合批推理并批量回传概率，在 CPU 抽样和累计动作统计。完成一个回合后补入下一个指定回合；尾部空槽不计入统计，所有指定回合都完成后才结束。

每个回合以 `(seed + episode) mod 2^32` 重置环境，传感器和动作使用独立的回合随机流。协议为 `batched-policy-v2`；它不复现旧全局 Torch 采样的逐步轨迹。不同 batch size、设备的浮点舍入也可能导致轨迹差异，正式对照需统一评估并发数、设备与协议。

周期评估使用 seed=10000、sample 策略；训练期间同步等待结果，再进行课程判定。评估不推进训练环境、缓存、模型模式或训练 RNG。留出报告使用 seed=20000 和 sample/argmax 两种策略。均分、标准差和 **≥200 分比例**按回合统计，动作比例与熵按实际环境步数统计。

CLI 默认仍为 argmax、p=1；评估全观测任务须显式指定 `--sensor-failure-probability 0`。终止奖励 -100 可能包含不同失败原因，日志称为 `terminal_penalty`；不会仅凭这个标志宣称坠毁。超时单独统计。

## 诊断、产物与复现

```text
<run>/metadata.json           配置/源码指纹、依赖、硬件、预算、状态、评估时间占比
<run>/training.csv            调度值、PPO 指标、原始/实际优势、价值/目标/奖励统计
<run>/diagnostics.csv         周期梯度范数、共享梯度夹角、参数相对变化
<run>/episodes.csv            回报、长度、终止/截断
<run>/evaluation.csv          周期评估成绩、耗时、步数和吞吐
<run>/evaluation_episodes.csv 周期评估逐回合结果与最终状态
<run>/curriculum_transitions.csv  有阶段转换时创建
<run>/tensorboard/            标量日志，横轴累计训练环境步数
<run>/checkpoints/latest.pt   周期覆盖及最终权重
<run>/checkpoints/best_*.pt   各可比较观测条件的最佳周期评估权重
<run>/checkpoints/transition_stage_*.pt  成绩课程转换权重
<run>/diagnostics/*.pt        首次一致性警告和硬失败现场
<evaluate-run>/evaluation.json  留出结果、训练来源和耗时
```

`accepted_sample_visits` 计数已接受 minibatch 中真实 loss token 的访问次数，允许多 epoch 重复；`sample_reuse_count` 为它除以 rollout 样本数；`unique_sample_coverage` 表示至少参与过一次接受更新的唯一样本比例。预热和 padding 不计入。KL 停止可能导致部分数据完全未更新。

梯度诊断采用固定的第一批学习序列，在更新前运行 eager 前向及 `autograd.grad`，不消费 minibatch shuffle RNG、不写 `.grad`。策略目标包含当前熵项，价值目标包含 value coefficient。共享梯度夹角只在共享参数上定义；独立骨干为空，零范数也为空。参数变化在本轮更新后测量。该探针只代表选定批次，不代表完整 rollout 的平均梯度。

所有原始/实际优势、价值、目标及奖励统计均记录均值、总体标准差、最小和最大值。诊断开销单独记录在 `diagnostic_seconds`，同时包含于 PPO 总耗时。

源码指纹覆盖本项目顶层 Python 文件及共享组件源码；配置指纹覆盖完整传入配置。checkpoint 保持 v2，增加来源字段；旧兼容 v2 权重仍可加载，但缺乏新来源证据的旧日志不进入正式汇总。依赖元数据缺失记录为 null，正式实验仍须通过手册的环境预检。

训练默认 `--compile auto`，仅在支持的 Linux/CUDA 环境预热成功后启用。FP32、math SDPA、TF32/AMP 关闭；预热阶段允许 auto 回退，训练开始后的异常停止，不重试已经修改缓存的步骤。更新前 log-prob 默认超过 1e-4 警告、5e-4 停止；行为旧概率不会被覆盖。

`latest.pt` 是权重与配置快照，不包含恢复完整训练所需的优化器和环境状态。5000 万步实验从头训练。运行输出统一放在 Git 忽略的 `runs/`，不自动备份或归档。

参考：[Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)、[PPO](https://arxiv.org/abs/1707.06347)、[PPO 实现细节](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)、[共享 Transformer](../shared/README.md)。

## Stage2：固定 XY Fourier 实验

保留原有 20 维输入，追加 x 四档、y 两档确定性 sin/cos，共 32 维；默认关闭，独立运行 `XY_PO01/XY_PO03` 各三个种子。
频率依据、相位监测及历史对照边界见 [实验设计](reports/XY-PO01-PO03-design.md)。
在仓库根目录 PowerShell 一键启动（正式批次 `xy-fourier-10m-v1`，6 次训练＋36 次留出评估）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_xy_fourier.ps1
```

附加 `-DryRun` 查看任务，`-Smoke` 跑独立短程烟测，`-Status` 查进度；Ctrl+C 后原命令可补跑。
结果位于 `runs/experiments/xy-fourier-10m-v1/summary/overview.md`，历史对照见同目录的 `xy-comparison.md`。
直接训练也可使用 `train.py --backbone-mode separate --xy-fourier`；评估自动读取 checkpoint 中的编码配置。
