# 分阶段实验命令手册

Windows 第一阶段优先使用[统一入口](README.md#windows-统一实验入口)：默认只做 FO01 三种子、留出评估和统计，支持检测中断并补跑。其他 FO 组件通过 `-Groups` 显式选择。第一阶段七组完成后可使用[第二阶段一键入口](README.md#第二阶段一键入口)，自动校验并冻结 FO04、运行三种 PO 方案；5000 万步长跑仍需另行启动。

以下命令在 Linux 容器的 **`/workspace/lunarlander-ppo-pomdp`** 执行，可直接从 `root@容器:/workspace/lunarlander-ppo-pomdp#` 开始。使用 Bash；同一阶段的公共参数、函数和执行循环放在同一个终端运行。这里没有自动启动的实验任务。

以下手工命令的训练/评估目录存在时拒绝覆盖；手工重跑应更换整个批次名，不能把同条件的旧失败记录和新记录混在一个正式汇总里。统一入口另行管理 `runs/experiments/` 下的 attempt 和精确汇总清单，可以在相同配置下补跑失败任务；请勿对该目录使用递归扫描全部历史 attempt 的手工汇总命令。

## 1. 环境、测试与烟测

```bash
set -euo pipefail
cd /workspace/lunarlander-ppo-pomdp
export TORCHINDUCTOR_COMPILE_THREADS=2
python -m pip install --no-index --no-deps --no-build-isolation -e ../shared
python -m pip check
python - <<'PY'
import importlib.metadata as m
import torch
import gymnasium as gym
assert str(torch.__version__).split('+')[0] == '2.9.1', torch.__version__
assert gym.__version__ == '1.2.2', gym.__version__
assert torch.cuda.is_available(), 'CUDA unavailable'
print(torch.__version__, torch.cuda.get_device_name(), m.version('box2d-py'))
env = gym.make('LunarLander-v3')
try:
    env.reset(seed=42)
    env.step(0)
finally:
    env.close()
PY
```

只有 CPU 时改用 `--device cpu --compile off`，并跳过 CUDA 断言；正式同一批次统一设备与依赖。新镜像的启动方式见[根 README](../README.md)。

```bash
validation_root="runs/checks-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$validation_root"
python -B -m unittest discover -s tests -v 2>&1 | tee "$validation_root/project-tests.log"
python -B -m unittest discover -s ../shared/tests -v 2>&1 | tee "$validation_root/shared-tests.log"
python -B train.py --help
python -B evaluate.py --help
python -B summarize_experiments.py --help

for backbone in shared separate; do
  python -B train.py --device cuda --compile on --seed 42 \
    --num-envs 2 --num-steps 16 --total-timesteps 64 \
    --hidden-dim 32 --num-layers 2 --num-heads 2 --context-length 4 \
    --sequence-length 6 --sequences-per-minibatch 3 --update-epochs 2 \
    --max-episode-steps 16 --backbone-mode "$backbone" \
    --lr-schedule linear --ent-coef-schedule linear --value-clip-coef 0.2 \
    --no-normalize-advantage --diagnostic-interval 1 \
    --eval-interval 1 --eval-episodes 2 --eval-num-envs 8 \
    --experiment-id "SMOKE-$backbone" --run-dir "$validation_root/$backbone"
done
```

短回合仅检查实现、文件、截断和课程/评估链路，不代表标准任务成绩。CUDA 测试 skip 不等于 CUDA 通过。实际执行证据见 [VALIDATION.md](VALIDATION.md)。

## 2. 第一阶段：全观测组件对照

先回答“基础训练是否稳定、组件是否有收益”。所有组从头训练，seed=42、43、44，每次请求 10,000,000 步，默认 rollout 下实际 9,998,336 步。评估每 100 次 PPO 更新进行 20 回合，sample、seed=10000、p=0、8 环境；50 轮评估适合另开统一设置的批次，不能只改某一组。

| ID | 共享/独立骨干 | 学习率 | 优势标准化 | 价值裁剪 | 熵系数 |
|---|---|---|---|---|---|
| FO01 | shared | 固定 | 开 | 关 | 固定 |
| FO02 | shared | 线性到 0 | 开 | 关 | 固定 |
| FO03 | separate | 固定 | 开 | 关 | 固定 |
| FO04 | separate | 线性到 0 | 开 | 关 | 固定 |
| FO05 | shared | 固定 | 关 | 关 | 固定 |
| FO06 | shared | 固定 | 开 | 0.2 | 固定 |
| FO07 | shared | 固定 | 开 | 关 | 线性到 0 |

核心四组比较架构与学习率的组合，另外三组相对 FO01 只改一个组件。共 21 次训练，约 2.1 亿步；不运行全部组件的笛卡尔积。

先定义本阶段公共参数和函数：

```bash
fo_root='runs/fo-10m-v1'
common=(
  --device cuda --compile auto --total-timesteps 10000000
  --hidden-dim 256 --num-layers 4 --num-heads 4 --context-length 32 --alpha-init 1e-4
  --backbone-mode shared --previous-action
  --num-envs 8 --num-steps 256 --sequence-length 64 --sequences-per-minibatch 4 --update-epochs 4
  --learning-rate 3e-4 --lr-schedule constant --learning-rate-final 0
  --gamma 0.99 --gae-lambda 0.95 --clip-coef 0.2 --vf-coef 0.5
  --ent-coef 0.01 --ent-coef-schedule constant --ent-coef-final 0
  --normalize-advantage --value-clip-coef 0 --max-grad-norm 0.5 --target-kl 0.02
  --consistency-warning-tolerance 1e-4 --consistency-tolerance 5e-4
  --max-episode-steps 1000 --save-interval 10 --diagnostic-interval 50
  --sensor-mode full --curriculum-schedule fixed
  --eval-interval 100 --eval-episodes 20 --eval-num-envs 8 --eval-seed 10000 --eval-policy sample
)

run_fo() {
  local condition="$1" seed run log
  local -a component=()
  case "$condition" in
    FO01) ;;
    FO02) component=(--lr-schedule linear) ;;
    FO03) component=(--backbone-mode separate) ;;
    FO04) component=(--backbone-mode separate --lr-schedule linear) ;;
    FO05) component=(--no-normalize-advantage) ;;
    FO06) component=(--value-clip-coef 0.2) ;;
    FO07) component=(--ent-coef-schedule linear) ;;
    *) echo "Unknown condition: $condition" >&2; return 1 ;;
  esac
  for seed in 42 43 44; do
    run="$fo_root/$condition/seed$seed/train"
    log="$fo_root/console/$condition-seed$seed.log"
    if [[ -e "$run" || -e "$log" ]]; then
      echo "Refusing existing output: $run / $log" >&2; return 1
    fi
    mkdir -p "$fo_root/console"
    python -B train.py "${common[@]}" "${component[@]}" --seed "$seed" \
      --experiment-id "$condition" --run-dir "$run" 2>&1 | tee "$log"
  done
}
```

先顺序执行核心四组：

```bash
for condition in FO01 FO02 FO03 FO04; do run_fo "$condition"; done
```

之后执行附加三组；在同一终端中运行，或重新定义上面的公共参数和函数：

```bash
for condition in FO05 FO06 FO07; do run_fo "$condition"; done
```

分析与冻结：

```bash
fo_root='runs/fo-10m-v1'
python -B summarize_experiments.py --root "$fo_root" --output-dir "$fo_root/summary" \
  --expected-seeds 42 43 44 --conditions FO01 FO02 FO03 FO04 FO05 FO06 FO07
cat "$fo_root/summary/summary.md"
cat "$fo_root/summary/selection.json"
```

所有七组和三个种子必须完整、配置/源码/预算一致，才冻结候选。候选至少两个种子在最后连续两次周期评估中同时满足均分 ≥200、200 分比例 ≥80%。合格组按最后五次评估的跨种子平均回报排序，再比较该窗口的达标比例，最后按 ID 排序。

若 `eligible=false`，先查看 `issues`、缺失条件及诊断，不自动放宽门槛、混入不同预算或用留出种子挑配置。仍可如实报告这批训练未解决任务。

```bash
python - "$fo_root/summary/selection.json" <<'PY'
import json, sys
selection = json.load(open(sys.argv[1]))
assert selection['eligible'], selection['reason'] if not selection['eligible'] else ''
print('Freeze:', selection['experiment_id'])
PY
test ! -e "$fo_root/frozen-recipe.json"
cp "$fo_root/summary/selection.json" "$fo_root/frozen-recipe.json"
```

被冻结的是一个完整且实际训练过的配置，不把多个组的“有益组件”直接拼成未经验证的新配置。独立骨干参数更多，需要同时报告参数量、实际步数、耗时和更新覆盖率。

## 3. 第二阶段：固定配置比较课程

Windows 日常使用 `powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_stage2.ps1`。默认读取受管理的 `runs/experiments/baseline-v1`，在独立的 `runs/experiments/course-10m-v1` 内冻结配置、新增 9 次训练和 54 次留出评估，并引用 FO04 已有成绩。选项和补跑方式见 [README](README.md#第二阶段一键入口)。下面保留的是手工容器路径流程，不要用其递归汇总或手工命令写入受管理批次。

先完成相同的 1000 万步比较。沿用被冻结的模型、PPO、诊断与评估参数，只改变观测方案；三种子均从随机初始化开始。全观测直接引用第一阶段所选组，节省三次重复训练。

| ID | 观测方案 |
|---|---|
| 第一阶段所选 FOxx | 全观测 p=0；复用对应三条运行 |
| PO01 | 从头训练 p=1 |
| PO02 | 固定课程，前 10% p=0、接下来 70% 退化、最后 20% p=1 |
| PO03 | 成绩课程，步长 .1、门槛 200/80%、连续两次达标 |

本阶段新训练 9 次，约 9000 万步。先定义冻结参数和课程函数：

```bash
fo_root='runs/fo-10m-v1'
recipe="$fo_root/frozen-recipe.json"
python - "$recipe" <<'PY'
import json, sys
from runtime import source_fingerprint
selection = json.load(open(sys.argv[1]))
assert selection['eligible']
assert selection['source_fingerprint'] == source_fingerprint(), 'Source changed: start a new controlled batch'
PY
mapfile -t reference_args < <(python - "$recipe" <<'PY'
import json, sys
print('\n'.join(json.load(open(sys.argv[1]))['arguments']))
PY
)
selected=$(python - "$recipe" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))['experiment_id'])
PY
)

run_course() {
  local batch="$1" budget="$2" condition="$3" seed run log
  local -a observation=()
  case "$condition" in
    PO00) observation=(--sensor-mode full --curriculum-schedule fixed) ;;
    PO01) observation=(--sensor-mode partial --curriculum-schedule fixed) ;;
    PO02) observation=(--sensor-mode curriculum --curriculum-schedule fixed
                       --curriculum-start-fraction 0.1 --curriculum-end-fraction 0.8) ;;
    PO03) observation=(--sensor-mode curriculum --curriculum-schedule performance) ;;
    *) echo "Unknown condition: $condition" >&2; return 1 ;;
  esac
  for seed in 42 43 44; do
    run="$batch/$condition/seed$seed/train"
    log="$batch/console/$condition-seed$seed.log"
    if [[ -e "$run" || -e "$log" ]]; then
      echo "Refusing existing output: $run / $log" >&2; return 1
    fi
    mkdir -p "$batch/console"
    python -B train.py "${reference_args[@]}" "${observation[@]}" \
      --total-timesteps "$budget" --seed "$seed" --experiment-id "$condition" \
      --run-dir "$run" 2>&1 | tee "$log"
  done
}
```

执行 1000 万步批次：

```bash
course_root='runs/course-10m-v1'
for condition in PO01 PO02 PO03; do run_course "$course_root" 10000000 "$condition"; done
python -B summarize_experiments.py --root "$fo_root/$selected" --root "$course_root" \
  --output-dir "$course_root/summary" --conditions "$selected" PO01 PO02 PO03
```

跨课程不能拿最后几次的不同 p 分数直接排高低，应使用下面统一 p 的冻结权重评估。记录 `entered_final`、`final_task_solved`、当前 p 和各阶段步数；进程完成不等于任务达标。

## 4. 留出评估与独立长跑

先冻结配置再看留出结果，不再用 seed=20000 回头挑组件。统一使用 `latest.pt`，100 回合、seed=20000、8 环境，p=0/.5/1，sample 和 argmax 分开报告。原始奖励、最大回合长度继承 checkpoint。

```bash
heldout_condition() {
  local train_root="$1" eval_root="$2" condition="$3" seed probability policy checkpoint run log
  for seed in 42 43 44; do
    checkpoint="$train_root/$condition/seed$seed/train/checkpoints/latest.pt"
    test -f "$checkpoint"
    for probability in 0 0.5 1; do
      for policy in sample argmax; do
        run="$eval_root/$condition/seed$seed/p${probability}-${policy}"
        log="$eval_root/console/$condition-seed$seed-p${probability}-${policy}.log"
        if [[ -e "$run" || -e "$log" ]]; then
          echo "Refusing existing output: $run / $log" >&2; return 1
        fi
        mkdir -p "$eval_root/console"
        python -B evaluate.py --checkpoint "$checkpoint" --device cuda \
          --episodes 100 --num-envs 8 --seed 20000 --policy "$policy" \
          --sensor-failure-probability "$probability" --experiment-id "$condition" \
          --run-dir "$run" 2>&1 | tee "$log"
      done
    done
  done
}

heldout_root='runs/heldout-10m-v1'
for condition in FO01 FO02 FO03 FO04 FO05 FO06 FO07; do
  heldout_condition "$fo_root" "$heldout_root" "$condition"
done
for condition in PO01 PO02 PO03; do
  heldout_condition "$course_root" "$heldout_root" "$condition"
done
python -B summarize_experiments.py --root "$heldout_root" --output-dir "$heldout_root/summary"
```

FO 组件比较以 p=0 为主；最终部分观测目标比较以 p=1 为主，p=.5 表示中间难度泛化。每组报告三个训练种子的均值和样本标准差；100 个评估回合不是 100 个独立训练种子。3 个训练种子不支持强显著性结论。

1000 万步报告完成后，可独立启动 5000 万步批次。重新进入终端时先重新定义上一节的 `reference_args`、`run_course` 和本节 `heldout_condition`。所有四组重新训练，不从 1000 万步 checkpoint 续接；学习率和固定课程都依赖总预算。实际每次 49,999,872 步，12 次约 6 亿步。

```bash
long_root='runs/course-50m-v1'
for condition in PO00 PO01 PO02 PO03; do run_course "$long_root" 50000000 "$condition"; done
python -B summarize_experiments.py --root "$long_root" --output-dir "$long_root/summary" \
  --conditions PO00 PO01 PO02 PO03

long_heldout='runs/heldout-50m-v1'
for condition in PO00 PO01 PO02 PO03; do heldout_condition "$long_root" "$long_heldout" "$condition"; done
python -B summarize_experiments.py --root "$long_heldout" --output-dir "$long_heldout/summary"
```

1000 万和 5000 万步分别汇总，不能混成一个条件。若更改组件、种子、并发数、课程门槛或代码，另起批次并明确研究问题。

## 5. 观察、性能和存储

训练时重点看：

- 周期独立评估的均分、≥200 分比例、超时比例与当前 p。
- 原始优势/价值目标是否突然放大、价值 explained variance 是否恶化。
- 主引擎概率、四动作比例和熵是否塌缩。
- KL 停止、实际 optimizer step、`sample_reuse_count`、`unique_sample_coverage`。
- 策略/价值梯度范数、共享梯度夹角、相对参数变化，以及评估时间占比。

重诊断的第一批序列不代表整个 rollout；独立骨干共享梯度夹角为空。`grad_norm` 是裁剪前范数，KL 拒绝发生在 step 前；若没有接受任何 minibatch，损失均值为 0，应结合拒绝标志和最大 KL 判断。首次 log-prob 警告与失败现场仍保留，可用 `diagnose_consistency.py --help` 查看离线回放入口。

真实 Box2D 评估性能对照，不启动训练：

```bash
benchmark_root="runs/eval-benchmark-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$benchmark_root"
python -B benchmark_evaluation.py --device cuda --episodes 20 100 --repeats 3 --num-envs 8 \
  --output "$benchmark_root/results.json" 2>&1 | tee "$benchmark_root/console.log"
```

基准固定未训练的 D256 模型，只测执行速度。旧路径使用全局 Torch 动作 RNG，新路径使用回合独立 RNG，因此 sample 路径步数可能不同；同时报告端到端耗时和实际 steps/s。首轮包含模型副本创建，后续复用；不要把随机策略的成绩视作收敛证据。正式长回合的实际耗时还与回合长度和 CPU 物理步进有关。

在容器中临时查看新数据可运行 `tensorboard --logdir runs --host 0.0.0.0 --port 6006`；外部浏览器访问需已映射端口。仓库自带 TensorBoard 服务的启动方式见根 README。

所有运行都在忽略目录 `runs/` 下。`latest.pt` 周期覆盖，最佳/转换权重按阶段保存，不再备份整棵 runs。`diagnostics.csv` 只保存标量，完整现场仅首次警告/失败各一份。结束后先用汇总检查状态和目录，再手动删除明确不需要的整个批次；不要把归档复制当作默认步骤。

## 6. Encoder–Decoder 探索实验

独立配置为 `encoder_decoder_experiments.json`，Windows 入口为 `run_encoder_decoder.ps1`，完整参数见 [README](README.md#encoderdecoder-独立探索实验)。在仓库根目录先执行带 `-DryRun` 的入口可预览正式任务；`-Smoke` 在独立批次运行短程链路。去掉这两个参数才会启动正式探索实验，代码验收不自动执行该长跑。

| 实验组 | 模型 | 观测方案 | 训练种子 / 请求预算 |
|---|---|---|---|
| ED_PO01 | D256 / Encoder 4 层 / Decoder 4 层 / W32 | 直接 p=1 | 42、43、44，各 1000 万步 |
| ED_PO03 | 同上 | 成绩课程，p 步长 .1、门槛 200/80%、连续两次达标 | 同上 |

其余训练超参数与冻结 FO04 相同，实际每个种子完成 9,998,336 步。默认 6 次新训练、36 项新评估，评估最终 `latest.pt`，p=0/.5/1 × sample/argmax、100 回合、8 环境、seed=20000。保留原 PO01/PO03 的独立历史报告作参考；三个训练种子、已经使用过的评估种子集只支持探索性结论。

Encoder 与 Decoder 各用独立输入投影，价值头读取 H，Actor 经可微交叉注意力读取历史。相对 FO04，参数量、策略感受野与共享梯度均改变，因此不把成绩变化单独解释为 Cross-Attention 的效果。单层窗口为 32，价值、策略最长原始输入感受野为 125/249；学习序列 64 加保守预热前缀 249，最长输入为 313 token。正式实验保持 FP32/math SDPA 和现有 compile auto 口径，融合后端只在独立验证中检查。

先看 `summary/overview.md` 与最终统一条件成绩；`summary/curriculum.csv` 另存最终概率、阶段步数、是否进入 p=1、p=1 完整 rollout 的训练步数、首次实际训练起始位置及当时学习率。共享梯度夹角、Encoder 目标梯度、交叉注意力梯度和门控来自固定诊断批次，应结合运行成本与各训练种子分别解释。

例如在容器的本项目目录中手动启动新批次：

```bash
python -B run_experiments.py --config encoder_decoder_experiments.json \
  --batch encoder-decoder-10m-v1
```

生产形状的短程编译验收可单独复现，每次使用尚不存在的输出目录，仅运行 4096 环境步及少量评估：

```bash
python -B tests/smoke_encoder_decoder_cuda.py \
  --output-dir runs/encoder-decoder-cuda-shape-v2
```

## 固定 XY Fourier 输入实验

`xy_fourier_experiments.json` 从冻结 Stage2 的 FO04 配方复制；只启用 XY 编码，仍为独立 Actor/Critic。
`XY_PO01` 直接 p=1，`XY_PO03` 使用原成绩课程；种子 42/43/44，正式批次 `xy-fourier-10m-v1`。
x 的角频率为 π、2π、4π、8π，y 为 π、2π，保留原始 20 维后追加 12 维。
频段依据、适用边界、日志字段和复现方法见 [XY 实验设计](reports/XY-PO01-PO03-design.md)。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\lunarlander-ppo-pomdp\run_xy_fourier.ps1
```

入口支持 `-Smoke`、`-DryRun`、`-Status`、`-AnalyzeOnly`、`-Groups` 和 `-Batch`。
正式队列共 6 次训练、36 次统一留出评估；历史 PO01/PO03 按各自来源校验，
仅作为探索性参考，见新批次 `summary/xy-comparison.md`。不重训基线，也不修改旧批次。
