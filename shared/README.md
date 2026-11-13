# rl_shared：RL 时序 Transformer 公共组件

这是独立安装的轻量组件包，仅依赖 PyTorch。输入、输出都是 `(Batch, Seq_Len, Hidden_Dim)` 连续特征，提供单堆栈 Transformer 和时序 Encoder–Decoder。原始观测投影、Actor、Critic、RND、损失函数和优化器由调用方管理。

它在 RL 中承担“记忆”的角色：当前观测不足以判断状态时，策略可以通过注意力读取过去的线索。外部 Encoder 先编码观测，骨干整合历史，再由外部 Actor/Critic 输出动作分布和价值。能否学会利用历史，需要学习测试验证；仅检查输出形状和有限梯度还不够。

## 时序 Encoder–Decoder

`EncoderDecoderBackbone` 接收两路**独立投影后的、时间对齐的特征**，返回 Encoder 历史 `H` 和 Decoder 特征 `Z`。默认 Encoder/Decoder 各 4 层；`num_layers` 表示 Encoder 深度，`decoder_layers` 单独控制 Decoder 深度。两路共用宽度、head 数、每次注意力的窗口和 episode 内 RoPE 位置。

```python
import torch
from rl_shared import EncoderDecoderBackbone, EncoderDecoderConfig

model = EncoderDecoderBackbone(EncoderDecoderConfig(
    hidden_dim=128, num_layers=4, decoder_layers=4, context_length=32,
))
encoder_inputs = torch.randn(8, 64, 128)  # 外部独立投影 A
decoder_inputs = torch.randn(8, 64, 128)  # 外部独立投影 B
history, policy_features = model(encoder_inputs, decoder_inputs)
# critic(history), actor(policy_features) 由调用方定义

cache = model.create_cache(8)
h_t, z_t = model.step(encoder_inputs[:, :1], decoder_inputs[:, :1], cache)
```

并行 `forward(encoder_inputs, decoder_inputs, *, episode_starts=None, valid_mask=None, position_ids=None)` 的输入均为 `(B,T,D)`，输出 `(H,Z)` 的形状相同。两路时间轴必须一致；元数据的类型、padding、位置和 episode 语义与下文的单堆栈接口相同。此接口始终因果，不提供取消因果约束的选项。

每个 Decoder 层依次执行 Pre-RMSNorm 自注意力、交叉注意力、SwiGLU；每个子层都有小的可学习残差门控。交叉注意力分别归一化 Query 输入与 Encoder memory，用独立的 Q、融合 KV、输出投影，Q/K RMSNorm 后使用同一时间坐标的 RoPE。所有层读取最终的 `H`，**不 detach**，序列训练的策略梯度能够通过历史位置更新 Encoder。

三类注意力均只读取同一 episode 的有效 token，且 `0 <= t-s < W`，包括当前 `H_t`。padding 输入先清零，两路 padding 输出也清零。由于层间传递，Encoder/Critic 的原始输入感受野为 `1 + E*(W-1)`，策略最长为 `1 + (E+D)*(W-1)`；这里的窗口指每次注意力。4+4 层、W32 对应 125/249 个 token。配置的 `value_warmup_length`、`warmup_length` 按这两个公式给出保守重放长度。

`EncoderDecoderCache` 用一个预分配环形存储统一保存三组 K/V，层轴顺序为 Encoder self、Decoder self、Decoder cross。`encoder_keys/values`、`decoder_keys/values`、`cross_keys/values` 是只读检查用的视图；`keys/values` 为完整存储。各 Decoder 层的 memory 投影独立，每步只对新 `H_t` 做一次投影。共享 `lengths/next_positions` 每步只推进一次，`reset(mask)` 同时使三组缓存的对应环境失效。`tensors()` 返回全部可变状态，`nbytes` 统计存储与元数据。

`step(..., cache, episode_starts=None, enable_grad=False)` 与下文单堆栈相同：默认无梯度，历史缓存始终脱离计算图；启用当前步梯度时，当前 `H_t` 的交叉注意力路径保留梯度，并通过快照支持延迟 backward。PPO 训练应使用 `forward` 重算包含预热前缀的完整计算图，损失只选学习区。参数更新后从原始输入重建三组缓存，不复用旧参数产生的 memory KV。

通用组件使用 PyTorch SDPA 自动分派；调用方可用 `sdpa_kernel` 选择后端。LunarLander 正式对比维持 FP32/math SDPA，单独的 CUDA 测试强制 efficient attention 验证融合前向、反向及缓存一致性。

## 安装和运行

在仓库根目录、现有 Docker 镜像中安装，无需下载新依赖或修改 `PYTHONPATH`：

```bash
docker compose run --rm -T runner bash -lc \
  'python -m pip install --no-index --no-deps --no-build-isolation -e ./shared && python shared/examples/backbone_usage.py'
```

开发容器中直接执行 `python -m pip install --no-index --no-deps --no-build-isolation -e ./shared`。安装后可从任意实验目录 `import rl_shared`。一次性容器退出后需重新安装；源码通过 editable 安装实时生效。

```python
import torch
from torch import nn
from rl_shared import TransformerConfig, TransformerBackbone, init_head

model = TransformerBackbone(TransformerConfig(hidden_dim=128))
actor = init_head(nn.Linear(128, 6), gain=0.01)
critic = init_head(nn.Linear(128, 1), gain=1.0)

features = torch.randn(8, 64, 128)  # 外部 CNN/MLP 的输出
hidden = model(features)           # (8, 64, 128)，默认保留梯度
logits, values = actor(hidden), critic(hidden)

cache = model.create_cache(batch_size=8)
token = torch.randn(8, 1, 128)
with torch.no_grad():              # 若需要，也要关闭外部 Encoder/Heads 的梯度
    current = model.step(token, cache)
    action_logits = actor(current[:, 0])
```

## 架构和配置

每层为 `x += alpha_attn * Attention(RMSNorm(x))`，再执行 `x += alpha_ffn * SwiGLU(RMSNorm(x))`。融合 QKV 和 SwiGLU gate/up 投影；Linear 无 bias、无 Dropout。这样减少额外随机性，尤其避免 PPO 行为概率因 Dropout 重采样产生差异。训练、交互模式由 `forward` / `step` 明确选择，不随 `train()` / `eval()` 切换。

| 配置 | 默认值 | 含义 |
|---|---|---|
| `hidden_dim` | 必填 | 必须能被 heads 整除，且 head_dim 为偶数 |
| `num_layers`, `num_heads` | 2, 4 | 层数和注意力头数 |
| `context_length` | 256 | 每层窗口 W，包含当前 token |
| `ffn_dim` | `ceil((8D/3)/64)*64` | SwiGLU 中间宽度；D=128 时为 384 |
| `alpha_init` | 1e-4 | 两个独立可学习标量门控的初值，可为 0 |
| `qk_norm` | True | 每个 head 的 Q/K 无仿射 RMSNorm，位于 RoPE 之前 |
| `final_norm` | False | 末尾 RMSNorm；关闭时零门控的整个骨干为恒等映射 |
| `rms_eps`, `rope_theta` | 1e-6, 10000 | 归一化 epsilon 与 RoPE 频率基数 |

RMS 的半精度输入统计使用 FP32；RoPE 的相位和频率始终用 FP32 计算，且不缓存随 episode 长度增长的大表。整数频率索引在 `model.half()` / `bfloat16()` 后仍保持完整精度。骨干 Linear 使用 Xavier uniform 初始化。小门控在首次更新即可让分支获得梯度；门控为零时，首次更新通常仅打开门控，分支权重的首步梯度为零。

QK-RMSNorm 保留标准 SDPA 的 `1/sqrt(head_dim)` 缩放，是 QK 归一化的一种变体；不同于原 QKNorm 论文的 L2 归一化与可学习温度。这里实现稳定性机制，测试不等同于任何具体 RL 任务的收敛证据。

当前保留这组组件，理由如下：

| 组件 | 保留理由 |
|---|---|
| Pre-RMSNorm | 分支先归一化，保留直接的残差通路；半精度统计使用 FP32 |
| Q/K 归一化 | 减少注意力分数对 Q/K 模长变化的敏感度 |
| 小标量残差门控 | 每层只增加两个参数，初始接近恒等映射，再逐渐学习使用历史 |
| RoPE | 提供时序位置信息，支持窗口回绕和 episode 重置，无需不断增长的位置表 |
| SwiGLU | 当前未发现需要更换的稳定性问题，保留现有权重结构 |
| 无 Dropout、显式缓存 | 减少采样时的额外随机性，使 episode 生命周期由调用方明确控制 |

标量残差门控参考 [ReZero](https://arxiv.org/abs/2003.04887)；接近恒等映射的初始化在 [GTrXL](https://proceedings.mlr.press/v119/parisotto20a.html) 中也有 RL 实验依据。这里的组件组合和短记忆测试不能替代具体 RL 算法、环境的收敛验证。

## 并行训练：位置、episode 和掩码

```python
hidden = model(
    features,
    episode_starts=starts,  # 可选 bool (B,T)：当前 token 是新 episode 第一个观测
    valid_mask=valid,      # 可选 bool (B,T)：True 有效，右侧 padding 设 False
    position_ids=positions,# 可选 int32/int64 (B,T)：显式非负 RoPE 位置
    attn_mask=mask,        # 可选 bool 许可掩码或浮点加性 bias
    is_causal=True,
)
```

所有张量与输入在同一设备；省略的位置从 0 开始，遇到 `episode_starts=True` 重新从 0 开始。显式 `position_ids` 优先控制 RoPE；episode 隔离仍由 `episode_starts` 决定。窗口和因果关系按序列顺序计算，不依赖自定义位置的数值。有效输入应是有限连续特征；右侧 padding 输入会先被置零，padding 输出也置零，训练损失仍应排除这些位置。

默认每层仅允许同 episode、有效键、过去 W-1 步和自身。`is_causal=False` 取消因果与窗口限制，但保留 episode、padding、自定义掩码。显式 masks 和内部规则取交集；浮点 bias 只对允许位置生效，应为有限值或 `-inf`，不传 `NaN/+inf`。

`attn_mask` 形状为 `(T,T)` 或 `(B,1/H,T,T)`，batch 维也可为 1。布尔 `True` 表示允许注意力，与某些 `key_padding_mask` 的含义相反。浮点 mask 先合并内部限制，在 FP32 中对每行减去允许位置的最大 bias，再转换到投影 dtype；源 mask 或投影为 FP64 时使用 FP64。逐行减去常量保持注意力的相对权重，同时避免大偏移（如 `100000`）直接转换到 FP16 时溢出，或在 BF16 中抹掉较小差异。禁止位置不参与最大值计算，全屏蔽行保持 `-inf`。建议用 FP32 构建 bias，预处理无法恢复输入在调用前已丢失的精度。

全屏蔽行的 Attention 输出为零；有效 query 的残差和 FFN 仍会执行，因此整个骨干输出不必为零。

当不需要额外掩码且 T≤W，使用原生 `is_causal=True` 快速路径；其他约束合并为显式 SDPA mask。显式窗口 mask 仍是稠密矩阵，不把长序列计算复杂度变成稀疏 O(TW)。当前依赖下它可能选择 efficient attention 而非 FlashAttention；性能以实际后端和输入规模为准。

## 单步交互和局部重置

```python
cache = model.create_cache(8)
out = model.step(token, cache)  # (8,1,D)，默认关闭当前骨干的梯度

# done 是上一次 env.step 产生的 terminated | truncated；接下来输入 reset 后的新观测。
done = torch.tensor([False, True, False, False, True, False, False, False])
cache.reset(done)              # 仅重置第 1、4 个环境
out = model.step(token, cache)

# 等价方式：在处理当前 token 前重置对应环境；不要把 terminal observation 标为新 episode。
out = model.step(token, cache, episode_starts=done)
cache.reset()                  # 全部重置
```

每次推进整个固定批次，不接受 env_ids。每层 K/V 为预分配 `(B,H,W,head_dim)` 环形存储；`next_positions % W` 选择写入槽位。`cache.next_positions` 是 episode 内下一步位置，不随窗口回绕归零；`cache.lengths` 最多 W。重置将选中环境的两个计数归零，新输入写入 slot 0、RoPE position 0，未重置环境继续推进。

重置为**逻辑失效**，不清零历史 K/V 的物理存储；有效槽位掩码防止旧 episode 被读取。缓存为普通、无计算图张量，外部只读检查 `keys`、`values`、`lengths`、`next_positions`、`nbytes`；不要自行写入它们。数据应保持有限。对象不注册到模型，不出现在 `state_dict`，同一模型可持有多份独立缓存；不同模型/参数版本之间不要复用缓存。同一缓存不可被并发调用。

单 Query 与长 K/V 组合时，SDPA 的 `is_causal=True` 是左上对齐，不能表达“当前步关注全部有效历史”。Step 使用有效槽位掩码，并明确 `is_causal=False`。

### 当前步梯度

```python
current_input = torch.randn(8, 1, 128, requires_grad=True)
hidden = model.step(current_input, cache, enable_grad=True)
loss = critic(hidden).square().mean()
loss.backward()  # 当前输入、当前步骨干及外部 critic 都可获得梯度
```

`enable_grad=False` 默认关闭当前骨干梯度；True 使用局部 `torch.set_grad_enabled(True)`，也能覆盖外层 `no_grad()`。`inference_mode()` 不能这样覆盖，True 时明确报错，须先退出该上下文。外部 Encoder/Heads 的梯度仍由它们各自所在上下文控制。

历史 K/V 始终 detached，**不实现跨时间 BPTT**。正常的序列训练使用 `forward`，让末步损失能沿历史 token 的计算图传回；不要把一串 `step(enable_grad=True)` 当作等价的序列训练。开启单步梯度时，以 out-of-place scatter 构造独立 K/V 快照，当前 Q/K/V 保持可微，保存到持久缓存的副本 detached。因此可以先计算多个 Step、继续覆盖/重置缓存，再对先前输出执行 backward；正常的 optimizer 参数更新仍应在相关 backward 全部完成之后。快照带来额外分配和显存成本，默认无梯度路径不承担此成本。

### 混合精度

```python
model = model.cuda()  # 保留 FP32 参数，使用 autocast
cache = model.create_cache(8, dtype=torch.bfloat16)
with torch.autocast("cuda", dtype=torch.bfloat16):
    out = model.step(token.cuda(), cache)
```

缓存 dtype 必须与**投影后 K/V** 一致，不一定等于参数或输入 dtype。`create_cache` 默认使用参数的 dtype/device，不自动推断未来的 autocast。结构、批大小、设备和 dtype 不兼容时明确报错；需要改变时新建缓存，不隐式迁移。若直接将模型转为 BF16/FP16，输入也应匹配模型 dtype。

## RL 生命周期与外部头

- `terminated` 与 `truncated` 后的新 episode 都应重置 context。时间限制截断的价值 bootstrap 通常还需要旧 episode 的 final observation，应先按算法处理，再重置；不要将自动 reset 后的新观测作为旧 episode 的 bootstrap。
- rollout 收满但环境没有 reset 时，不等于 episode 边界。PPO 等算法更新 Encoder/Backbone 参数后，原缓存却属于旧参数版本；应清空，或用当前网络重建必要上下文。Encoder 参数也改变时，要从原始观测重新编码。
- 训练时保留时间顺序与 episode 信息，不要将时间步任意打散后当作完整历史。切片应包含预热上下文，损失排除预热区和 padding。
- 每层窗口 W 不意味着整个网络只依赖最近 W 个原始输入；L 层的有限感受野可达 `1 + L*(W-1)` 个 token。严格对齐参考是“完整前缀 + 每层同样的窗口/episode mask”，而不是只截取最近 W 个输入重算。重建缓存可以重放最近 `1 + L*(W-1)` 个原始 token，并保留其中的 episode 信息；足够预热后各层保留的状态具有完整上下文。重放完整 episode 前缀会保留原位置编号；只重放有限上下文会重新选择 RoPE 原点，相对位置数学等价，浮点结果不保证逐位相同。
- 梯度裁剪、学习率、优势/回报归一化属于算法层，由调用方选择；示例展示裁剪，不在骨干内部修改梯度。

`init_head(linear, gain=1.0, bias=0.0)` 仅初始化明确指定的 `nn.Linear`，返回该层；不猜测复杂头的最后一层。权重正交初始化，存在 bias 时按参数填充。离散 Actor 常用 gain=0.01，Critic 常用 1.0；MLP 自定义头对其输出层显式调用即可，隐藏层初始化与分布参数化由算法决定。RND 目标网络是否冻结同样属于外部逻辑。

## 测试和基准

完整 CPU 测试（包括短记忆学习；CUDA 项自动跳过）：

```bash
docker compose run --rm -T runner bash -lc \
  'python -m pip install --no-index --no-deps --no-build-isolation -e ./shared && python -m unittest discover -s shared/tests -v'
```

GPU、BF16/FP16、真实 Inductor fullgraph 编译训练验证（首次编译较慢）：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm -T runner bash -lc \
  'python -m pip install --no-index --no-deps --no-build-isolation -e ./shared && python -m unittest discover -s shared/tests -v'
```

`test_float_masks.py` 检查大正负偏移、非均匀/逐 head bias、全屏蔽行，以及因果、窗口、episode、padding 的组合；比较输出与梯度，并在 CUDA FP32/BF16/FP16 下检查真实 fullgraph 编译反向。

`test_memory_learning.py` 使用固定种子 `0/1/2`：开头给出二元线索，中间有独立噪声，最后一步预测线索；仅末步交叉熵参与训练。配置为 T=16、B=64、D=32、2 层、2 heads、W=16、FFN=64，保留默认门控与其他结构选项；外接 `Linear(4,32)` 编码器和 gain=0.01 的两分类输出头。Adam 学习率 3e-4，梯度裁剪 1.0，训练 300 次，在独立的 1024 个样本上要求每个种子准确率至少 95%，只看当前 token 的同一网络准确率在 40%–60%。还检查训练后的缓存交互与整序列 logits 一致。

学习测试覆盖 CPU FP32 和 CUDA FP32/BF16/FP16，参数保留 FP32；FP16 使用 GradScaler（初始 scale=1024），裁剪在 unscale 后执行。每次运行打印各个种子的结果。只运行记忆测试时，把上述命令的 discover 参数改为 `-s shared/tests -p test_memory_learning.py -v`。

性能基准：

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm -T runner bash -lc \
  'python -m pip install --no-index --no-deps --no-build-isolation -e ./shared && python shared/benchmarks/benchmark.py --device cuda --dtype bfloat16 --output shared/runs/benchmark.json'
```

基准包含训练 forward/backward、完整前缀重算、缓存无梯度 Step、当前步带梯度 Step、10% 环境重置；输出延迟、吞吐、峰值显存和采样观察的 SDPA 后端。可加 `--compile` 编译训练/前缀路径，Step 保持 eager；编译时间和 warmup 不计入计时。详见 [VALIDATION.md](VALIDATION.md) 的实际结果。

参考：[PyTorch SDPA](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html)、[ReZero](https://arxiv.org/abs/2003.04887)、[QKNorm](https://arxiv.org/abs/2010.04245)、[RoPE](https://arxiv.org/abs/2104.09864)、[GTrXL](https://arxiv.org/abs/1910.06764)。
