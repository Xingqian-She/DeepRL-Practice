"""20 维输入，可选共享或独立的策略/价值时序骨干。"""

from copy import deepcopy
from dataclasses import dataclass
from itertools import chain

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from rl_shared import KVCache, TransformerBackbone, init_head

from config import AgentConfig
from sensors import MODEL_INPUT_DIM, OBSERVATION_DIM


@dataclass
class SeparateCache:
    """两个独立 KV 状态；两分支读取相同时间轴和回合重置信号。"""
    policy: KVCache
    value: KVCache

    @property
    def next_positions(self):
        # 两分支每次一起推进；Collector/History 只需要一份输入位置编号。
        return self.policy.next_positions


def cache_tensors(cache):
    """用于 eager/compile 验证，两个分支的全部可变缓存都必须比较。"""
    if isinstance(cache, SeparateCache):
        return {f"{branch}.{name}": tensor
                for branch in ("policy", "value")
                for name, tensor in cache_tensors(getattr(cache, branch)).items()}
    return {name: getattr(cache, name) for name in ("keys", "values", "lengths", "next_positions")}


def make_model_input(observations, previous_actions, use_previous_action=True):
    """(...,16) + (...) 动作编号 → (...,20)；-1 表示回合首步没有先前动作。

    即使动作输入关闭，也保留 4 个零通道，确保同一 checkpoint 可以直接评估。
    observations 已含实际抽样的缺失读数与可用性，不在模型内部重新抽样。
    """
    if observations.shape[:-1] != previous_actions.shape or observations.shape[-1] != OBSERVATION_DIM:
        raise ValueError("观测须为 (...,16)，先前动作须为相同前缀形状 (...)")
    if previous_actions.dtype != torch.long:
        raise TypeError("先前动作应为 torch.long；-1 表示尚无动作")
    if ((previous_actions < -1) | (previous_actions > 3)).any():
        raise ValueError("先前动作只能为 -1 或 0..3")
    one_hot = F.one_hot(previous_actions.clamp_min(0), num_classes=4).to(observations.dtype)
    one_hot = one_hot * (previous_actions >= 0).unsqueeze(-1)
    if not use_previous_action:
        one_hot = torch.zeros_like(one_hot)
    return torch.cat((observations, one_hot), dim=-1)


class Agent(nn.Module):
    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(nn.Linear(MODEL_INPUT_DIM, config.hidden_dim), nn.Tanh())
        self.backbone = TransformerBackbone(config.backbone_config())
        self.actor = init_head(nn.Linear(config.hidden_dim, 4), gain=0.01)
        self.critic = init_head(nn.Linear(config.hidden_dim, 1), gain=1.0)
        if config.backbone_mode == "separate":
            # 只复制初始数值，不共享 Parameter/storage；不额外消耗 RNG。
            # 同 seed 的两种模式具有完全相同的初始策略、价值和采样随机流。
            self.value_encoder = deepcopy(self.encoder)
            self.value_backbone = deepcopy(self.backbone)
        # 编译的是绑定函数，不用 OptimizedModule 包装模型，避免改变权重名称。
        self._compiled_step = None
        self._compiled_sequence = None

    def encode(self, observations, previous_actions):
        """策略 Encoder；独立模式的价值编码在时序前向中单独计算。"""
        features = make_model_input(observations, previous_actions, self.config.use_previous_action)
        return self.encoder(features)

    def heads(self, hidden, value_hidden=None):
        if value_hidden is None:
            if self.config.backbone_mode == "separate":
                raise ValueError("separate 模式必须提供独立的 value_hidden")
            value_hidden = hidden
        return self.actor(hidden), self.critic(value_hidden).squeeze(-1)

    def sample(self, logits):
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions)

    def create_cache(self, batch_size):
        policy = self.backbone.create_cache(batch_size, dtype=self.actor.weight.dtype)
        if self.config.backbone_mode == "shared":
            return policy
        return SeparateCache(policy, self.value_backbone.create_cache(batch_size, dtype=self.critic.weight.dtype))

    def policy_parameters(self):
        return chain(self.encoder.parameters(), self.backbone.parameters(), self.actor.parameters())

    def value_parameters(self):
        if self.config.backbone_mode == "shared":
            return chain(self.encoder.parameters(), self.backbone.parameters(), self.critic.parameters())
        return chain(self.value_encoder.parameters(), self.value_backbone.parameters(), self.critic.parameters())

    def parameter_counts(self):
        policy = {id(p): p.numel() for p in self.policy_parameters()}
        value = {id(p): p.numel() for p in self.value_parameters()}
        return {"total": sum(p.numel() for p in self.parameters()),
                "policy": sum(policy.values()), "value": sum(value.values()),
                "shared": sum(policy[key] for key in policy.keys() & value.keys())}

    def _step_core(self, features, cache, episode_starts):
        encoded = self.encoder(features).unsqueeze(1)
        if self.config.backbone_mode == "separate":
            hidden = self.backbone.step(encoded, cache.policy, episode_starts=episode_starts)
            value_hidden = self.value_backbone.step(
                self.value_encoder(features).unsqueeze(1), cache.value, episode_starts=episode_starts,
            )
            return self.heads(hidden[:, 0], value_hidden[:, 0])
        hidden = self.backbone.step(encoded, cache, episode_starts=episode_starts)
        return self.heads(hidden[:, 0])

    def _sequence_core(self, features, episode_starts, valid_mask, position_ids):
        hidden = self.backbone(
            self.encoder(features), episode_starts=episode_starts,
            valid_mask=valid_mask, position_ids=position_ids,
        )
        if self.config.backbone_mode == "separate":
            value_hidden = self.value_backbone(
                self.value_encoder(features), episode_starts=episode_starts,
                valid_mask=valid_mask, position_ids=position_ids,
            )
            return self.heads(hidden, value_hidden)
        return self.heads(hidden)

    @torch.no_grad()
    def step_outputs(self, observations, previous_actions, cache, episode_starts):
        """交互：每个环境只推进一个 token；返回 (N,4)、(N,)。"""
        features = make_model_input(observations, previous_actions, self.config.use_previous_action)
        with sdpa_kernel(SDPBackend.MATH):
            core = self._compiled_step or self._step_core
            return core(features, cache, episode_starts)

    def forward_sequence(self, observations, previous_actions, *, episode_starts, valid_mask, position_ids):
        """并行参考/后继估值路径，保持 eager，避免为变化的估值批次反复编译。"""
        features = make_model_input(observations, previous_actions, self.config.use_previous_action)
        with sdpa_kernel(SDPBackend.MATH):
            return self._sequence_core(features, episode_starts, valid_mask, position_ids)

    def evaluate_actions(self, batch):
        features = make_model_input(
            batch["observations"], batch["previous_actions"], self.config.use_previous_action,
        )
        with sdpa_kernel(SDPBackend.MATH):
            core = self._compiled_sequence or self._sequence_core
            logits, values = core(features, batch["episode_starts"], batch["valid_mask"], batch["position_ids"])
        distribution = Categorical(logits=logits)
        return distribution.log_prob(batch["actions"]), distribution.entropy(), values
