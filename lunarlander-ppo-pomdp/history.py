"""有界策略输入历史：保留已抽样的缺失读数、提示和动作，不重抽缺失模式。"""

from dataclasses import dataclass

import torch


@dataclass
class Token:
    observations: torch.Tensor       # (N,16)，失灵置零的读数与可用性向量
    previous_actions: torch.Tensor   # (N,)，-1 表示回合首步
    episode_starts: torch.Tensor     # (N,)
    position_ids: torch.Tensor       # (N,)，这次行为策略实际使用的位置

    def clone(self):
        return Token(*(getattr(self, name).detach().clone() for name in self.__dataclass_fields__))


class History:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tokens = []

    def append(self, token):
        self.tokens.append(token.clone())
        self.tokens = self.tokens[-self.capacity:]

    def snapshot(self):
        return [token.clone() for token in self.tokens]

    @torch.no_grad()
    def rebuild(self, agent, num_envs):
        """参数更新后，从原始输入重建新缓存；不修改 shared 缓存的只读张量。

        有限历史可能从 episode 中途开始。新缓存会选择新的 RoPE 原点，
        因而必须同步记录重放时实际采用的位置，训练不能沿用上轮的编号。
        所有环境按同一条向量时间轴重放；episode_starts 负责各环境局部重置。
        """
        cache = agent.create_cache(num_envs)
        for token in self.tokens:
            token.position_ids = torch.where(token.episode_starts, 0, cache.next_positions).clone()
            agent.step_outputs(
                token.observations, token.previous_actions, cache, token.episode_starts,
            )
        return cache
