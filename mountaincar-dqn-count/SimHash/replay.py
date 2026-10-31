"""练习 2：先存 CPU 上的原始观测，采样时统一转换成训练设备上的张量。"""

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float       # 采集当时混合好的奖励，回放时不重新计算 bonus
    next_obs: np.ndarray
    terminated: bool   # 只存真正的 MDP 终止；不要把 truncated 混进来


class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        self.memory = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, reward, next_obs, terminated):
        """TODO 2a：构造 Transition 并 append；两份观测都要 copy。

        复制是为了避免环境复用数组或调用方修改数组时，历史样本被改写。
        """
        # raise NotImplementedError("TODO 2a: 保存一条 transition")
        transition = Transition(
            obs=np.array(obs, copy=True),
            action=int(action),
            reward=float(reward),
            next_obs=np.array(next_obs, copy=True),
            terminated=bool(terminated)
        )
        self.memory.append(transition)

    def sample(self, batch_size, device):
        """TODO 2b：用 self.rng.sample 无放回采样，返回含以下键的 dict。

        obs / next_obs: float32 [B,2]（先 np.stack，再 as_tensor）
        actions: int64 [B,1]，供 Q.gather(1, actions) 使用
        rewards: float32 [B]；terminated: bool [B]
        所有张量都放到 device；不足 B 条时由调用方避免采样。
        """
        # raise NotImplementedError("TODO 2b: 随机采样并组成张量 batch")
        batch = self.rng.sample(self.memory, batch_size)
        obs = torch.as_tensor(
                    np.stack([t.obs for t in batch]),
                    dtype=torch.float32,
                    device=device
                )
        next_obs = torch.as_tensor(
                    np.stack([t.next_obs for t in batch]),
                    dtype=torch.float32,
                    device=device
                )
        actions = torch.as_tensor(
                    [t.action for t in batch],
                    dtype=torch.int64,
                    device=device
                ).unsqueeze(1)
        rewards = torch.as_tensor(
                    [t.reward for t in batch],
                    dtype=torch.float32,
                    device=device
                )
        terminated = torch.as_tensor(
                    [t.terminated for t in batch],
                    dtype=torch.bool,
                    device=device
                )
        
        return {
            'obs': obs,
            'next_obs': next_obs,
            'actions': actions,
            'rewards': rewards,
            'terminated': terminated
        }
        