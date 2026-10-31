"""已提供：沿用 PPO-RND 的 residual 风格，输出改为每个动作的 Q 值。"""

import torch
from torch import nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, d_model * expansion_factor, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * expansion_factor, d_model, bias=False),
        )
        # 和原模型一样：初始残差分支权重为 0，训练时再学习分支贡献。
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.alpha * self.ffn(x)


class ResidualQNetwork(nn.Module):
    def __init__(self, n_observations=2, n_actions=3, d_model=128, num_blocks=2):
        super().__init__()
        if n_observations != 2:
            raise ValueError("本教程的固定输入缩放仅适用于 MountainCar")
        self.model_config = dict(n_observations=n_observations, n_actions=n_actions,
                                 d_model=d_model, num_blocks=num_blocks)
        self.register_buffer("obs_center", torch.tensor([-0.3, 0.0]))
        self.register_buffer("obs_scale", torch.tensor([0.9, 0.07]))
        self.backbone = nn.Sequential(
            nn.Linear(n_observations, d_model, bias=False),
            *[ResidualMLPBlock(d_model) for _ in range(num_blocks)],
        )
        self.q_head = nn.Sequential(
            ResidualMLPBlock(d_model),
            nn.RMSNorm(d_model),
            nn.Linear(d_model, n_actions),
        )

    def forward(self, obs):
        """原始 obs: [B, 2] -> Q(s, ·): [B, 3]；不要再做 softmax。"""
        x = (obs - self.obs_center) / self.obs_scale
        return self.q_head(self.backbone(x))
