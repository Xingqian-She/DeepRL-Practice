import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

D_MODEL = 128
NUM_BLOCKS = 2
EXPANSION_FACTOR = 4

class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model, expansion_factor):
        super().__init__()
        hidden_dim = d_model * expansion_factor
        
        self.ffn = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.alpha * self.ffn(x)


class ResMLPBackbone(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, num_blocks):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            *[ResidualMLPBlock(d_model, EXPANSION_FACTOR) for _ in range(num_blocks)],
            nn.RMSNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x):
        return self.net(x)


class PPOSeparateActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions, d_model=D_MODEL, num_blocks=NUM_BLOCKS):
        super().__init__()
        self.actor = ResMLPBackbone(n_observations, n_actions, d_model, num_blocks)
        self.critic = ResMLPBackbone(n_observations, 1, d_model, num_blocks)

    def get_value(self, x):
        # 输出形状直接从 (B, 1) 变成 (B,)
        return self.critic(x).squeeze(-1)

    def act(self, x):
        """与环境交互时调用（采样动作）"""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        # 返回：动作, 动作对数概率, 状态价值
        return action, probs.log_prob(action), self.get_value(x)

    def evaluate_actions(self, x, action):
        """PPO 训练更新时调用（评估旧动作）"""
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        # 返回：旧动作在新网络下的对数概率, 策略熵, 状态价值
        return probs.log_prob(action), probs.entropy(), self.get_value(x)