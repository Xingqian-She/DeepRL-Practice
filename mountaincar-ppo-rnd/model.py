import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn

PPO_D_MODEL = 128
NUM_BLOCKS = 2
PPO_EXPANSION_FACTOR = 4
RND_D_MODEL = 64
RND_EXPANSION_FACTOR = 2

class ResidualMLPBlock(nn.Module):
    def __init__(self, d_model, expansion_factor):
        super().__init__()
        hidden_dim = d_model * expansion_factor
        
        self.ffn = nn.Sequential(
            nn.RMSNorm(d_model),
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model, bias=False)
        )
        
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.alpha * self.ffn(x)


class PPOActorCritic(nn.Module):
    def __init__(self, n_observations, n_actions, d_model=PPO_D_MODEL, num_blocks=NUM_BLOCKS,
                 architecture="residual", normalize_input=False):
        super().__init__()
        self.model_config = dict(n_observations=int(n_observations), n_actions=int(n_actions),
                                 d_model=int(d_model), num_blocks=int(num_blocks),
                                 architecture=architecture, normalize_input=bool(normalize_input))
        self.normalize_input = normalize_input
        if normalize_input:
            if n_observations != 2:
                raise ValueError("Fixed input scaling is specific to MountainCar")
            self.register_buffer("obs_center", torch.tensor([-0.3, 0.0]))
            self.register_buffer("obs_scale", torch.tensor([0.9, 0.07]))
        if architecture == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(n_observations, 64), nn.Tanh(),
                nn.Linear(64, 64), nn.Tanh(),
            )
            self.actor = nn.Linear(64, n_actions)
            self.v_extrinsic = nn.Linear(64, 1)
            self.v_intrinsic = nn.Linear(64, 1)
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=2 ** 0.5)
                    nn.init.zeros_(layer.bias)
            nn.init.orthogonal_(self.actor.weight, gain=0.01)
            nn.init.orthogonal_(self.v_extrinsic.weight, gain=1.0)
            nn.init.orthogonal_(self.v_intrinsic.weight, gain=1.0)
            return
        if architecture != "residual":
            raise ValueError(f"Unknown architecture: {architecture}")

        # 主干网络：共享特征提取
        self.backbone = nn.Sequential(
            nn.Linear(n_observations, d_model, bias=False),
            *[ResidualMLPBlock(d_model, PPO_EXPANSION_FACTOR) for _ in range(num_blocks)],
        )

        # Actor 头
        self.actor = nn.Sequential(
            ResidualMLPBlock(d_model, PPO_EXPANSION_FACTOR),
            nn.RMSNorm(d_model),
            nn.Linear(d_model, n_actions)
        )
        
        # 外在价值头 (针对 extrinsic reward)
        self.v_extrinsic = nn.Sequential(
            ResidualMLPBlock(d_model, PPO_EXPANSION_FACTOR),
            nn.RMSNorm(d_model),
            nn.Linear(d_model, 1)
        )
        
        # 内在价值头 (针对 intrinsic reward)
        self.v_intrinsic = nn.Sequential(
            ResidualMLPBlock(d_model, PPO_EXPANSION_FACTOR),
            nn.RMSNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        # 共享特征
        if self.normalize_input:
            x = (x - self.obs_center) / self.obs_scale
        feat = self.backbone(x)
        logits = self.actor(feat)
        v_ext = self.v_extrinsic(feat).squeeze(-1)
        v_int = self.v_intrinsic(feat).squeeze(-1)
        return logits, v_ext, v_int

    def act(self, x):
        """交互用：返回动作、对数概率、外在价值、内在价值"""
        logits, v_ext, v_int = self(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action, probs.log_prob(action), v_ext, v_int

    def evaluate_actions(self, x, action):
        """训练用：返回对数概率、熵、外在价值、内在价值"""
        logits, v_ext, v_int = self(x)
        probs = Categorical(logits=logits)
        return probs.log_prob(action), probs.entropy(), v_ext, v_int



class RNDTargetPredictor(nn.Module):
    def __init__(self, obs_dim, rnd_d_model=RND_D_MODEL):
        super().__init__()
        # Target: 随机权重且冻结
        self.target = nn.Sequential(
            nn.Linear(obs_dim, rnd_d_model * RND_EXPANSION_FACTOR),
            nn.SiLU(),
            nn.Linear(rnd_d_model * RND_EXPANSION_FACTOR, rnd_d_model)
        )
        # Predictor: 需要训练
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, int(rnd_d_model * RND_EXPANSION_FACTOR * 1.1)),
            nn.SiLU(),
            nn.Linear(int(rnd_d_model * RND_EXPANSION_FACTOR * 1.1), rnd_d_model)
        )
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.target(x), self.predictor(x)
