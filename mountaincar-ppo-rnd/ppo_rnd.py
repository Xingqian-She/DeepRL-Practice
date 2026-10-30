"""PPO + RND 更新：同一个 minibatch 先更新 Agent，再更新 Predictor。

数据在 buffer.py，RND 奖励与统计量在 rnd.py；保留它们的旧导入路径。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from buffer import RolloutBuffer
from rnd import RunningMeanStd, RewardForwardFilter, normalize_rnd_obs


LR = 3e-4
GAMMA = 0.99
EXT_LAMBDA = 0.95
INT_LAMBDA = 0.85
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 64
EXT_COEF = 2.0  # 外在奖励优势权重
INT_COEF = 1.0  # 内在好奇心优势权重
RND_UPDATE_PROPORTION = 0.25  # 每次 minibatch 更新时的样本保留概率


def combine_advantages(adv_ext, adv_int, ext_coef, int_coef):
    """先保留两路优势的相对尺度，再对整个 rollout 做一次共同标准化。

    统一标准化是本实现的 PPO 稳定化选择；不是对内在优势单独标准化。
    critic 的 returns 不经过此变换。
    """
    advantages = ext_coef * adv_ext + int_coef * adv_int
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)


def make_optimizer(model, lr):
    """只优化可训练参数；矩阵权重衰减，偏置和归一化参数不衰减。"""
    parameters = [p for p in model.parameters() if p.requires_grad]
    return optim.AdamW(
        [
            {"params": [p for p in parameters if p.ndim >= 2], "weight_decay": 1e-2},
            {"params": [p for p in parameters if p.ndim < 2], "weight_decay": 0.0},
        ],
        lr=lr, eps=1e-5, amsgrad=True,
    )


class PPOTrainer:
    """负责 PPO-RND 的损失计算与参数更新"""

    def __init__(
        self,
        agent: nn.Module,
        rnd_model: nn.Module,
        lr=LR,
        gamma=GAMMA,
        ext_lambda=EXT_LAMBDA,
        int_lambda=INT_LAMBDA,
        clip_coef=CLIP_COEF,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        ext_coef=EXT_COEF,
        int_coef=INT_COEF,
        rnd_update_proportion=RND_UPDATE_PROPORTION,
        max_grad_norm=MAX_GRAD_NORM,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
        rnd_lr=4e-5,
        compile_models=True,
    ):
        self.agent = torch.compile(agent, dynamic=True) if compile_models else agent
        self.agent_optimizer = make_optimizer(self.agent, lr)
        self.rnd_model = torch.compile(rnd_model, dynamic=True) if compile_models else rnd_model
        self.rnd_optimizer = make_optimizer(self.rnd_model, rnd_lr)

        # 超参数
        self.gamma = gamma
        self.ext_lambda = ext_lambda
        self.int_lambda = int_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.ext_coef = ext_coef
        self.int_coef = int_coef
        self.rnd_update_proportion = rnd_update_proportion
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

    def update(self, buffer: RolloutBuffer, obs_rms: RunningMeanStd = None):
        """所有 epoch 共用固定的奖励、回报目标和混合优势。

        obs_rms 必须与采样时相同，调用者应在本方法完成后更新统计量。
        返回本轮所有 minibatch 的平均指标，避免最后一个 minibatch 的噪声。
        """
        batch = buffer.get_batch_data()
        indices = np.arange(batch["obs"].shape[0])
        advantages = combine_advantages(
            batch["adv_ext"], batch["adv_int"], self.ext_coef, self.int_coef
        )
        minibatch_logs = []
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.minibatch_size):
                selected = indices[start:start + self.minibatch_size]
                minibatch = {name: value[selected] for name, value in batch.items()}
                logs = self._update_agent(minibatch, advantages[selected])
                logs["rnd_loss"] = self._update_predictor(minibatch["rnd_obs"], obs_rms)
                minibatch_logs.append(logs)

        buffer.reset()  # 只重置写指针，观测仍可用于更新 obs_rms。
        return {key: float(np.mean([row[key] for row in minibatch_logs]))
                for key in minibatch_logs[0]}

    def _update_agent(self, batch, advantages):
        """L = L_clip + vf_coef * (L_Vext + L_Vint) - ent_coef * H。"""
        new_logprob, entropy, new_v_ext, new_v_int = self.agent.evaluate_actions(
            batch["obs"], batch["actions"].long()
        )
        ratio = (new_logprob - batch["logprobs"]).exp()
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()

        # 最大化 min(ratio*A, clip(ratio)*A)，转换为最小化负值。
        loss_unclipped = -advantages * ratio
        loss_clipped = -advantages * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
        policy_loss = torch.max(loss_unclipped, loss_clipped).mean()
        value_ext_loss = 0.5 * ((new_v_ext - batch["returns_ext"]) ** 2).mean()
        value_int_loss = 0.5 * ((new_v_int - batch["returns_int"]) ** 2).mean()
        entropy_mean = entropy.mean()
        loss = (
            policy_loss - self.ent_coef * entropy_mean
            + self.vf_coef * (value_ext_loss + value_int_loss)
        )
        self.agent_optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
        self.agent_optimizer.step()
        return {
            "policy_loss": policy_loss.item(),
            "value_ext_loss": value_ext_loss.item(),
            "value_int_loss": value_int_loss.item(),
            "entropy": entropy_mean.item(),
            "clip_fraction": clip_fraction,
            "agent_grad_norm": grad_norm.item(),
        }

    def _update_predictor(self, observations, obs_rms):
        """只拟合 Target 特征，不用缩放后的内在奖励训练 Predictor。"""
        if obs_rms is not None:
            observations = normalize_rnd_obs(observations, obs_rms)
        target_feat, predict_feat = self.rnd_model(observations)
        errors = ((predict_feat - target_feat.detach()) ** 2).mean(dim=-1)
        # 每个 minibatch 重新抽取约 25% 样本，控制 Predictor 的学习速度。
        mask = (
            torch.rand(len(errors), device=observations.device)
            < self.rnd_update_proportion
        ).float()
        loss = (errors * mask).sum() / mask.sum().clamp(min=1.0)
        self.rnd_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.rnd_model.parameters(), self.max_grad_norm)
        self.rnd_optimizer.step()
        return loss.item()
