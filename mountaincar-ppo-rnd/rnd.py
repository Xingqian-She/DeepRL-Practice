"""RND 奖励和归一化：管理观测尺度、原始奖励及奖励尺度。"""

import numpy as np
import torch


class RunningMeanStd:
    """对RND使用, 动态计算状态和奖励的 Mean 和 Std"""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return np.sqrt(self.var)


class RewardForwardFilter:
    """每个环境独立维护 F_t = gamma * F_{t-1} + r_t。

    这是历史奖励的前向滤波器，用于估计回报尺度，不是 critic 的未来回报目标。
    内在奖励按 continuing task 处理，因此跨 rollout 和 episode 均不清零。
    """

    def __init__(self, gamma, num_envs):
        self.gamma = gamma
        self.discounted_rewards = np.zeros(num_envs, dtype=np.float64)

    def update(self, rewards):
        self.discounted_rewards = (
            self.gamma * self.discounted_rewards
            + np.asarray(rewards, dtype=np.float64)
        )
        return self.discounted_rewards.copy()


def normalize_rnd_obs(obs, obs_rms):
    """奖励计算和 predictor 更新共用此变换；一轮内必须固定 obs_rms。"""
    mean = torch.as_tensor(obs_rms.mean, dtype=obs.dtype, device=obs.device)
    std = torch.as_tensor(obs_rms.std, dtype=obs.dtype, device=obs.device)
    return torch.clamp((obs - mean) / (std + 1e-8), -5.0, 5.0)


@torch.no_grad()
def compute_intrinsic_reward(rnd_model, obs_tensor, obs_rms):
    """使用归一化后的状态计算 RND 内在好奇心奖励"""
    norm_obs = normalize_rnd_obs(obs_tensor, obs_rms)

    target_feat, predict_feat = rnd_model(norm_obs)
    # 内在奖励为预测特征与目标特征的均方误差
    reward_int = ((predict_feat - target_feat) ** 2).mean(dim=-1)
    return reward_int


def scale_intrinsic_rewards(buffer, reward_filter, reward_rms):
    """用历史累计奖励的标准差缩放单步奖励，返回日志指标。

    F_t = gamma * F_{t-1} + r_t；F 只估计尺度，不作为 critic 目标。
    每个环境独立滤波，跨回合和 rollout 保留历史。
    """
    raw_rewards = buffer.rewards_int.cpu().numpy().copy()
    filtered_rewards = np.stack([
        reward_filter.update(rewards) for rewards in raw_rewards
    ])
    reward_rms.update(filtered_rewards.reshape(-1))
    buffer.rewards_int /= (
        torch.as_tensor(
            reward_rms.std, dtype=buffer.rewards_int.dtype,
            device=buffer.rewards_int.device,
        ) + 1e-8
    )
    return {
        "reward_int_raw_mean": float(raw_rewards.mean()),
        "reward_int_scaled_mean": buffer.rewards_int.mean().item(),
        "int_return_std": float(reward_rms.std),
    }
