"""练习 3：普通 DQN；暂不加入 Double DQN、优先级回放或其他技巧。"""

import numpy as np
import torch
from torch.nn import functional as F


def epsilon_at(step, config):
    """已提供：epsilon 按环境步数线性衰减，而不是按回合数衰减。"""
    fraction = min(step / max(config.eps_decay_steps, 1), 1.0)
    return config.eps_start + fraction * (config.eps_end - config.eps_start)


def select_action(q_net, obs, epsilon, rng, n_actions, device):
    """TODO 3a：epsilon-greedy，返回 Python int。

    rng 是 np.random.Generator；探索用 rng.integers(n_actions)。
    利用时：obs [2] -> float32 [1,2] -> Q [1,3] -> argmax。
    利用分支用 torch.no_grad()，不构建梯度图。
    """
    # raise NotImplementedError("TODO 3a: epsilon-greedy 动作选择")
    if rng.random() > epsilon:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(q_net(obs_tensor).argmax(dim=1).item())
            return action
    else:
        action = int(rng.integers(n_actions))
        return action


def compute_loss(policy, target, batch, gamma):
    """TODO 3b：返回标量 Huber loss（F.smooth_l1_loss）。

    1. policy(obs).gather(1, actions).squeeze(1) 得到 q_sa: [B]。
    2. 在 no_grad 内取 target(next_obs).max(dim=1).values: [B]。
    3. y = rewards + gamma * (~terminated).float() * next_q。
    4. 比较 q_sa 和 y，避免 [B] 与 [B,1] 广播成 [B,B]。

    MountainCar 的 TimeLimit 截断仍 bootstrap；真正到达目标时不 bootstrap。
    所有 next_obs 都是真实后继观测，因此可以统一计算，再用 mask 屏蔽。
    """
    # raise NotImplementedError("TODO 3b: Bellman target 和 TD loss")
    obs = batch['obs']
    actions = batch['actions']
    rewards = batch['rewards']
    next_obs = batch['next_obs']
    terminated = batch['terminated']

    q_sa = policy(obs).gather(1, actions).squeeze(1)

    with torch.no_grad():
        next_q = target(next_obs).max(dim=1).values
        y = rewards + gamma * (~terminated).float() * next_q
    
    return F.smooth_l1_loss(q_sa, y)


def update(policy, target, optimizer, replay, config, device):
    """已提供：采样、反传、梯度裁剪；公式由你在 compute_loss 中实现。"""
    batch = replay.sample(config.batch_size, device)
    loss = compute_loss(policy, target, batch, config.gamma)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return loss.item()
