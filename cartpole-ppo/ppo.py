import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
UPDATE_EPOCHS = 4
MINIBATCH_SIZE = 64


class RolloutBuffer:
    """存储固定长度 rollout；每一行对应一次 env.step() 的 transition。"""

    def __init__(self, num_steps, num_envs, obs_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, obs_dim)).to(device)
        self.actions = torch.zeros((num_steps, num_envs)).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.terminated = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.truncated = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
        self.next_values = torch.zeros((num_steps, num_envs)).to(device)
        
        self.advantages = torch.zeros((num_steps, num_envs)).to(device)
        self.returns = torch.zeros((num_steps, num_envs)).to(device)

        self.step = 0

    def add(self, obs, action, logprob, reward, value, *, terminated, truncated, next_value):
        """保存当前状态、动作、原始奖励、价值及本次 step 的结束标志和后继价值。

        除 obs 为 (num_envs, obs_dim) 外，各字段形状均为 (num_envs,)。
        截断时 next_value 必须来自旧回合的最后状态，而非 reset 后的状态。
        """
        if self.step >= self.num_steps:
            raise RuntimeError("RolloutBuffer is full; reset it before adding more data")
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.values[self.step] = value
        self.terminated[self.step] = terminated
        self.truncated[self.step] = truncated
        self.next_values[self.step] = next_value
        self.step += 1

    def reset(self):
        """每次更新完网络后，重置指针即可，新数据会覆盖旧数据"""
        self.step = 0

    @torch.no_grad()
    def compute_returns_and_advantages(self, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
        """使用 GAE (Generalized Advantage Estimation) 计算优势和回报

        真正终止时不 bootstrap；终止或截断时都不连接下一个回合的优势。
        rollout 收满不代表 episode 结束：末步仍用 next_values 做 bootstrap，
        但没有采样到的后续优势以 0 作为递推边界。
        """
        if self.step != self.num_steps:
            raise RuntimeError("Fill the rollout before computing returns and advantages")
        last_advantage = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            bootstrap_mask = (~self.terminated[t]).float()
            trace_mask = (~(self.terminated[t] | self.truncated[t])).float()

            # TD error 是否使用后继价值，只取决于是否真正终止。
            delta = (
                self.rewards[t]
                + gamma * bootstrap_mask * self.next_values[t]
                - self.values[t]
            )

            # GAE 是否接上下一步优势，取决于是否仍处于同一个回合。
            last_advantage = delta + gamma * gae_lambda * trace_mask * last_advantage
            self.advantages[t] = last_advantage

        # 目标价值回报: Return = Advantage + Value
        self.returns = self.advantages + self.values

    def get_batch_data(self):
        """将 (num_steps, num_envs) 展平为 (batch_size,) 供训练使用"""
        b_obs = self.obs.reshape((-1, self.obs.shape[-1]))
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)
        return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values


class PPOTrainer:
    """负责 PPO 的损失计算与参数更新"""

    def __init__(
        self,
        agent: nn.Module,
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_coef=CLIP_COEF,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        update_epochs=UPDATE_EPOCHS,
        minibatch_size=MINIBATCH_SIZE,
    ):
        self.agent = torch.compile(agent, dynamic=True)
        optim_groups = [
            {'params': [p for p in self.agent.parameters() if p.ndim >= 2], 'weight_decay': 1e-2},
            {'params': [p for p in self.agent.parameters() if p.ndim < 2],  'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(optim_groups, lr=lr, eps=1e-5, amsgrad=True)

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef  # PPO 截断范围 epsilon
        self.ent_coef = ent_coef  # 熵正则化系数（鼓励探索）
        self.vf_coef = vf_coef  # 价值损失系数
        self.max_grad_norm = max_grad_norm  # 梯度裁剪阈值
        self.update_epochs = update_epochs  # 每次收集数据后更新几轮
        self.minibatch_size = minibatch_size

    def update(self, buffer: RolloutBuffer):
        """执行 PPO 更新"""
        # 1. 获取展平后的全部数据
        b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values = (
            buffer.get_batch_data()
        )
        batch_size = b_obs.shape[0]
        b_inds = np.arange(batch_size)

        # 用于记录日志的统计变量
        clipfracs = []

        # 2. 多轮次 (Epochs) 更新
        for epoch in range(self.update_epochs):
            # 打乱数据索引
            np.random.shuffle(b_inds)

            # 3. 切分 Mini-batch 进行更新
            for start in range(0, batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                # 将 mini-batch 数据送入 model.py 中重新评估
                newlogprob, entropy, newvalue = self.agent.evaluate_actions(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )

                # 计算新旧策略的概率比值: ratio = pi_theta(a|s) / pi_theta_old(a|s)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # 统计有多少比例的数据触发了截断（用于 Debug 监控）
                with torch.no_grad():
                    clipfracs += [
                        ((ratio - 1.0).abs() > self.clip_coef)
                        .float()
                        .mean()
                        .item()
                    ]

                # 优势标准化 (Advantage Normalization) - 极其重要的小技巧，稳定训练
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # --- A. 策略损失 (Policy / Actor Loss) ---
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # --- B. 价值损失 (Value / Critic Loss) ---
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                # --- C. 熵损失 (Entropy Loss) ---
                entropy_loss = entropy.mean()

                # --- 总损失 ---
                # 目标是最小化 pg_loss 和 v_loss，最大化 entropy_loss（所以前面是减号）
                loss = (
                    pg_loss
                    - self.ent_coef * entropy_loss
                    + self.vf_coef * v_loss
                )

                # 4. 反向传播与梯度更新
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.max_grad_norm
                )  # 梯度裁剪
                self.optimizer.step()

        # 重置缓冲区，准备下一次采样
        buffer.reset()

        return {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "clip_fraction": np.mean(clipfracs),
        }
