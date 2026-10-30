"""Rollout 数据及两路 GAE：只保存和处理数据，不更新网络。"""

import torch


class RolloutBuffer:
    """存储固定长度 rollout；每一行对应一次 env.step() 的 transition。"""

    def __init__(self, num_steps, num_envs, obs_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = torch.zeros((num_steps, num_envs, obs_dim)).to(device)
        # PPO 使用 s_t；RND 使用真实后继状态 s_{t+1}（含 final_obs）。
        self.rnd_obs = torch.zeros_like(self.obs)
        self.actions = torch.zeros((num_steps, num_envs)).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.terminated = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
        self.truncated = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)

        self.values_ext = torch.zeros((num_steps, num_envs)).to(device)
        self.values_int = torch.zeros((num_steps, num_envs)).to(device)
        self.next_values_ext = torch.zeros((num_steps, num_envs)).to(device)
        self.next_values_int = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards_ext = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards_int = torch.zeros((num_steps, num_envs)).to(device)
        self.returns_ext = torch.zeros((num_steps, num_envs)).to(device)
        self.returns_int = torch.zeros((num_steps, num_envs)).to(device)
        self.adv_ext = torch.zeros((num_steps, num_envs)).to(device)
        self.adv_int = torch.zeros((num_steps, num_envs)).to(device)

        self.step = 0

    def add(self, *, obs, rnd_obs, action, logprob, 
            reward_ext, reward_int, value_ext, next_value_ext, value_int, next_value_int, 
            terminated, truncated):
        """保存当前状态、动作、原始奖励、价值及本次 step 的结束标志和后继价值。

        obs 和 rnd_obs 为 (num_envs, obs_dim)，其余字段为 (num_envs,)。
        截断时 next_value_ext 来自旧回合最后状态；next_value_int 来自 reset 状态。
        """
        if self.step >= self.num_steps:
            raise RuntimeError("RolloutBuffer is full; reset it before adding more data")
        self.obs[self.step] = obs
        self.rnd_obs[self.step] = rnd_obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.terminated[self.step] = terminated
        self.truncated[self.step] = truncated

        self.values_ext[self.step] = value_ext
        self.values_int[self.step] = value_int
        self.next_values_ext[self.step] = next_value_ext
        self.next_values_int[self.step] = next_value_int
        self.rewards_ext[self.step] = reward_ext
        self.rewards_int[self.step] = reward_int
        
        self.step += 1

    def reset(self):
        """每次更新完网络后，重置指针即可，新数据会覆盖旧数据"""
        self.step = 0

    @torch.no_grad()
    def compute_ext_returns_and_advantages(self, *, gamma, lmbda):
        if self.step != self.num_steps:
            raise RuntimeError("Fill the rollout before computing returns and advantages")
        last_advantage = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            bootstrap_mask = (~self.terminated[t]).float()
            trace_mask = (~(self.terminated[t] | self.truncated[t])).float()

            # TD error 是否使用后继价值，只取决于是否真正终止。
            delta = (
                self.rewards_ext[t]
                + gamma * bootstrap_mask * self.next_values_ext[t]
                - self.values_ext[t]
                )

            # GAE 是否接上下一步优势，取决于是否仍处于同一个回合。
            last_advantage = (
                delta 
                + gamma * lmbda * trace_mask * last_advantage
                )
            self.adv_ext[t] = last_advantage

        # 目标价值回报: Return = Advantage + Value
        self.returns_ext = self.adv_ext + self.values_ext

    @torch.no_grad()
    def compute_int_returns_and_advantages(self, *, gamma, lmbda):
        if self.step != self.num_steps:
            raise RuntimeError("Fill the rollout before computing returns and advantages")
        last_advantage = torch.zeros(self.num_envs, device=self.device)
        for t in reversed(range(self.num_steps)):
            # 好奇心跨回合传播
            delta = (
                self.rewards_int[t] 
                + gamma * self.next_values_int[t] 
                - self.values_int[t]
            )
            self.adv_int[t] = last_advantage = (
                delta 
                + gamma * lmbda * last_advantage
            )

        # 目标价值回报: Return = Advantage + Value
        self.returns_int = self.adv_int + self.values_int

    def get_batch_data(self):
        """将 (num_steps, num_envs) 展平为 (batch_size,) 供训练使用"""
        return {
            "obs": self.obs.reshape((-1, self.obs.shape[-1])),
            "rnd_obs": self.rnd_obs.reshape((-1, self.rnd_obs.shape[-1])),
            "logprobs": self.logprobs.reshape(-1),
            "actions": self.actions.reshape(-1),
            "adv_ext": self.adv_ext.reshape(-1),
            "adv_int": self.adv_int.reshape(-1),
            "returns_ext": self.returns_ext.reshape(-1),
            "returns_int": self.returns_int.reshape(-1),
            "values_ext": self.values_ext.reshape(-1),
            "values_int": self.values_int.reshape(-1),
        }
