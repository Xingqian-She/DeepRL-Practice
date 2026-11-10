"""向量采样：在新观测上抽样失灵，保存策略实际看见的输入。"""

import numpy as np
import torch

from buffer import RolloutBuffer
from history import History, Token
from sensors import SensorProcessor


class Collector:
    def __init__(self, envs, agent, device, seed, *, sensor_config, total_timesteps):
        self.envs, self.agent, self.device = envs, agent, device
        self.sensor_config, self.total_timesteps = sensor_config, total_timesteps
        self.sensor_processor = SensorProcessor(seed)
        self.global_step = 0
        self.num_envs = envs.num_envs
        observations, _ = envs.reset(seed=[(seed + i) % 2**32 for i in range(self.num_envs)])
        self.raw_observations = np.array(observations, copy=True)
        self.observations = self.process_observation(observations)
        self.previous_actions = torch.full((self.num_envs,), -1, dtype=torch.long, device=device)
        self.episode_starts = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        self.history = History(agent.config.warmup_length)
        self.cache = None
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float64)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int64)

    @property
    def failure_probability(self):
        return self.sensor_config.failure_probability(self.global_step, self.total_timesteps)

    def process_observation(self, raw_observations):
        processed = self.sensor_processor.process(raw_observations, self.failure_probability)
        return torch.as_tensor(processed, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def peek_values(self, observations, previous_actions, env_indices):
        """给真实后继状态估值，不把它写入在线缓存，也不重复推进采样位置。

        用同一输入构造函数、历史和位置编号做并行计算。调用者传入刚执行的动作，
        即 successor 的 previous_action；时间截断的 final observation 不标为新回合。
        """
        indices = torch.as_tensor(env_indices, dtype=torch.long, device=self.device)
        count = len(indices)
        history = self.history.tokens
        obs = torch.stack([token.observations[indices] for token in history], dim=1)
        actions = torch.stack([token.previous_actions[indices] for token in history], dim=1)
        starts = torch.stack([token.episode_starts[indices] for token in history], dim=1)
        positions = torch.stack([token.position_ids[indices] for token in history], dim=1)
        obs = torch.cat((obs, observations[:, None]), dim=1)
        actions = torch.cat((actions, previous_actions[:, None]), dim=1)
        starts = torch.cat((starts, torch.zeros(count, 1, dtype=torch.bool, device=self.device)), dim=1)
        positions = torch.cat((positions, self.cache.next_positions[indices, None]), dim=1)
        valid = torch.ones(obs.shape[:2], dtype=torch.bool, device=self.device)
        _, values = self.agent.forward_sequence(
            obs, actions, episode_starts=starts, valid_mask=valid, position_ids=positions,
        )
        return values[:, -1]

    @torch.no_grad()
    def collect(self, num_steps):
        self.agent.eval()
        # 使用原样保存的策略输入重建缓存；不能按新课程进度重新遮蔽历史。
        self.cache = self.history.rebuild(self.agent, self.num_envs)
        buffer = RolloutBuffer(
            num_steps, self.num_envs, self.device, self.history.snapshot(), self.history.capacity,
        )
        buffer.sensor_probability_start = self.failure_probability
        for t in range(num_steps):
            positions = torch.where(self.episode_starts, 0, self.cache.next_positions).clone()
            token = Token(self.observations, self.previous_actions, self.episode_starts, positions)
            logits, values = self.agent.step_outputs(
                self.observations, self.previous_actions, self.cache, self.episode_starts,
            )
            actions, log_probs = self.agent.sample(logits)
            # 普通 transition 的后继价值就是下一次实际采样时的价值，避免重复推缓存。
            if t:
                continuing = ~(buffer.terminated[t - 1] | buffer.truncated[t - 1])
                buffer.next_values[t - 1, continuing] = values[continuing]
            self.history.append(token)
            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions.cpu().numpy())
            term = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
            trunc = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)
            buffer.add(token, actions, log_probs, values,
                       torch.as_tensor(rewards, dtype=torch.float32, device=self.device), term, trunc,
                       logits=logits)

            # 后继观测属于新时刻；final 和 reset 均使用该时刻的概率，各自只抽样一次。
            self.global_step += self.num_envs

            timeout_indices = np.flatnonzero(truncated & ~terminated)
            if len(timeout_indices):
                if "final_obs" not in infos:
                    raise RuntimeError("SAME_STEP 截断缺少 final_obs，不能用 reset observation 代替")
                final_obs = self.process_observation(np.stack([infos["final_obs"][i] for i in timeout_indices]))
                indices = torch.as_tensor(timeout_indices, dtype=torch.long, device=self.device)
                buffer.next_values[t, indices] = self.peek_values(final_obs, actions[indices], indices)
            # terminated 的 next_values 保持零，即使同时标了 truncated 也优先按真正终止处理。
            self.episode_returns += rewards
            self.episode_lengths += 1
            done = terminated | truncated
            for env in np.flatnonzero(done):
                buffer.episodes.append({
                    "global_step": self.global_step, "env_index": int(env),
                    "return": float(self.episode_returns[env]), "length": int(self.episode_lengths[env]),
                    "terminated": bool(terminated[env]), "truncated": bool(truncated[env]),
                })
                self.episode_returns[env], self.episode_lengths[env] = 0, 0
            self.raw_observations = np.array(next_obs, copy=True)
            self.observations = self.process_observation(next_obs)
            self.episode_starts = torch.as_tensor(done, dtype=torch.bool, device=self.device)
            self.previous_actions = actions.clone().masked_fill(self.episode_starts, -1)

        continuing = ~(buffer.terminated[-1] | buffer.truncated[-1])
        indices = continuing.nonzero(as_tuple=True)[0]
        if len(indices):
            buffer.next_values[-1, indices] = self.peek_values(
                self.observations[indices], self.previous_actions[indices], indices,
            )
        buffer.sensor_probability_end = self.failure_probability
        return buffer

    def invalidate_cache(self):
        self.cache = None

    def refresh_pending_observation(self):
        """Only at a curriculum transition, before acting on the pending raw state."""
        self.observations = self.process_observation(self.raw_observations)
