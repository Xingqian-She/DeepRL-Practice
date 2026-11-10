"""确定性八维测试环境；模型和 PPO 使用真实实现。"""

from dataclasses import replace

import gymnasium as gym
import numpy as np

from config import AgentConfig, TrainConfig


def small_config(**changes):
    return replace(TrainConfig(
        agent=AgentConfig(hidden_dim=32, num_layers=2, num_heads=2,
                          context_length=4, alpha_init=0.7),
        total_timesteps=52, num_envs=2, num_steps=13, sequence_length=4,
        sequences_per_minibatch=3, update_epochs=2, max_episode_steps=5,
        save_interval=1, seed=3, device="cpu",
    ), **changes)


class FixtureEnv(gym.Env):
    """每回合更换 x，y 记录步数，使 final/reset observation 明显不同。"""
    metadata = {"render_modes": []}

    def __init__(self, limit=5, terminate=False, index=0, both=False):
        self.limit, self.terminate, self.index, self.both = limit, terminate, index, both
        self.observation_space = gym.spaces.Box(-1000, 1000, (8,), np.float32)
        self.action_space = gym.spaces.Discrete(4)
        self.episode, self.steps = -1, 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode = 0 if seed is not None else self.episode + 1
        self.steps = 0
        return self._observation(), {}

    def _observation(self):
        return np.array([self.index + self.episode * 10, self.steps, self.steps + 0.5,
                         -0.25, 0.4, -0.3, self.steps % 2, (self.steps + 1) % 2], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.limit
        return (self._observation(), 1.0, done and (self.terminate or self.both),
                done and (not self.terminate or self.both), {})


def fixture_vector(limits=(3, 5), terminations=(True, False), both=False):
    return gym.vector.SyncVectorEnv(
        [lambda limit=limit, terminate=terminate, index=index: FixtureEnv(limit, terminate, index, both)
         for index, (limit, terminate) in enumerate(zip(limits, terminations))],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
