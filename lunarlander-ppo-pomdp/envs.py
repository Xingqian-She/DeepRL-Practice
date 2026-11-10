"""原始八维环境；传感器失灵和动作历史由采样器显式处理。"""

import gymnasium as gym
from config import ENV_ID


def make_env(max_episode_steps=1000, render_mode=None):
    return gym.make(
        ENV_ID, continuous=False, enable_wind=False,
        max_episode_steps=max_episode_steps, render_mode=render_mode,
    )


def make_vector_env(num_envs, max_episode_steps=1000):
    return gym.vector.SyncVectorEnv(
        [lambda: make_env(max_episode_steps) for _ in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
