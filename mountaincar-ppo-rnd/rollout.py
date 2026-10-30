"""环境交互：预热、真实后继状态还原、固定长度采样。"""

import gymnasium as gym
import numpy as np
import torch

from rnd import compute_intrinsic_reward


def make_env(env_id):
    def thunk():
        env = gym.make(env_id)
        return env

    return thunk


@torch.no_grad()
def get_transition_next_info(agent, next_obs, terminations, truncations, infos):
    """获取真实的后继状态 transition_next_obs，以及对应的外在/内在后继价值。

    1. 外在价值 (next_v_ext): 需还原被自动 reset 覆盖的 final_obs，且在终止(terminated)时置 0。
    2. 内在价值 (next_v_int): 好奇心跨回合连续传播，因此直接使用 reset 后的 next_obs，且不置 0。
    """
    transition_next_obs = next_obs.clone()
    # 只要发生了自动重置 (terminated 或 truncated)，真实踏入的后继状态都在 final_obs 中
    done_indices = np.flatnonzero(terminations | truncations)
    if len(done_indices) > 0:
        indices = torch.as_tensor(
            done_indices, dtype=torch.long, device=next_obs.device
        )
        transition_next_obs[indices] = torch.as_tensor(
            np.stack([infos["final_obs"][i] for i in done_indices]),
            dtype=next_obs.dtype,
            device=next_obs.device,
        )

    # 1. 计算外在后继价值（使用真实后继状态 transition_next_obs）
    _, next_v_ext, _ = agent(transition_next_obs)
    terminated = torch.as_tensor(
        terminations, dtype=torch.bool, device=next_obs.device
    )
    next_v_ext = next_v_ext.masked_fill(terminated, 0.0)

    # 2. 计算内在后继价值（跨回合不断开，直接使用下一步即将采样的 next_obs）
    _, _, next_v_int = agent(next_obs)

    return transition_next_obs, next_v_ext, next_v_int


def warmup_observation_stats(envs, obs_rms, num_steps):
    """随机交互预热 RND 输入统计量，结束后由调用者重新 reset。"""
    envs.reset()
    observations = []
    for _ in range(num_steps):
        actions = np.array([
            envs.single_action_space.sample() for _ in range(envs.num_envs)
        ])
        next_obs, _, terminated, truncated, infos = envs.step(actions)
        rnd_obs = next_obs.copy()
        for i in np.flatnonzero(terminated | truncated):
            rnd_obs[i] = infos["final_obs"][i]
        observations.append(rnd_obs)
    obs_rms.update(np.concatenate(observations, axis=0))


@torch.no_grad()
def collect_rollout(
    envs, agent, rnd_model, buffer, obs, obs_rms, episode_tracker,
    start_step, reward_fn=compute_intrinsic_reward,
):
    """收满一个 rollout，返回下一次采样状态；期间不改变网络和 obs_rms。

    obs 是采取动作前的状态；rnd_obs 是动作真正到达的后继状态。
    start_step 是此前正式训练的环境交互总数（包含所有并行环境）。
    """
    for step in range(buffer.num_steps):
        action, logprob, v_ext, v_int = agent.act(obs)
        next_obs_np, rewards_ext, terminated, truncated, infos = envs.step(
            action.cpu().numpy()
        )
        global_step = start_step + (step + 1) * envs.num_envs
        episode_tracker.update(rewards_ext, terminated, truncated, global_step)
        next_obs = torch.as_tensor(next_obs_np, dtype=obs.dtype, device=obs.device)
        transition_next_obs, next_v_ext, next_v_int = get_transition_next_info(
            agent, next_obs, terminated, truncated, infos
        )
        rewards_int = reward_fn(rnd_model, transition_next_obs, obs_rms)
        buffer.add(
            obs=obs, rnd_obs=transition_next_obs,
            action=action, logprob=logprob,
            reward_ext=torch.as_tensor(rewards_ext, dtype=obs.dtype, device=obs.device),
            reward_int=rewards_int,
            value_ext=v_ext, next_value_ext=next_v_ext,
            value_int=v_int, next_value_int=next_v_int,
            terminated=torch.as_tensor(terminated, dtype=torch.bool, device=obs.device),
            truncated=torch.as_tensor(truncated, dtype=torch.bool, device=obs.device),
        )
        obs = next_obs
    return obs
