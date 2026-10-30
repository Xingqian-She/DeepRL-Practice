"""训练诊断和独立评估；不修改奖励、优势或训练环境。"""

import gymnasium as gym
import numpy as np
import torch


@torch.no_grad()
def rollout_metrics(buffer, trainer):
    ext = trainer.ext_coef * buffer.adv_ext.flatten()
    intrinsic = trainer.int_coef * buffer.adv_int.flatten()
    obs = buffer.rnd_obs.reshape(-1, 2)
    ext_std = ext.std(unbiased=False).item()
    int_std = intrinsic.std(unbiased=False).item()
    return {
        "adv_ext_weighted_std": ext_std,
        "adv_int_weighted_std": int_std,
        "adv_int_ext_std_ratio": int_std / (ext_std + 1e-8),
        "adv_ext_mean": ext.mean().item(),
        "adv_int_mean": intrinsic.mean().item(),
        "value_ext_mean": buffer.values_ext.mean().item(),
        "position_min": obs[:, 0].min().item(),
        "position_max": obs[:, 0].max().item(),
        "speed_abs_max": obs[:, 1].abs().max().item(),
        "action_left_fraction": (buffer.actions == 0).float().mean().item(),
        "action_idle_fraction": (buffer.actions == 1).float().mean().item(),
        "action_right_fraction": (buffer.actions == 2).float().mean().item(),
    }


@torch.no_grad()
def evaluate(agent, device, seed=10000, episodes=20, deterministic=True):
    """固定独立种子，分别支持 argmax 和采样策略；使用局部 RNG 上下文。"""
    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make("MountainCar-v0") for _ in range(episodes)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    returns = np.zeros(episodes)
    successes = np.zeros(episodes, dtype=bool)
    finished = np.zeros(episodes, dtype=bool)
    position_max = np.full(episodes, -np.inf)
    try:
        obs, _ = envs.reset(seed=seed)
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            for _ in range(200):
                logits, _, _ = agent(torch.as_tensor(obs, dtype=torch.float32, device=device))
                actions = (logits.argmax(-1) if deterministic else
                           torch.distributions.Categorical(logits=logits).sample())
                obs, rewards, terminated, truncated, infos = envs.step(actions.cpu().numpy())
                active = ~finished
                returns[active] += rewards[active]
                actual = obs.copy()
                for i in np.flatnonzero(terminated | truncated):
                    actual[i] = infos["final_obs"][i]
                position_max[active] = np.maximum(position_max[active], actual[active, 0])
                successes[active] |= terminated[active]
                finished |= terminated | truncated
                if finished.all():
                    break
        return {"return_mean": float(returns.mean()),
                "success_rate": float(successes.mean()),
                "position_max_mean": float(position_max.mean()),
                "episodes": episodes, "seed": seed}
    finally:
        envs.close()
