"""Reusable policy-only evaluator with episode-local random streams and batched inference."""

from copy import deepcopy
import time

import numpy as np
import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from envs import make_env
from model import make_model_input
from runtime import EVALUATION_PROTOCOL, preserve_rng_state
from sensors import OBSERVATION_DIM, SENSOR_NAMES, SensorProcessor, validate_probability


class PolicySnapshot(nn.Module):
    """An eager actor copy; no critic, value backbone, or training compiled functions."""
    def __init__(self, agent):
        super().__init__()
        self.config = agent.config
        self.encoder = deepcopy(agent.encoder)
        self.backbone = deepcopy(agent.backbone)
        self.actor = deepcopy(agent.actor)
        self.eval()

    def sync(self, agent):
        if self.config != agent.config:
            raise ValueError("评估器只能复用相同结构的模型")
        for name in ("encoder", "backbone", "actor"):
            getattr(self, name).load_state_dict(getattr(agent, name).state_dict(), strict=True)

    @torch.no_grad()
    def step(self, features, cache, starts):
        with sdpa_kernel(SDPBackend.MATH):
            hidden = self.backbone.step(self.encoder(features).unsqueeze(1), cache, episode_starts=starts)
            return self.actor(hidden[:, 0])


def episode_outcome(terminated, truncated, reward):
    if truncated and not terminated:
        return "timeout"
    if reward == -100:
        return "terminal_penalty"
    if reward == 100:
        return "positive_terminal"
    return "terminated_other"


def summarize_episodes(rows, probability, policy, entropy_sum, main_sum):
    returns = np.asarray([row["return"] for row in rows], dtype=np.float64)
    steps = sum(row["length"] for row in rows)
    counts = np.asarray([sum(row[f"action_count_{i}"] for row in rows) for i in range(4)])
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "sensor_failure_probability": probability, "policy": policy, "episodes": rows,
        "mean_return": float(returns.mean()), "std_return": float(returns.std()),
        "mean_length": float(np.mean([row["length"] for row in rows])),
        "score_200_rate": float(np.mean(returns >= 200)),
        "terminal_penalty_rate": float(np.mean([row["terminal_reward"] == -100 for row in rows])),
        "positive_terminal_rate": float(np.mean([row["outcome"] == "positive_terminal" for row in rows])),
        "timeout_rate": float(np.mean([row["outcome"] == "timeout" for row in rows])),
        "terminated_rate": float(np.mean([row["terminated"] for row in rows])),
        "mean_terminal_reward": float(np.mean([row["terminal_reward"] for row in rows])),
        "policy_entropy": entropy_sum / steps, "main_engine_probability": main_sum / steps,
        **{f"return_p{q}": float(np.percentile(returns, q)) for q in (10, 50, 90)},
        **{f"action_fraction_{i}": float(counts[i] / steps) for i in range(4)},
    }


class EvaluationRunner:
    def __init__(self, agent, *, num_envs=8, max_episode_steps=1000, env_factory=None,
                 mode="none", video_dir=None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("评估 num_envs 必须是正整数")
        if type(max_episode_steps) is not int or max_episode_steps < 1:
            raise ValueError("max_episode_steps 必须是正整数")
        if mode not in ("none", "human", "video"):
            raise ValueError("无效的评估 mode")
        if mode == "video" and video_dir is None:
            raise ValueError("video 模式需要 video_dir")
        self.device = next(agent.parameters()).device
        self.num_envs = num_envs if mode == "none" else 1
        self.max_episode_steps, self.factory = max_episode_steps, env_factory or make_env
        self.mode, self.video_dir = mode, video_dir
        with preserve_rng_state(self.device):
            self.policy = PolicySnapshot(agent).to(device=self.device, dtype=torch.float32)
        self.envs = []
        self.last_performance = {}

    def _ensure_envs(self, count):
        for _ in range(len(self.envs), count):
            env = self.factory(self.max_episode_steps, {"none": None, "human": "human", "video": "rgb_array"}[self.mode])
            try:
                if self.mode == "video":
                    import gymnasium as gym
                    env = gym.wrappers.RecordVideo(env, str(self.video_dir), episode_trigger=lambda _: True)
                self.envs.append(env)
            except BaseException:
                env.close()
                raise

    @torch.no_grad()
    def evaluate(self, agent, *, episodes=20, seed=10000, sensor_failure_probability=1.0,
                 policy="sample", on_episode=None):
        if type(episodes) is not int or episodes < 1 or type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("episodes 必须为正整数，seed 必须在 [0, 2**32) 内")
        if policy not in ("argmax", "sample"):
            raise ValueError("无效的评估 policy")
        validate_probability(sensor_failure_probability)
        started = time.perf_counter()
        with preserve_rng_state(self.device):
            try:
                self.policy.sync(agent)
                count = min(self.num_envs, episodes)
                self._ensure_envs(count)
                cache = self.policy.backbone.create_cache(count)
                observations = np.zeros((count, OBSERVATION_DIM), dtype=np.float32)
                previous = np.full(count, -1, dtype=np.int64)
                starts = np.ones(count, dtype=bool)
                slots = [None] * count
                next_episode = 0
                results = []
                entropy_sum, main_sum = 0.0, 0.0

                def assign(slot):
                    nonlocal next_episode
                    if next_episode == episodes:
                        slots[slot] = None
                        observations[slot] = 0
                        previous[slot], starts[slot] = -1, True
                        return
                    episode = next_episode
                    next_episode += 1
                    episode_seed = (seed + episode) % 2**32
                    raw, _ = self.envs[slot].reset(seed=episode_seed)
                    sensors = SensorProcessor(episode_seed)
                    slots[slot] = {
                        "episode": episode, "seed": episode_seed, "sensors": sensors,
                        "actions_rng": np.random.default_rng(np.random.SeedSequence([episode_seed, 0x4143544E])),
                        "return": 0.0, "length": 0, "counts": np.zeros(4, dtype=np.int64),
                    }
                    observations[slot] = sensors.process(raw, sensor_failure_probability)
                    previous[slot], starts[slot] = -1, True

                for slot in range(count):
                    assign(slot)
                while len(results) < episodes:
                    # Validate/encode on CPU, then one H2D observation and one D2H probability batch.
                    features = make_model_input(torch.from_numpy(observations), torch.from_numpy(previous),
                                                self.policy.config.use_previous_action).to(self.device)
                    logits = self.policy.step(features, cache, torch.from_numpy(starts).to(self.device))
                    probabilities = logits.softmax(-1).cpu().numpy().astype(np.float64)
                    if not np.isfinite(probabilities).all() or (probabilities.sum(axis=-1) <= 0).any():
                        raise RuntimeError("评估动作概率出现非有限值或零和")
                    probabilities /= probabilities.sum(axis=-1, keepdims=True)
                    for slot, state in enumerate(slots):
                        if state is None:
                            continue
                        probs = probabilities[slot]
                        if policy == "argmax":
                            action = int(probs.argmax())
                        else:
                            cdf = probs.cumsum()
                            cdf[-1] = 1.0
                            action = int(np.searchsorted(cdf, state["actions_rng"].random(), side="right"))
                        nonzero = probs > 0
                        entropy_sum -= float((probs[nonzero] * np.log(probs[nonzero])).sum())
                        main_sum += float(probs[2])
                        raw, reward, terminated, truncated, _ = self.envs[slot].step(action)
                        state["return"] += float(reward)
                        state["length"] += 1
                        state["counts"][action] += 1
                        if terminated or truncated:
                            row = {"episode": state["episode"], "seed": state["seed"],
                                   "return": state["return"], "length": state["length"],
                                   "terminated": bool(terminated), "truncated": bool(truncated),
                                   "terminal_reward": float(reward),
                                   "outcome": episode_outcome(terminated, truncated, reward),
                                   **{f"final_{name}": float(raw[i]) for i, name in enumerate(SENSOR_NAMES)},
                                   **{f"action_count_{i}": int(state["counts"][i]) for i in range(4)}}
                            results.append(row)
                            if on_episode is not None:
                                on_episode(row)
                            assign(slot)
                        else:
                            observations[slot] = state["sensors"].process(raw, sensor_failure_probability)
                            previous[slot], starts[slot] = action, False
                results.sort(key=lambda row: row["episode"])
                summary = summarize_episodes(results, sensor_failure_probability, policy, entropy_sum, main_sum)
                elapsed = time.perf_counter() - started
                steps = sum(row["length"] for row in results)
                self.last_performance = {"evaluation_seconds": elapsed, "evaluation_environment_steps": steps,
                                         "evaluation_steps_per_second": steps / elapsed,
                                         "evaluation_num_envs": count}
                return summary
            except BaseException:
                self.close()
                raise

    def close(self):
        envs, self.envs = self.envs, []
        first_error = None
        for env in envs:
            try:
                env.close()
            except BaseException as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error
