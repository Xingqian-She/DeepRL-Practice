"""Reproduce coordinate-step calibration; no training or held-out evaluation seeds.

Run in the pinned project image from the repository root:
python -B lunarlander-ppo-pomdp/reports/scripts/calibrate_xy_fourier.py
"""

import argparse
import importlib.metadata
import json
from pathlib import Path

import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import heuristic
import numpy as np


def calibrate():
    results = {}
    for strategy in ("random", "heuristic", "main_engine"):
        deltas = []
        for seed in range(30000, 30256):
            env = gym.make("LunarLander-v3", continuous=False, enable_wind=False, max_episode_steps=1000)
            try:
                obs, _ = env.reset(seed=seed)
                rng = np.random.default_rng(seed)
                for _ in range(1000):
                    action = (int(rng.integers(4)) if strategy == "random" else
                              2 if strategy == "main_engine" else heuristic(env, obs))
                    successor, _, terminated, truncated, _ = env.step(action)
                    deltas.append(np.abs(successor[:2] - obs[:2]))
                    obs = successor
                    if terminated or truncated:
                        break
            finally:
                env.close()
        delta = np.asarray(deltas)
        results[strategy] = {"transitions": len(delta),
                             "abs_delta_q50_q99_q999_max": np.quantile(delta, [.5, .99, .999, 1], axis=0).tolist(),
                             "selected_max_phase_over_pi": (delta.max(axis=0) * [8, 2]).tolist(),
                             "next_octave_max_phase_over_pi": (delta.max(axis=0) * [16, 4]).tolist()}
    return {"seed_start": 30000, "seed_count_per_strategy": 256, "axes": ["x", "y"],
            "environment": "LunarLander-v3; discrete; no wind; max 1000 steps",
            "dependencies": {name: importlib.metadata.version(name) for name in ("gymnasium", "numpy", "box2d-py")},
            "strategies": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(calibrate(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
