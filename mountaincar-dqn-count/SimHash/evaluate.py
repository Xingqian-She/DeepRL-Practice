"""已提供：独立环境、贪心动作、原始回报；不创建或更新 hash 计数。"""

import argparse

import gymnasium as gym
import numpy as np
import torch

from model import ResidualQNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes 必须为正")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = ResidualQNetwork(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    env = gym.make(checkpoint["config"]["env_id"],
                   render_mode="human" if args.render else None)
    returns, successes = [], []
    try:
        with torch.no_grad():
            for episode in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + episode)
                total = 0.0
                while True:
                    q = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                    obs, reward, terminated, truncated, _ = env.step(q.argmax(1).item())
                    total += reward
                    if terminated or truncated:
                        returns.append(total)
                        successes.append(terminated)
                        break
    finally:
        env.close()
    print(f"episodes={args.episodes}, mean_return_ext={np.mean(returns):.2f}, "
          f"success_rate={np.mean(successes):.1%}")


if __name__ == "__main__":
    main()
