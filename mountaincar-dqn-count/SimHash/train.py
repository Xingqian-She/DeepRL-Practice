"""练习 4：将前面模块接成单环境训练循环。运行方式见 README。"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import uuid

import gymnasium as gym
import numpy as np
import torch

from config import Config
from dqn import epsilon_at, select_action, update
from exploration import SimHashCounter
from model import ResidualQNetwork
from replay import ReplayBuffer


def train(config, run_dir, device):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(config), "device": str(device)}, indent=2), encoding="utf-8")
    env = gym.make(config.env_id)
    try:
        policy = ResidualQNetwork().to(device)
        target = ResidualQNetwork().to(device)
        target.load_state_dict(policy.state_dict())
        target.eval()
        target.requires_grad_(False)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=0.0)
        replay = ReplayBuffer(config.capacity, config.seed)
        counter = SimHashCounter(config.n_bits, config.seed, config.append_constant)
        obs, _ = env.reset(seed=config.seed)
        episode_return, episode_bonus, episode_length = 0.0, 0.0, 0
        with (run_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as log:
            writer = csv.DictWriter(log, fieldnames=[
                "step", "return_ext", "bonus_sum", "length", "success", "unique_buckets"])
            writer.writeheader()
            for step in range(config.total_steps):
                epsilon = epsilon_at(step, config)
                # TODO 4a：调用 select_action，env.step，拿到真实 next_obs。
                # 然后 counter.observe(next_obs)（beta=0 时也统计覆盖范围），
                # reward_train = ext_reward_scale * reward_ext + beta * bonus。
                # replay.push(..., terminated)，不要传 terminated or truncated。
                # 最后更新三个回合统计量；return_ext 必须累加原始环境奖励。
                # 本循环不是向量环境，没有自动 reset，next_obs 就是最终观测。
                # raise NotImplementedError("TODO 4a: 交互、计数、混合奖励、入池、累计统计")
                action = select_action(policy, obs, epsilon, rng, env.action_space.n, device)
                next_obs, reward_ext, terminated, truncated, info = env.step(action)
                bonus, n = counter.observe(next_obs)
                reward_train = config.ext_reward_scale * reward_ext + config.beta * bonus
                replay.push(obs, action, reward_train, next_obs, terminated)
                episode_return += reward_ext
                episode_bonus += bonus
                episode_length += 1

                # 已提供：每个环境步至多一次梯度更新。
                if step + 1 >= config.learning_starts and len(replay) >= config.batch_size:
                    update(policy, target, optimizer, replay, config, device)
                if (step + 1) % config.target_update_interval == 0:
                    target.load_state_dict(policy.state_dict())

                # TODO 4b：先 obs = next_obs，再处理回合结束。
                # if terminated or truncated:
                #   writer.writerow({...})，键名参考上面的 fieldnames；log.flush()
                #   success 用 bool(terminated)，本教程环境到达目标才 terminated。
                #   打印原始回报、长度、累计 bonus、len(counter.counts)。
                #   obs, _ = env.reset()；三个回合统计量清零。
                # 注意：不要重建 counter / replay，也不要重设 seed。
                # raise NotImplementedError("TODO 4b: 后继状态、结束回合、记录和重置")
                obs = next_obs
                if terminated or truncated:
                    writer.writerow({
                        "step": step + 1,
                        "return_ext": episode_return,
                        "bonus_sum": episode_bonus,
                        "length": episode_length,
                        "success": int(terminated),
                        "unique_buckets": len(counter.counts)
                    })
                    log.flush()
                    print(f"Step {step+1:6d} | Return: {episode_return:5.1f} | "
                          f"Length: {episode_length:3d} | Buckets: {len(counter.counts):4d}")
                    # 只在首次 reset 时设 seed；之后继续使用环境 RNG，
                    # 否则每回合都会回到完全相同的初始位置。
                    obs, _ = env.reset()
                    episode_return, episode_bonus, episode_length = 0.0, 0.0, 0
        # 已提供：用于独立评估，不作为断点续训文件。
        # 若要续训，还需保存 replay、计数表及投影、optimizer、RNG 和环境状态。
        torch.save({"model_config": policy.model_config,
                    "state_dict": policy.state_dict(), "config": asdict(config)},
                   run_dir / "q_network.pt")
        print(f"完成 {config.total_steps} 步，输出：{run_dir}")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--append-constant", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.steps <= 0 or args.beta < 0:
        parser.error("steps 必须为正，beta 必须非负")
    config = Config(seed=args.seed, total_steps=args.steps, beta=args.beta,
                    append_constant=args.append_constant)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_dir or Path(__file__).resolve().parent / "runs" / (
        f"{stamp}-seed{args.seed}-{uuid.uuid4().hex[:6]}")
    train(config, run_dir, torch.device(args.device))


if __name__ == "__main__":
    main()
