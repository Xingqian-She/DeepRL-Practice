"""训练入口。按采样 → 奖励缩放 → GAE → 网络更新 → 统计/输出的顺序阅读。"""

from pathlib import Path
import argparse
import json
import random
import time

import gymnasium as gym
import numpy as np
import torch

from model import PPOActorCritic, RNDTargetPredictor
from buffer import RolloutBuffer
from ppo_rnd import PPOTrainer
from rnd import (
    RunningMeanStd, RewardForwardFilter, normalize_rnd_obs,
    compute_intrinsic_reward, scale_intrinsic_rewards,
)
from rollout import make_env, get_transition_next_info, warmup_observation_stats, collect_rollout
from reporting import EpisodeTracker, moving_average, save_training_logs, print_progress
from checkpoint import save_checkpoint
from diagnostics import rollout_metrics, evaluate

# 训练规模和运行位置；PPO 超参数在 ppo_rnd.py，网络宽度在 model.py。
ENV_ID = "MountainCar-v0"
NUM_ENVS = 4
NUM_STEPS = 128
TOTAL_TIMESTEPS = 500000
INT_GAMMA = 0.99
WARMUP_STEPS = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN_DIR = Path(__file__).resolve().parent / "runs"
LOG_INTERVAL = 10
TRAINER_KWARGS = {}
ARCHITECTURE = "residual"
SEED = None
PLOT_INTERVAL = 100
EVAL_INTERVAL = 0
NORMALIZE_POLICY_INPUT = False
EXT_REWARD_SCALE = 1.0


def train():
    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
    checkpoint_dir = RUN_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_ID) for _ in range(NUM_ENVS)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    try:
        _train(envs, checkpoint_dir)
    finally:
        envs.close()


def _train(envs, checkpoint_dir):
    obs_dim = envs.single_observation_space.shape[0]
    started = time.monotonic()
    agent = PPOActorCritic(obs_dim, envs.single_action_space.n, architecture=ARCHITECTURE,
                           normalize_input=NORMALIZE_POLICY_INPUT).to(DEVICE)
    rnd_model = RNDTargetPredictor(obs_dim).to(DEVICE)
    trainer = PPOTrainer(agent, rnd_model, **TRAINER_KWARGS)
    buffer = RolloutBuffer(NUM_STEPS, NUM_ENVS, obs_dim, DEVICE)
    obs_rms = RunningMeanStd(shape=(obs_dim,))
    int_rew_rms = RunningMeanStd(shape=())
    int_reward_filter = RewardForwardFilter(INT_GAMMA, NUM_ENVS)
    episode_tracker = EpisodeTracker(NUM_ENVS)
    history = []

    if SEED is not None:
        envs.reset(seed=SEED)
        for i, env in enumerate(envs.envs):
            env.action_space.seed(SEED + i)
    warmup_observation_stats(envs, obs_rms, WARMUP_STEPS)
    obs, _ = envs.reset(seed=SEED)
    obs = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
    num_updates = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)

    for update in range(1, num_updates + 1):
        # 1. 固定网络和观测统计量，采集 128 × 4 条 transition。
        obs = collect_rollout(
            envs, agent, trainer.rnd_model, buffer, obs, obs_rms, episode_tracker,
            start_step=(update - 1) * NUM_STEPS * NUM_ENVS,
            reward_fn=compute_intrinsic_reward,
        )
        # 2. 缩放内在奖励，再分别计算两路优势和 critic 目标。
        reward_logs = scale_intrinsic_rewards(buffer, int_reward_filter, int_rew_rms)
        # 仅改变训练奖励单位，EpisodeTracker 仍记录原始环境回报。
        buffer.rewards_ext *= EXT_REWARD_SCALE
        buffer.compute_ext_returns_and_advantages(
            gamma=trainer.gamma, lmbda=trainer.ext_lambda
        )
        buffer.compute_int_returns_and_advantages(
            gamma=INT_GAMMA, lmbda=trainer.int_lambda
        )
        # 3. 用固定目标更新 PPO 和 RND，再更新下一轮的观测统计量。
        diagnostic_logs = rollout_metrics(buffer, trainer)
        logs = trainer.update(buffer, obs_rms=obs_rms)
        obs_rms.update(buffer.rnd_obs.cpu().numpy().reshape(-1, obs_dim))

        # 4. 记录训练损失和实际回合表现。
        history.append(
            {
                "update": update,
                "global_step": update * NUM_STEPS * NUM_ENVS,
                **logs,
                **episode_tracker.metrics(),
                **reward_logs,
                **diagnostic_logs,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if update % LOG_INTERVAL == 0 or update == num_updates:
            save_training_logs(history, RUN_DIR, episode_tracker.episodes,
                               plot=update % PLOT_INTERVAL == 0 or update == num_updates)
            print_progress(update, num_updates, {**logs, **diagnostic_logs}, episode_tracker.metrics())
        if EVAL_INTERVAL and (update % EVAL_INTERVAL == 0 or update == num_updates):
            evaluation = {mode: evaluate(agent, DEVICE, deterministic=mode == "deterministic")
                          for mode in ["deterministic", "stochastic"]}
            evaluation["global_step"] = update * NUM_STEPS * NUM_ENVS
            with (RUN_DIR / "evaluations.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(evaluation) + "\n")
            print("EVAL", json.dumps(evaluation), flush=True)

    model_path = checkpoint_dir / "ppo_rnd_model.pth"
    save_checkpoint(
        model_path, agent, rnd_model, obs_rms, int_rew_rms, int_reward_filter
    )
    print(f"训练完成！日志和曲线保存在 {RUN_DIR}，模型保存在 {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=DEVICE.type)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--architecture", choices=["residual", "mlp"], default=ARCHITECTURE)
    parser.add_argument("--normalize-policy-input", action="store_true")
    parser.add_argument("--ext-reward-scale", type=float, default=1.0)
    parser.add_argument("--rnd-lr", type=float, default=4e-5)
    parser.add_argument("--int-coef", type=float, default=1.0)
    parser.add_argument("--ext-coef", type=float, default=2.0)
    parser.add_argument("--int-lambda", type=float, default=0.85)
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    SEED, TOTAL_TIMESTEPS, RUN_DIR = args.seed, args.total_timesteps, args.run_dir
    DEVICE, ARCHITECTURE = torch.device(args.device), args.architecture
    NUM_ENVS, NUM_STEPS, EVAL_INTERVAL = args.num_envs, args.num_steps, args.eval_interval
    NORMALIZE_POLICY_INPUT, EXT_REWARD_SCALE = args.normalize_policy_input, args.ext_reward_scale
    TRAINER_KWARGS = dict(rnd_lr=args.rnd_lr, int_coef=args.int_coef,
                          ext_coef=args.ext_coef, int_lambda=args.int_lambda,
                          compile_models=not args.no_compile)
    torch.set_num_threads(args.threads)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config["torch_version"] = torch.__version__
    (RUN_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (RUN_DIR / "evaluations.jsonl").write_text("", encoding="utf-8")
    train()
