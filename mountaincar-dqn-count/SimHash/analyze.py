"""对已有 checkpoint 做配对评估，保存逐回合数据、训练摘要和静态图。

所有模型使用相同评估 seed；批量前向以加速 CPU 评估，每个环境仍单独 step。
不训练，不更新计数，不覆盖已有输出目录。默认比较原来的三组 seed0 实验。
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from exploration import SimHashCounter
from model import ResidualQNetwork


def load_model(run_dir):
    path = run_dir / "q_network.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = ResidualQNetwork(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint["config"]


@torch.inference_mode()
def evaluate(model, config, start_seed, episodes, batch_size):
    rows = []
    for start in range(0, episodes, batch_size):
        seeds = list(range(start_seed + start, start_seed + min(start + batch_size, episodes)))
        envs = [gym.make(config["env_id"]) for _ in seeds]
        try:
            obs = np.stack([env.reset(seed=seed)[0] for env, seed in zip(envs, seeds)])
            initial = obs.copy()
            active = np.ones(len(seeds), dtype=bool)
            returns = np.zeros(len(seeds))
            lengths = np.zeros(len(seeds), dtype=int)
            min_gap = np.full(len(seeds), np.inf)
            max_position = obs[:, 0].copy()
            while active.any():
                indices = np.flatnonzero(active)
                q = model(torch.as_tensor(obs[indices], dtype=torch.float32))
                top2 = q.topk(2, dim=1).values
                gaps = (top2[:, 0] - top2[:, 1]).numpy()
                actions = q.argmax(1).tolist()
                for index, action, gap in zip(indices, actions, gaps):
                    next_obs, reward, terminated, truncated, _ = envs[index].step(action)
                    obs[index] = next_obs
                    returns[index] += reward
                    lengths[index] += 1
                    min_gap[index] = min(min_gap[index], gap)
                    max_position[index] = max(max_position[index], next_obs[0])
                    if terminated or truncated:
                        active[index] = False
                        rows.append(dict(seed=seeds[index], initial_position=float(initial[index, 0]),
                                         return_ext=float(returns[index]), length=int(lengths[index]),
                                         success=int(terminated), truncated=int(truncated),
                                         max_position=float(max_position[index]),
                                         min_q_gap=float(min_gap[index])))
        finally:
            for env in envs:
                env.close()
    return sorted(rows, key=lambda row: row["seed"])


def summary(rows):
    success = [row for row in rows if row["success"]]
    failure = [row for row in rows if not row["success"]]
    def mean(items, key):
        return float(np.mean([row[key] for row in items])) if items else None
    return dict(episodes=len(rows), successes=len(success), failures=len(failure),
                success_rate=len(success) / len(rows), mean_return_ext=mean(rows, "return_ext"),
                success_mean_length=mean(success, "length"), failure_mean_length=mean(failure, "length"),
                failure_initial_position_range=(
                    [min(row["initial_position"] for row in failure),
                     max(row["initial_position"] for row in failure)] if failure else None),
                failure_mean_max_position=mean(failure, "max_position"),
                return_quantiles=dict(zip(["min", "p25", "median", "p75", "max"],
                    np.quantile([row["return_ext"] for row in rows], [0, .25, .5, .75, 1]).tolist())))


def training_rows(run_dir):
    with (run_dir / "episodes.csv").open(encoding="utf-8", newline="") as file:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(file)]


def training_summary(rows, config):
    last = rows[-50:]
    first_success = next((row["step"] for row in rows if row["success"]), None)
    return dict(complete_episodes=len(rows), first_success_step=first_success,
                last50_success_rate=float(np.mean([row["success"] for row in last])),
                last50_mean_return=float(np.mean([row["return_ext"] for row in last])),
                final_unique_buckets=int(rows[-1]["unique_buckets"]),
                mean_weighted_bonus_per_step=config["beta"] * sum(row["bonus_sum"] for row in rows)
                    / sum(row["length"] for row in rows),
                last50_weighted_bonus_per_step=config["beta"] * sum(row["bonus_sum"] for row in last)
                    / sum(row["length"] for row in last))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_overview(results, training, output):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for label, rows in results.items():
        ordered = sorted(rows, key=lambda row: row["initial_position"])
        axes[0, 0].scatter([row["initial_position"] for row in ordered],
                            [row["length"] for row in ordered], s=9, alpha=.6, label=label)
        successes = sorted(row["length"] for row in rows if row["success"])
        # Unconditional CDF: its endpoint is success rate, not 100% by construction.
        axes[0, 1].step(successes, np.arange(1, len(successes) + 1) / len(rows),
                       where="post", label=label)
        train = training[label]
        returns = np.array([row["return_ext"] for row in train])
        window = min(30, len(returns))
        axes[1, 0].plot([row["step"] for row in train][window - 1:],
                       np.convolve(returns, np.ones(window) / window, mode="valid"), label=label)
        axes[1, 1].plot([row["step"] for row in train],
                       [row["unique_buckets"] for row in train], label=label)
    axes[0, 0].set(title="Same initial positions, different completion times",
                   xlabel="Initial position (velocity = 0)", ylabel="Episode length (cap = 200)")
    axes[0, 1].set(title="Fraction of ALL episodes that succeed by step t",
                   xlabel="Step t", ylabel="Cumulative successful fraction", ylim=(0, 1.02))
    axes[1, 0].set(title="Training: 30-episode rolling raw return",
                   xlabel="Environment steps", ylabel="Mean raw return")
    axes[1, 1].set(title="Training: cumulative visited hash buckets",
                   xlabel="Environment steps", ylabel="Visited buckets")
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    figure.savefig(output / "overview.png", dpi=160)
    plt.close(figure)


def plot_hash(configs, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    position, velocity = np.meshgrid(np.linspace(-1.2, .6, 241), np.linspace(-.07, .07, 241))
    scaled = np.stack([(position + .3) / .9, velocity / .07], axis=-1)
    info = {}
    for ax, append in zip(axes, [False, True]):
        # Use the configuration from the relevant run; these diagnostics are not counts.
        config = next(config for config in configs.values() if config["append_constant"] == append)
        counter = SimHashCounter(config["n_bits"], config["seed"], append)
        features = np.concatenate([scaled, np.ones((*scaled.shape[:2], 1))], axis=-1) if append else scaled
        projections = features @ counter.projection.T
        keys = np.packbits(projections >= 0, axis=-1).reshape(-1, (config["n_bits"] + 7) // 8)
        n_cells = len(np.unique(keys, axis=0))
        info[str(append)] = n_cells
        for bit in range(config["n_bits"]):
            values = projections[:, :, bit]
            if values.min() < 0 < values.max():
                ax.contour(position, velocity, values, levels=[0], colors="steelblue", linewidths=.6)
        ax.set(title=f"append_constant={append}; grid sees {n_cells} codes",
               xlabel="Position", ylabel="Velocity", xlim=(-1.2, .6), ylim=(-.07, .07))
    figure.savefig(output / "hash_boundaries.png", dpi=160)
    plt.close(figure)
    return info


def paired_summary(results):
    labels = list(results)
    pairs = {}
    for i, left in enumerate(labels):
        for right in labels[i + 1:]:
            a, b = results[left], results[right]
            assert [row["seed"] for row in a] == [row["seed"] for row in b]
            np.testing.assert_array_equal([row["initial_position"] for row in a],
                                          [row["initial_position"] for row in b])
            both = [(x, y) for x, y in zip(a, b) if x["success"] and y["success"]]
            pairs[f"{left} vs {right}"] = dict(
                both_success=len(both),
                left_only_success=sum(x["success"] and not y["success"] for x, y in zip(a, b)),
                right_only_success=sum(y["success"] and not x["success"] for x, y in zip(a, b)),
                mean_return_difference_left_minus_right=float(np.mean(
                    [x["return_ext"] - y["return_ext"] for x, y in zip(a, b)])),
                both_success_mean_length_left=float(np.mean([x["length"] for x, y in both])) if both else None,
                both_success_mean_length_right=float(np.mean([y["length"] for x, y in both])) if both else None)
    return pairs


@torch.inference_mode()
def trace_examples(run_dirs, results, output):
    # 用训练 seed 的起点和各模型首个失败起点做例子，不将挑选的例子当总体统计。
    seeds = list(dict.fromkeys(json.loads((run / "config.json").read_text(encoding="utf-8"))["seed"]
                               for run in run_dirs))
    for rows in results.values():
        failure = next((row for row in rows if not row["success"]), None)
        if failure is not None and failure["seed"] not in seeds:
            seeds.append(int(failure["seed"]))
    figure, axes = plt.subplots(2, len(seeds), figsize=(5 * len(seeds), 7),
                                squeeze=False, constrained_layout=True)
    traces, outcomes = [], {}
    for run in run_dirs:
        model, config = load_model(run)
        for column, seed in enumerate(seeds):
            env = gym.make(config["env_id"])
            try:
                obs, _ = env.reset(seed=seed)
                positions, velocities, actions = [float(obs[0])], [float(obs[1])], []
                while True:
                    q = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0]
                    action = q.argmax().item()
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    traces.append(dict(run=run.name, seed=seed, step=len(actions),
                                       position=float(obs[0]), velocity=float(obs[1]), action=action,
                                       q_left=q[0].item(), q_neutral=q[1].item(), q_right=q[2].item(),
                                       next_position=float(next_obs[0]), next_velocity=float(next_obs[1])))
                    actions.append(action)
                    positions.append(float(next_obs[0]))
                    velocities.append(float(next_obs[1]))
                    obs = next_obs
                    if terminated or truncated:
                        break
                outcomes[f"{run.name}/seed{seed}"] = dict(length=len(actions), success=bool(terminated),
                    initial_position=positions[0], min_position=min(positions), max_position=max(positions),
                    action_counts=np.bincount(actions, minlength=3).tolist())
                # 单样本推理与批量推理在这些样例上的结果应完全一致。
                reference = next((row for row in results[run.name] if row["seed"] == seed), None)
                if reference:
                    assert len(actions) == reference["length"] and int(terminated) == reference["success"]
                axes[0, column].plot(positions, label=run.name)
                axes[1, column].plot(velocities, label=run.name)
                axes[0, column].set(title=f"seed={seed}, initial x={positions[0]:.5f}",
                                    xlabel="Step", ylabel="Position")
                axes[1, column].set(xlabel="Step", ylabel="Velocity")
            finally:
                env.close()
    for column in range(len(seeds)):
        axes[0, column].axhline(.5, color="black", linestyle="--", alpha=.5, label="Goal position")
        for row in range(2):
            axes[row, column].grid(alpha=.2)
            axes[row, column].legend(fontsize=7)
    figure.savefig(output / "example_trajectories.png", dpi=160)
    plt.close(figure)
    write_csv(output / "example_trajectories.csv", traces)
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", default=[
        Path("runs/baseline-seed0"), Path("runs/simhash-seed0"),
        Path("runs/simhash-seed0-appendconstant")])
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.batch_size <= 0:
        parser.error("episodes 和 batch-size 必须为正")
    if len({run.name for run in args.runs}) != len(args.runs):
        parser.error("run 目录名称必须各不相同")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results, configs, training = {}, {}, {}
    report = dict(evaluation=dict(seed=args.seed, episodes=args.episodes,
                                  batch_size=args.batch_size, greedy=True, device="cpu"),
                  versions=dict(python=platform.python_version(), torch=torch.__version__,
                                numpy=np.__version__, gymnasium=gym.__version__), runs={})
    for run in args.runs:
        model, config = load_model(run)
        disk_config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        assert config == {key: value for key, value in disk_config.items() if key != "device"}
        rows = evaluate(model, config, args.seed, args.episodes, args.batch_size)
        # Confirm return/length semantics rather than assuming failure always means -200.
        assert all(row["return_ext"] == -row["length"] for row in rows)
        results[run.name], configs[run.name] = rows, config
        training[run.name] = training_rows(run)
        report["runs"][run.name] = dict(
            path=str(run.resolve()), checkpoint_sha256=hashlib.sha256((run / "q_network.pt").read_bytes()).hexdigest(),
            config=config, evaluation=summary(rows), training=training_summary(training[run.name], config))
        write_csv(args.output_dir / f"{run.name}.csv", rows)
        print(run.name, json.dumps(report["runs"][run.name]["evaluation"]), flush=True)
    report["paired"] = paired_summary(results)
    plot_overview(results, training, args.output_dir)
    if {config["append_constant"] for config in configs.values()} == {False, True}:
        report["hash_grid_unique_codes"] = plot_hash(configs, args.output_dir)
    report["example_trajectories"] = trace_examples(args.runs, results, args.output_dir)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
