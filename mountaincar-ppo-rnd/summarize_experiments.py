"""汇总 diagnosis 下各次独立训练；不运行训练、不改写原日志。"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent / "experiments" / "diagnosis"


def summarize():
    summary = []
    histories = {}
    for directory in sorted(ROOT.iterdir()):
        if not directory.is_dir() or not (directory / "training.csv").exists():
            continue
        with (directory / "training.csv").open(encoding="utf-8") as file:
            history = list(csv.DictReader(file))
        with (directory / "episodes.csv").open(encoding="utf-8") as file:
            episodes = list(csv.DictReader(file))
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        histories[directory.name] = history
        last = history[-1]
        evaluations_path = directory / "evaluations.jsonl"
        evaluations = ([json.loads(line) for line in evaluations_path.read_text().splitlines()]
                       if evaluations_path.exists() else [])
        last_eval = evaluations[-1] if evaluations else None
        holdout_path = directory / "evaluation-seed20000.json"
        holdout = json.loads(holdout_path.read_text()) if holdout_path.exists() else None
        successes = [row for row in episodes if int(row["success"])]
        summary.append({
            "experiment": directory.name,
            "seed": config["seed"], "device": config["device"],
            "architecture": config["architecture"],
            "normalize_policy_input": config.get("normalize_policy_input", False),
            "ext_reward_scale": config.get("ext_reward_scale", 1),
            "ext_coef": config["ext_coef"], "int_coef": config["int_coef"],
            "steps": int(last["global_step"]),
            "episodes": len(episodes), "successes": len(successes),
            "first_success_step": int(successes[0]["global_step"]) if successes else "",
            "recent50_success": float(last["success_rate"]),
            "eval20_step": last_eval["global_step"] if last_eval else "",
            "eval20_argmax_success": last_eval["deterministic"]["success_rate"] if last_eval else "",
            "eval20_sample_success": last_eval["stochastic"]["success_rate"] if last_eval else "",
            "holdout100_argmax_success": holdout["deterministic"]["success_rate"] if holdout else "",
            "holdout100_sample_success": holdout["stochastic"]["success_rate"] if holdout else "",
            "holdout100_argmax_return": holdout["deterministic"]["return_mean"] if holdout else "",
        })
    with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    selected = {
        "residual-lr4e-5-s0": "Original + lower RND LR (51k steps)",
        "residual-scaled-reward1-s0": "Scale policy input only (102k steps)",
        "residual-raw-reward001-s0": "Scale external reward only (102k steps)",
        "residual-scaled-reward001-s0": "Scale both (205k steps)",
    }
    metrics = [
        ("success_rate", "Training success rate (recent 50 episodes)"),
        ("position_max", "Maximum position in rollout (goal = 0.5)"),
        ("adv_int_ext_std_ratio", "Weighted intrinsic / extrinsic advantage std"),
        ("reward_int_scaled_mean", "Mean scaled intrinsic reward"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, metrics):
        for name, label in selected.items():
            history = histories[name]
            x = np.array([int(row["global_step"]) for row in history])
            y = np.array([float(row[key]) for row in history])
            ax.plot(x, y, linewidth=1.2, label=label)
        ax.set(title=title, xlabel="Environment steps")
        ax.grid(alpha=0.25)
        if key == "success_rate":
            ax.set_ylim(-0.02, 1.02)
            ax.legend(fontsize=8, loc="lower right")
        if key == "position_max":
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        if key in {"adv_int_ext_std_ratio", "reward_int_scaled_mean"}:
            ax.set_yscale("log")
    fig.suptitle("MountainCar PPO + RND: controlled scale comparisons (seed 0)")
    fig.savefig(ROOT / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"Summarized {len(summary)} experiments: {ROOT / 'summary.csv'}")
    print(f"Comparison figure: {ROOT / 'comparison.png'}")


if __name__ == "__main__":
    summarize()
