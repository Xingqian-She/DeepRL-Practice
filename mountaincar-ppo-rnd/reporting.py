"""回合统计、CSV 和图片输出；不参与训练目标的计算。"""

import csv
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 只保存图片，支持无显示器的训练环境。
import matplotlib.pyplot as plt
import numpy as np

WINDOW_SIZE = 50
EPISODE_WINDOW = 50


def print_progress(update, num_updates, logs, performance):
    """损失用于诊断更新，回合表现用于观察任务是否学会。"""
    print(
        f"Update: {update}/{num_updates} | "
        f"PG Loss: {logs['policy_loss']:.4f} | "
        f"V_Ext Loss: {logs['value_ext_loss']:.4f} | "
        f"V_Int Loss: {logs['value_int_loss']:.4f} | "
        f"RND Loss: {logs['rnd_loss']:.2e} | "
        f"Entropy: {logs['entropy']:.4f} | "
        f"Episodes: {performance['episodes_completed']} | "
        f"Return(50ep): {performance['episode_return_mean']:.1f} | "
        f"Length(50ep): {performance['episode_length_mean']:.1f} | "
        f"Success(50ep): {performance['success_rate']:.1%}"
        + (f" | MaxPos: {logs['position_max']:.3f}"
           f" | Int/Ext AdvStd: {logs['adv_int_ext_std_ratio']:.2e}"
           if "position_max" in logs else ""),
        flush=True,
    )


class EpisodeTracker:
    """独立累计每个环境的回合数据，不受 rollout 边界影响。"""

    def __init__(self, num_envs):
        self.returns = np.zeros(num_envs, dtype=np.float64)
        self.lengths = np.zeros(num_envs, dtype=np.int64)
        self.episodes = []
        self.recent = deque(maxlen=EPISODE_WINDOW)

    def update(self, rewards, terminated, truncated, global_step):
        self.returns += rewards
        self.lengths += 1
        for i in np.flatnonzero(terminated | truncated):
            # MountainCar-v0 中 terminated 表示到达终点；超时不算成功。
            episode = {
                "episode": len(self.episodes) + 1,
                "global_step": global_step,
                "env_index": int(i),
                "return_ext": float(self.returns[i]),
                "length": int(self.lengths[i]),
                "success": int(terminated[i]),
                "terminated": int(terminated[i]),
                "truncated": int(truncated[i]),
            }
            self.episodes.append(episode)
            self.recent.append(episode)
            self.returns[i] = 0.0
            self.lengths[i] = 0

    def metrics(self):
        # 尚无完整回合时保留 NaN，避免把“没有观测”误记为 0 分或 0% 成功。
        def recent_mean(key):
            return float(np.mean([e[key] for e in self.recent])) if self.recent else float("nan")

        return {
            "episodes_completed": len(self.episodes),
            "episode_return_mean": recent_mean("return_ext"),
            "episode_length_mean": recent_mean("length"),
            "success_rate": recent_mean("success"),
        }


def moving_average(values, window_size=WINDOW_SIZE):
    if window_size < 1:
        raise ValueError("window_size 必须大于等于 1")

    averages = []
    for i in range(len(values)):
        start = max(0, i - window_size + 1)
        window = [v for v in values[start : i + 1] if np.isfinite(v)]
        average = sum(window) / len(window) if window else float("nan")
        averages.append(average)

    return averages


def save_training_logs(history, run_dir, episodes, plot=True):
    """将本次训练的完整日志覆盖写入 CSV 和 PNG，不追加上次训练的数据。"""
    if not history:
        return
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    fields = list(history[0])
    with (run_dir / "training.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)

    with (run_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "episode", "global_step", "env_index", "return_ext", "length",
            "success", "terminated", "truncated",
        ])
        writer.writeheader()
        writer.writerows(episodes)

    if not plot:
        return

    metrics = [
        ("policy_loss", "Policy Loss"),
        ("value_ext_loss", "Extrinsic Value Loss"),
        ("value_int_loss", "Intrinsic Value Loss"),
        ("rnd_loss", "RND Predictor Loss"),
        ("entropy", "Entropy"),
        ("clip_fraction", "Clip Fraction"),
        ("episode_return_mean", "Episode Return (recent 50 episodes)"),
        ("episode_length_mean", "Episode Length (recent 50 episodes)"),
        ("success_rate", "Success Rate (recent 50 episodes)"),
        ("reward_int_raw_mean", "Raw Intrinsic Reward"),
        ("reward_int_scaled_mean", "Scaled Intrinsic Reward"),
        ("int_return_std", "Discounted Intrinsic Reward Std"),
    ]
    if "adv_ext_weighted_std" in history[0]:
        metrics += [
            ("adv_ext_weighted_std", "Weighted Extrinsic Advantage Std"),
            ("adv_int_weighted_std", "Weighted Intrinsic Advantage Std"),
            ("adv_int_ext_std_ratio", "Intrinsic / Extrinsic Advantage Std"),
            ("position_min", "Minimum Position (rollout)"),
            ("position_max", "Maximum Position (rollout; goal = 0.5)"),
            ("speed_abs_max", "Maximum Absolute Speed (rollout)"),
        ]
    steps = [row["global_step"] for row in history]
    rows = (len(metrics) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(16, rows * 3.5), constrained_layout=True)
    try:
        for ax, (key, title) in zip(axes.flat, metrics):
            values = [row[key] for row in history]
            averages = moving_average(values, WINDOW_SIZE)
            ax.plot(steps, values, alpha=0.3, linewidth=1, marker=".", label="Raw")
            ax.plot(
                steps,
                averages,
                linewidth=2,
                marker=".",
                label=f"Moving average ({WINDOW_SIZE})",
            )
            ax.set_title(title)
            ax.set_xlabel("Environment steps")
            ax.set_ylabel(title)
            if len(steps) > 1:
                ax.set_xlim(steps[0], steps[-1])
            if key == "success_rate":
                ax.set_ylim(0.0, 1.0)
            if key in {"rnd_loss", "reward_int_raw_mean", "reward_int_scaled_mean",
                       "adv_ext_weighted_std", "adv_int_weighted_std", "adv_int_ext_std_ratio"}:
                ax.set_yscale("symlog", linthresh=1e-8)
            if key == "position_max":
                ax.axhline(0.5, color="green", linestyle="--", label="Goal")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.savefig(run_dir / "training.png", dpi=150)
    finally:
        plt.close(fig)
