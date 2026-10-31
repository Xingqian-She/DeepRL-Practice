"""只根据已保存 CSV 重画起点图，减少横轴刻度，保留绘图来源。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_dir", type=Path)
    args = parser.parse_args()
    summary_path = args.analysis_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with (args.analysis_dir / "per_episode.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    seeds = summary["training_manifest"]["training_seeds"]
    figure, axes = plt.subplots(len(seeds), 3, figsize=(12, 3.3 * len(seeds)),
                                sharex=True, sharey=True, squeeze=False, constrained_layout=True)
    for row_index, seed in enumerate(seeds):
        for column, variant in enumerate(["baseline", "simhash", "append"]):
            name = f"{variant}-seed{seed}"
            selected = [row for row in rows if row["run"] == name]
            ax = axes[row_index, column]
            ax.scatter([float(row["initial_position"]) for row in selected],
                       [int(row["length"]) for row in selected], s=3, alpha=.4, color=f"C{column}")
            boundary_successes = [row for row in selected if int(row["success"]) and int(row["length"]) == 200]
            if boundary_successes:
                ax.scatter([float(row["initial_position"]) for row in boundary_successes],
                           [200] * len(boundary_successes), s=45, facecolors="none", edgecolors="black",
                           label="Success at step 200")
                ax.legend(fontsize=7, loc="upper right")
            success = summary["runs"][name]["evaluation"]["success_rate"]
            ax.set(title=f"{variant}, seed={seed}: {success:.1%} success",
                   xlabel="Initial position", ylabel="Episode length", xlim=(-.605, -.395))
            ax.set_xticks([-.6, -.55, -.5, -.45, -.4])
            ax.grid(alpha=.2)
    figure.savefig(args.analysis_dir / "initial_position_vs_length.png", dpi=170)
    plt.close(figure)
    script = Path(__file__).resolve()
    shutil.copy2(script, args.analysis_dir / "plot_refinement.py")
    summary["plot_refinement"] = dict(source="plot_refinement.py",
        sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        input="per_episode.csv", changed="initial_position_vs_length.png")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
