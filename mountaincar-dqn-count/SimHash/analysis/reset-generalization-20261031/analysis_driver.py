"""固定预算、三个训练 seed 的 reset 对照。训练使用 GPU，评估使用 CPU 批量前向。

从 SimHash 目录运行。run 会冻结源码、保存初始状态和命令，并启动独立训练进程。
analyze 在同一环境下评估新旧模型，生成逐回合数据、训练 seed 级统计和图像。
"""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


VARIANTS = {"baseline": (0.0, False), "simhash": (0.1, False), "append": (0.1, True)}
SOURCE_FILES = ["train.py", "config.py", "dqn.py", "replay.py", "model.py",
                "exploration.py", "evaluate.py", "analyze.py"]
LEGACY = {"baseline": "runs/baseline-seed0", "simhash": "runs/simhash-seed0",
          "append": "runs/simhash-seed0-appendconstant"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def versions():
    import gymnasium
    import numpy
    import torch
    return dict(python=platform.python_version(), torch=torch.__version__,
                gymnasium=gymnasium.__version__, numpy=numpy.__version__,
                cuda_available=torch.cuda.is_available(), cuda_runtime=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cudnn_version=torch.backends.cudnn.version(),
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled())


def worker(args):
    import gymnasium as gym
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("本实验要求 CUDA，禁止静默回退 CPU")
    sys.path.insert(0, str(args.source.resolve()))
    import train
    from config import Config

    beta, append = VARIANTS[args.variant]
    config = Config(seed=args.seed, total_steps=args.steps, beta=beta, append_constant=append)
    original_make = gym.make
    starts = []

    class ResetRecorder(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.file = (args.run_dir / "resets.csv").open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=[
                "episode_index", "seed_argument", "position", "velocity"])
            self.writer.writeheader()

        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            starts.append(float(obs[0]))
            self.writer.writerow(dict(episode_index=len(starts) - 1,
                seed_argument=kwargs.get("seed"), position=float(obs[0]), velocity=float(obs[1])))
            self.file.flush()
            if len(starts) == 1:
                assert kwargs.get("seed") == args.seed
            else:
                assert kwargs.get("seed") is None, "后续回合不应重新设 seed"
            return obs, info

        def close(self):
            try:
                super().close()
            finally:
                self.file.close()

    train.gym.make = lambda *a, **kw: ResetRecorder(original_make(*a, **kw))
    started = time.monotonic()
    train.train(config, args.run_dir, torch.device("cuda"))
    torch.cuda.synchronize()
    assert len(starts) > 1 and len(set(starts)) > 1
    write_json(args.run_dir / "complete.json", dict(
        completed_utc=now(), elapsed_seconds=time.monotonic() - started,
        checkpoint_sha256=digest(args.run_dir / "q_network.pt"),
        reset_count=len(starts), unique_initial_positions=len(set(starts)),
        initial_position_range=[min(starts), max(starts)],
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), versions=versions()))


def run(args):
    if not versions()["cuda_available"]:
        raise RuntimeError("GPU Docker 环境不可用")
    args.study.mkdir(parents=True, exist_ok=False)
    source = args.study / "source"
    source.mkdir()
    (args.study / "logs").mkdir()
    script = Path(__file__).resolve()
    project = script.parents[1]
    for filename in SOURCE_FILES:
        shutil.copy2(project / filename, source / filename)
    shutil.copy2(script, source / "reset_study.py")
    jobs = []
    for seed in args.seeds:
        for variant in VARIANTS:
            name = f"{variant}-seed{seed}"
            command = [sys.executable, "-u", str(source / "reset_study.py"), "worker", "--source", str(source),
                       "--run-dir", str(args.study / "runs" / name),
                       "--variant", variant, "--seed", str(seed), "--steps", str(args.steps)]
            jobs.append(dict(name=name, variant=variant, seed=seed, command=command))
    manifest = dict(created_utc=now(), versions=versions(), steps_per_run=args.steps,
        training_seeds=args.seeds, max_concurrent_jobs=args.jobs,
        reset_mode="first_seed_only", evaluation_ranges=[[10000, 10499], [20000, 20499]],
        source_sha256={filename: digest(source / filename) for filename in [*SOURCE_FILES, "reset_study.py"]},
        legacy_sha256={variant: digest(project / path / "q_network.pt") for variant, path in LEGACY.items()},
        jobs=jobs)
    write_json(args.study / "manifest.json", manifest)
    pending, active, completed, failed = list(jobs), [], [], []
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    last_print = 0.0
    try:
        while pending or active:
            while pending and len(active) < args.jobs and not failed:
                job = pending.pop(0)
                log = (args.study / "logs" / (job["name"] + ".log")).open("w", encoding="utf-8")
                process = subprocess.Popen(job["command"], stdout=log, stderr=subprocess.STDOUT, env=environment)
                active.append((job, process, log))
                print(f"START {job['name']} pid={process.pid}", flush=True)
            for job, process, log in list(active):
                code = process.poll()
                if code is not None:
                    log.close()
                    active.remove((job, process, log))
                    (completed if code == 0 else failed).append(job["name"])
                    print(f"FINISH {job['name']} exit={code}", flush=True)
            status = dict(updated_utc=now(), completed=completed, failed=failed,
                          active=[job["name"] for job, _, _ in active],
                          pending=[job["name"] for job in pending])
            write_json(args.study / "status.json", status)
            if failed and not active:
                raise RuntimeError(f"训练失败，请查看 logs: {failed}")
            if time.monotonic() - last_print >= 30:
                print(json.dumps(status), flush=True)
                last_print = time.monotonic()
            if pending or active:
                time.sleep(3)
    finally:
        for _, process, log in active:
            process.terminate()
            process.wait()
            log.close()
    print("ALL TRAINING COMPLETE", flush=True)


def analyze(args):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    torch.set_num_threads(1)
    manifest = json.loads((args.study / "manifest.json").read_text(encoding="utf-8"))
    source = args.study / "source"
    for filename, expected in manifest["source_sha256"].items():
        assert digest(source / filename) == expected, f"Frozen source changed: {filename}"
    # 本轮只补充过分析逻辑；确认实际 worker 及其运行时 helper 与冻结版本一致。
    def functions(path):
        module = ast.parse(path.read_text(encoding="utf-8"))
        return {node.name: ast.dump(node, include_attributes=False) for node in module.body
                if isinstance(node, ast.FunctionDef)}
    original_functions, current_functions = functions(source / "reset_study.py"), functions(Path(__file__))
    assert all(original_functions[name] == current_functions[name]
               for name in ["worker", "versions", "write_json", "digest", "now"])
    sys.path.insert(0, str(source.resolve()))
    from analyze import evaluate, load_model, summary, training_rows, training_summary
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "analysis_driver.py")
    project = Path(__file__).resolve().parents[1]
    entries = [dict(name=job["name"], variant=job["variant"], seed=job["seed"], cohort="varied",
                    directory=args.study / "runs" / job["name"]) for job in manifest["jobs"]]
    entries.extend(dict(name=f"legacy-{variant}-seed0", variant=variant, seed=0, cohort="fixed_legacy",
                        directory=project / path) for variant, path in LEGACY.items())
    all_rows, per_run, training, evaluation, reset_rows = [], [], {}, {}, {}
    control_root = args.study / "fixed-controls"
    control_manifest = None
    if control_root.exists():
        control_manifest = json.loads((control_root / "manifest.json").read_text(encoding="utf-8"))
        assert control_manifest["training_source_sha256"] == manifest["source_sha256"]
        assert digest(control_root / "fixed_start_control.py") == control_manifest["worker_sha256"]
        entries.extend(dict(name=f"matched-fixed-{variant}-seed0", variant=variant, seed=0,
            cohort="fixed_matched", directory=control_root / f"{variant}-seed0") for variant in VARIANTS)
    detail = dict(evaluated_utc=now(), versions=versions(), training_manifest=manifest,
                  analysis_script_sha256=digest(Path(__file__)), fixed_control_manifest=control_manifest, runs={})
    detail["worker_matches_frozen_source"] = True
    for entry in entries:
        directory, name = entry["directory"], entry["name"]
        if entry["cohort"] in ["varied", "fixed_matched"]:
            complete = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
            assert digest(directory / "q_network.pt") == complete["checkpoint_sha256"]
        else:
            assert digest(directory / "q_network.pt") == manifest["legacy_sha256"][entry["variant"]]
        model, config = load_model(directory)
        rows = []
        for start, end in manifest["evaluation_ranges"]:
            rows.extend(evaluate(model, config, start, end - start + 1, batch_size=128))
        assert all(row["return_ext"] == -row["length"] for row in rows)
        statistics = summary(rows)
        train = training_rows(directory)
        train_stats = training_summary(train, config)
        training[name], evaluation[name] = train, rows
        record = {key: entry[key] for key in ["name", "variant", "seed", "cohort"]}
        record.update({key: statistics[key] for key in ["episodes", "successes", "failures",
                       "success_rate", "mean_return_ext", "success_mean_length"]})
        record.update(train_stats)
        per_run.append(record)
        detail["runs"][name] = dict(config=config, checkpoint_sha256=digest(directory / "q_network.pt"),
                                     evaluation=statistics, training=train_stats)
        if entry["cohort"] != "fixed_legacy":
            from exploration import SimHashCounter
            with (directory / "resets.csv").open(encoding="utf-8", newline="") as file:
                resets = list(csv.DictReader(file))
            reset_rows[name] = resets
            positions = [float(row["position"]) for row in resets]
            if entry["cohort"] == "varied":
                assert resets[0]["seed_argument"] == str(entry["seed"])
                assert all(row["seed_argument"] == "" for row in resets[1:])
                assert len(set(positions)) > 1
            else:
                assert all(row["effective_seed"] == "0" for row in resets)
                assert len(set(positions)) == 1
            counter = SimHashCounter(config["n_bits"], config["seed"], config["append_constant"])
            initial_keys = {counter.key([float(row["position"]), float(row["velocity"])]) for row in resets}
            detail["runs"][name]["reset_audit"] = dict(
                reset_count=len(resets), unique_initial_positions=len(set(positions)),
                initial_position_min=min(positions), initial_position_max=max(positions),
                distinct_hash_keys_of_reset_observations=len(initial_keys))
        for row in rows:
            all_rows.append(dict(run=name, **row))
        print(name, json.dumps(statistics), flush=True)
    reference = evaluation[entries[0]["name"]]
    for rows in evaluation.values():
        assert [r["seed"] for r in rows] == [r["seed"] for r in reference]
        np.testing.assert_array_equal([r["initial_position"] for r in rows],
                                      [r["initial_position"] for r in reference])
    group_rows = []
    for variant in VARIANTS:
        selected = [row for row in per_run if row["cohort"] == "varied" and row["variant"] == variant]
        row = dict(variant=variant, training_runs=len(selected))
        for metric in ["success_rate", "mean_return_ext", "success_mean_length"]:
            values = np.array([r[metric] for r in selected if r[metric] is not None])
            row[metric + "_valid_runs"] = len(values)
            row[metric + "_mean"] = float(values.mean()) if len(values) else None
            row[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
            row[metric + "_min"] = float(values.min()) if len(values) else None
            row[metric + "_max"] = float(values.max()) if len(values) else None
        group_rows.append(row)
    detail["groups"] = group_rows
    detail["reset_sequences_match_by_seed"] = {}
    for seed in manifest["training_seeds"]:
        sequences = [[row["position"] for row in reset_rows[f"{variant}-seed{seed}"]] for variant in VARIANTS]
        common = min(map(len, sequences))
        assert all(seq[:common] == sequences[0][:common] for seq in sequences)
        detail["reset_sequences_match_by_seed"][str(seed)] = dict(identical_prefix_length=common)
    if control_manifest is not None:
        detail["paired_reset_seed0"], detail["fixed_vs_legacy_weights"] = {}, {}
        for variant in VARIANTS:
            varied = evaluation[f"{variant}-seed0"]
            fixed = evaluation[f"matched-fixed-{variant}-seed0"]
            detail["paired_reset_seed0"][variant] = dict(
                varied_only_success=sum(x["success"] and not y["success"] for x, y in zip(varied, fixed)),
                fixed_only_success=sum(y["success"] and not x["success"] for x, y in zip(varied, fixed)),
                both_success=sum(x["success"] and y["success"] for x, y in zip(varied, fixed)),
                mean_return_difference_varied_minus_fixed=float(np.mean(
                    [x["return_ext"] - y["return_ext"] for x, y in zip(varied, fixed)])))
            fixed_model, _ = load_model(control_root / f"{variant}-seed0")
            legacy_model, _ = load_model(project / LEGACY[variant])
            a, b = fixed_model.state_dict(), legacy_model.state_dict()
            detail["fixed_vs_legacy_weights"][variant] = dict(
                tensors_identical=all(torch.equal(a[key], b[key]) for key in a),
                max_abs_difference=max((a[key] - b[key]).abs().max().item() for key in a),
                episode_logs_identical=training[f"matched-fixed-{variant}-seed0"] ==
                                       training[f"legacy-{variant}-seed0"])
    write_csv(args.output / "per_run.csv", per_run)
    write_csv(args.output / "per_episode.csv", all_rows)
    write_csv(args.output / "by_variant.csv", group_rows)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for column, metric in enumerate(["success_rate", "mean_return_ext"]):
        for index, variant in enumerate(VARIANTS):
            rows = [r for r in per_run if r["cohort"] == "varied" and r["variant"] == variant]
            values = np.array([r[metric] for r in rows]) * (100 if metric == "success_rate" else 1)
            axes[column].bar(index, values.mean(), alpha=.25, color=f"C{index}")
            for row, value in zip(rows, values):
                axes[column].scatter(index + (row["seed"] - 1) * .12, value,
                                     color=f"C{index}", marker=["o", "s", "^"][row["seed"] % 3], s=50)
            old = next(r for r in per_run if r["cohort"] == "fixed_legacy" and r["variant"] == variant)
            axes[column].scatter(index, old[metric] * (100 if metric == "success_rate" else 1),
                                 color="black", marker="x", s=70)
            if control_manifest is not None:
                matched = next(r for r in per_run if r["cohort"] == "fixed_matched" and r["variant"] == variant)
                axes[column].scatter(index, matched[metric] * (100 if metric == "success_rate" else 1),
                    facecolors="none", edgecolors="darkorange", marker="D", s=120)
        axes[column].set_xticks(range(3), VARIANTS)
        axes[column].grid(axis="y", alpha=.2)
    axes[0].set(title="Success rate: each dot is one training seed", ylabel="Success (%)", ylim=(0, 105))
    axes[1].set(title="Mean raw return: higher is better", ylabel="Mean return")
    figure.suptitle("100k steps; black x = legacy seed0; hollow diamond = matched fixed-start seed0", fontsize=10)
    figure.savefig(args.output / "comparison.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(3, 3, figsize=(13, 10), sharex=True, sharey=True, constrained_layout=True)
    for row_index, seed in enumerate(manifest["training_seeds"]):
        for col_index, variant in enumerate(VARIANTS):
            rows = evaluation[f"{variant}-seed{seed}"]
            axes[row_index, col_index].scatter([r["initial_position"] for r in rows],
                [r["length"] for r in rows], s=3, color=f"C{col_index}", alpha=.4)
            axes[row_index, col_index].set(title=f"{variant}, train seed={seed}",
                xlabel="Initial position", ylabel="Episode length")
            axes[row_index, col_index].grid(alpha=.2)
    figure.savefig(args.output / "initial_position_vs_length.png", dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for column, variant in enumerate(VARIANTS):
        for seed in manifest["training_seeds"]:
            rows = training[f"{variant}-seed{seed}"]
            window = min(30, len(rows))
            steps = [r["step"] for r in rows][window - 1:]
            for axrow, metric in enumerate(["return_ext", "success"]):
                values = np.array([r[metric] for r in rows])
                axes[axrow, column].plot(steps, np.convolve(values, np.ones(window) / window, mode="valid"),
                                          label=f"seed{seed}")
                axes[axrow, column].grid(alpha=.2)
                axes[axrow, column].legend()
                axes[axrow, column].set(title=variant, xlabel="Environment steps", ylabel=f"Rolling {metric}")
    figure.savefig(args.output / "training_curves.png", dpi=170)
    plt.close(figure)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for column, seed in enumerate(manifest["training_seeds"]):
        starts = [float(row["position"]) for row in reset_rows[f"baseline-seed{seed}"]]
        axes[column].hist(starts, bins=20, range=(-.6, -.4), density=True, alpha=.7)
        if seed == 0 and control_manifest is not None:
            fixed_start = float(reset_rows["matched-fixed-baseline-seed0"][0]["position"])
            axes[column].axvline(fixed_start, color="darkorange", label="Fixed seed0 start")
            axes[column].legend(fontsize=8)
        axes[column].set(title=f"Varied starts: train seed={seed}", xlabel="Reset position", ylabel="Density")
    figure.savefig(args.output / "reset_distributions.png", dpi=170)
    plt.close(figure)
    write_json(args.output / "summary.json", detail)
    print("ANALYSIS COMPLETE", json.dumps(group_rows), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    runner = subparsers.add_parser("run")
    runner.add_argument("--study", type=Path, required=True)
    runner.add_argument("--steps", type=int, default=100_000)
    runner.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    runner.add_argument("--jobs", type=int, default=3)
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--source", type=Path, required=True)
    worker_parser.add_argument("--run-dir", type=Path, required=True)
    worker_parser.add_argument("--variant", choices=VARIANTS, required=True)
    worker_parser.add_argument("--seed", type=int, required=True)
    worker_parser.add_argument("--steps", type=int, required=True)
    analyzer = subparsers.add_parser("analyze")
    analyzer.add_argument("--study", type=Path, required=True)
    analyzer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        if args.steps <= 0 or args.jobs <= 0:
            parser.error("steps 和 jobs 必须为正")
        run(args)
    elif args.command == "worker":
        worker(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
