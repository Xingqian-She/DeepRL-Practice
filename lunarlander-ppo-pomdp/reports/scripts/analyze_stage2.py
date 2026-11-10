"""Audit selected stage-2 artifacts and reproduce report tables/figures.

Run from the repository root:
    python -B lunarlander-ppo-pomdp/reports/scripts/analyze_stage2.py

Requires numpy, pandas, matplotlib. Reads runs without loading checkpoints or
importing training code; writes only to reports/stage2-analysis and reports/assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUPS = ("FO04", "PO01", "PO02", "PO03")
SEEDS = (42, 43, 44)
COLORS = dict(zip(GROUPS, ("#7b8794", "#277da8", "#dd8c27", "#3b9b72")))
LABELS = {
    "FO04": "FO04: full observation",
    "PO01": "PO01: direct p=1",
    "PO02": "PO02: fixed curriculum",
    "PO03": "PO03: performance curriculum",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tasks_of(state):
    tasks = state["tasks"]
    return list(tasks.values()) if isinstance(tasks, dict) else tasks


def selected_attempt(task):
    return next(a for a in task["attempts"] if a["number"] == task["current_attempt"])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    require(np.isclose(actual, expected, rtol=1e-8, atol=1e-7),
            f"{message}: {actual!r} != {expected!r}")


def episode_metrics(frame):
    returns = frame["return"].astype(float)
    lengths = frame["length"].astype(int)
    out = {
        "mean_return": returns.mean(), "std_return": returns.std(ddof=0),
        "mean_length": lengths.mean(), "score_200_rate": (returns >= 200).mean(),
        "positive_terminal_rate": (frame.outcome == "positive_terminal").mean(),
        "terminal_penalty_rate": (frame.terminal_reward == -100).mean(),
        "timeout_rate": (frame.outcome == "timeout").mean(),
        "mean_terminal_reward": frame.terminal_reward.mean(),
        "return_p10": returns.quantile(.1), "return_p50": returns.quantile(.5),
        "return_p90": returns.quantile(.9), "return_lt_minus300_rate": (returns < -300).mean(),
        "return_lt_minus500_rate": (returns < -500).mean(),
        "final_abs_angle": frame.final_angle.abs().mean(),
        "final_abs_x": frame.final_x.abs().mean(),
    }
    for action in range(4):
        out[f"action_fraction_{action}"] = frame[f"action_count_{action}"].sum() / lengths.sum()
    for outcome in ("positive_terminal", "terminal_penalty", "timeout"):
        subset = frame[frame.outcome == outcome]
        # None is serialized as an empty CSV cell / JSON null, never as NaN.
        out[f"{outcome}_mean_return"] = subset["return"].mean() if len(subset) else None
        out[f"{outcome}_mean_length"] = subset.length.mean() if len(subset) else None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--batch", default="course-10m-v1")
    args = parser.parse_args()
    project = args.project_dir.resolve()
    batch = project / "runs" / "experiments" / args.batch
    frozen = read_json(batch / "frozen-recipe.json")
    baseline = batch.parent / frozen["baseline_batch"]
    state, baseline_state = read_json(batch / "state.json"), read_json(baseline / "state.json")
    summary = read_json(batch / "summary" / "summary.json")
    output, assets = project / "reports" / "stage2-analysis", project / "reports" / "assets"
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    manifest, checked = [], {}

    def check_artifacts(run, expected_hashes, origin):
        for relative, expected in expected_hashes.items():
            path = (run / relative).resolve()
            if path not in checked:
                actual = sha256(path)
                checked[path] = actual
                manifest.append({"path": path.relative_to(project).as_posix(), "origin": origin,
                                 "sha256": actual, "expected_sha256": expected,
                                 "bytes": path.stat().st_size, "matches": actual == expected})
            require(checked[path] == expected, f"Artifact hash mismatch: {path}")

    new_entries = []
    for task in tasks_of(state):
        require(task["status"] == "completed", f"Unfinished task: {task['spec']['key']}")
        attempt = selected_attempt(task)
        require(attempt["exit_code"] == 0, f"Nonzero exit: {task['spec']['key']}")
        run = batch / attempt["path"] / "run"
        check_artifacts(run, attempt["artifact_hashes"], "stage2 selected attempt")
        new_entries.append((run, task["spec"], attempt))

    baseline_tasks = {task["spec"]["key"]: task for task in tasks_of(baseline_state)}
    for evidence in frozen["training_evidence"] + frozen["reference_runs"]:
        task = baseline_tasks[evidence["task"]]
        attempt = selected_attempt(task)
        require(attempt["number"] == evidence["attempt"] and attempt["run_id"] == evidence["run_id"],
                f"Changed baseline attempt: {evidence['task']}")
        check_artifacts(baseline / evidence["path"], evidence["artifact_hashes"], "frozen baseline evidence")

    references = [(baseline / r["path"], baseline_tasks[r["task"]]["spec"],
                   selected_attempt(baseline_tasks[r["task"]])) for r in frozen["reference_runs"]]
    entries = references + new_entries
    require(len(new_entries) == 63 and len(references) == 21, "Unexpected task count")
    files = list(project.glob("*.py")) + list((project.parent / "shared" / "src").rglob("*.py"))
    current_source = digest({p.relative_to(project.parent).as_posix(): sha256(p)
                             for p in sorted(files) if "__pycache__" not in p.parts})
    require(current_source == frozen["source_fingerprint"], "Training source changed since the experiment")

    train_frames, eval_frames, train_rows, heldout_rows, all_episodes = {}, {}, [], [], []
    metadata, periodic_rows = {}, []
    common_environment = {key: frozen["environment"][key] for key in
                          ("precision", "device", "python", "dependencies", "hardware", "evaluation_protocol")}
    train_lookup = {(spec["group"], spec["seed"]): (run, attempt) for run, spec, attempt in entries
                    if spec["kind"] == "train"}
    episode_checks, comparison_checks = 0, 0
    for run, spec, attempt in entries:
        m = read_json(run / "metadata.json")
        group, seed = spec["group"], spec["seed"]
        require(m["status"] == "completed" and m["run_id"] == attempt["run_id"], f"Bad metadata: {run}")
        require(m["experiment_id"] == group and m["source_fingerprint"] == current_source,
                f"Bad source/condition: {run}")
        require({key: m[key] for key in common_environment} == common_environment, f"Changed environment: {run}")
        require(digest(m["parameters"]) == m["config_fingerprint"], f"Bad config fingerprint: {run}")
        if spec["kind"] == "train":
            require(m["parameters"] == spec["parameters"], f"Bad training task parameters: {run}")
            expected = dict(frozen["parameters"], seed=seed)
            expected["sensors"] = spec["parameters"]["sensors"]
            require(m["parameters"] == expected, f"Changed non-sensor training parameter: {run}")
            t, e, ep = (pd.read_csv(run / name) for name in
                        ("training.csv", "evaluation.csv", "evaluation_episodes.csv"))
            require(len(t) == 4882 and np.array_equal(t["update"], np.arange(1, 4883)), f"Bad updates: {run}")
            require(np.array_equal(t.global_step, t["update"] * 2048), f"Bad step sequence: {run}")
            require(m["training_budget_completed"] and m["effective_total_timesteps"] == 9998336
                    and m["global_step"] == 9998336, f"Bad budget: {run}")
            require(list(e["update"]) == list(range(100, 4801, 100)) + [4882], f"Bad eval sequence: {run}")
            require(len(ep) == 980, f"Bad periodic episode count: {run}")
            for _, evaluation in e.iterrows():
                episodes = ep[ep["update"] == evaluation["update"]]
                require(len(episodes) == 20 and sorted(episodes.seed) == list(range(10000, 10020)),
                        f"Bad periodic episode seeds: {run}")
                require(np.allclose(episodes.sensor_failure_probability, evaluation.sensor_failure_probability),
                        f"Bad periodic sensor condition: {run}")
                for metric, value in episode_metrics(episodes).items():
                    if metric in e.columns:
                        close(evaluation[metric], value, f"{run.name} periodic {metric}")
                        comparison_checks += 1
                episode_checks += len(episodes)
            key = (group, seed)
            train_frames[key], eval_frames[key], metadata[key] = t, e, m
            periodic_rows.extend(e.assign(group=group, training_seed=seed).to_dict("records"))
            p1 = t[t.sensor_failure_probability_start == 1]
            last = e.tail(5)
            firstp1 = e[e.sensor_failure_probability == 1]
            row = {
                "group": group, "seed": seed, "updates": len(t), "steps": int(t.global_step.iloc[-1]),
                "late_return": last.mean_return.mean(), "late_score_200_rate": last.score_200_rate.mean(),
                "final_return": e.mean_return.iloc[-1], "final_score_200_rate": e.score_200_rate.iloc[-1],
                "final_p": e.sensor_failure_probability.iloc[-1],
                "final_task_solved": m.get("final_task_solved", False),
                "peak_p1_periodic_return": firstp1.mean_return.max() if len(firstp1) else None,
                "p1_full_rollout_steps": len(p1) * 2048,
                "p1_first_rollout_step": int(p1.global_step.iloc[0]) if len(p1) else None,
                "p1_entry_lr": p1.learning_rate.iloc[0] if len(p1) else None,
                "p1_entry_lr_fraction": p1.learning_rate.iloc[0] / .0003 if len(p1) else None,
                "p1_lr_mass_fraction": p1.learning_rate.sum() / t.learning_rate.sum(),
                "p1_optimizer_steps": int(p1.optimizer_steps.sum()),
                "kl_stop_rate": t.ppo_early_stop.mean(), "sample_reuse": t.sample_reuse_count.mean(),
                "unique_coverage": t.unique_sample_coverage.mean(),
                "late_explained_variance": t.explained_variance.tail(500).mean(),
                "late_value_loss": t.value_loss.tail(500).mean(),
                "late_behavior_entropy": t.behavior_entropy.tail(500).mean(),
                "optimizer_steps": int(t.optimizer_steps.sum()),
                "consistency_warnings": int(t.consistency_warning.sum()),
                "max_logprob_error": t.consistency_max_logprob_error.max(),
                "elapsed_hours": m["elapsed_seconds"] / 3600,
                "evaluation_fraction": m["evaluation_fraction"],
                "evaluation_seconds": m["evaluation_seconds"],
                "attempt_wall_seconds": (datetime.fromisoformat(attempt["finished_at"]) -
                                          datetime.fromisoformat(attempt["started_at"])).total_seconds(),
            }
            train_rows.append(row)
        else:
            d = read_json(run / "evaluation.json")
            ep, saved_ep = pd.DataFrame(d["episodes"]), pd.read_csv(run / "episodes.csv")
            require(len(ep) == len(saved_ep) == 100 and sorted(ep.seed) == list(range(20000, 20100)),
                    f"Bad heldout episodes: {run}")
            for field in ep.columns:
                a, b = ep.sort_values("episode")[field], saved_ep.sort_values("episode")[field]
                if pd.api.types.is_numeric_dtype(a):
                    require(np.allclose(a, b, rtol=1e-8, atol=1e-7), f"CSV/JSON mismatch: {run} {field}")
                else:
                    require(list(a) == list(b), f"CSV/JSON mismatch: {run} {field}")
            probability, policy = spec["parameters"]["sensor_failure_probability"], spec["parameters"]["policy"]
            for field, value in spec["parameters"].items():
                if field != "device":
                    require(m["parameters"][field] == value, f"Wrong evaluation protocol parameter: {run} {field}")
            training_run, training_attempt = train_lookup[group, seed]
            training_metadata = read_json(training_run / "metadata.json")
            dependency = attempt["dependency"]
            require(dependency["run_id"] == training_metadata["run_id"]
                    and dependency["checkpoint_sha256"] == checked[(training_run / "checkpoints/latest.pt").resolve()],
                    f"Bad checkpoint dependency: {run}")
            for field in ("run_id", "experiment_id", "config_fingerprint", "source_fingerprint", "evaluation_protocol"):
                require(m["training_provenance"][field] == training_metadata[field], f"Bad provenance: {run}")
            require(d["training_provenance"] == m["training_provenance"]
                    and m["checkpoint_global_step"] == 9998336, f"Bad evaluation result provenance: {run}")
            require(m["parameters"]["training_parameters"] == training_metadata["parameters"]
                    and m["parameters"]["training_seed"] == seed, f"Bad training reference: {run}")
            recomputed = episode_metrics(ep)
            for metric, value in recomputed.items():
                if metric in d:
                    close(d[metric], value, f"heldout {group}/{seed}/{probability}/{policy} {metric}")
                    comparison_checks += 1
            row = {"group": group, "seed": seed, "p": probability, "policy": policy,
                   **recomputed, "policy_entropy": d["policy_entropy"],
                   "main_engine_probability": d["main_engine_probability"],
                   "evaluation_seconds": m["evaluation_seconds"]}
            heldout_rows.append(row)
            all_episodes.append(ep.assign(group=group, training_seed=seed, p=probability, policy=policy))
            episode_checks += len(ep)

    training, heldout = pd.DataFrame(train_rows), pd.DataFrame(heldout_rows)
    episodes, periodic = pd.concat(all_episodes, ignore_index=True), pd.DataFrame(periodic_rows)
    require(len(training) == 12 and len(heldout) == 72, "Missing comparison runs")
    for (group, probability, policy), part in heldout.groupby(["group", "p", "policy"]):
        require(sorted(part.seed) == list(SEEDS), "Missing/duplicate training seed")
        expected = next(g for g in summary["groups"] if g["kind"] == "evaluate"
                        and g["experiment_id"] == group and g["sensor_probability"] == probability
                        and g["policy"] == policy)
        require(expected["valid"], f"Invalid saved summary condition: {group}/{probability}/{policy}")
        for metric in ("mean_return", "score_200_rate", "timeout_rate"):
            close(part[metric].mean(), expected[metric + "_mean"], f"Saved summary {group} {metric} mean")
            close(part[metric].std(ddof=1), expected[metric + "_std"], f"Saved summary {group} {metric} SD")
            comparison_checks += 2

    transitions, stages, fixed_cliff = [], [], []
    for (group, seed), t in train_frames.items():
        e, m = eval_frames[group, seed], metadata[group, seed]
        if group == "PO03":
            recorded = pd.read_csv(train_lookup[group, seed][0] / "curriculum_transitions.csv")
            require(recorded.global_step.tolist() == [r["global_step"] for r in m["curriculum"]["transitions"]],
                    f"Transition file mismatch: {group}/{seed}")
            actual_stages = t.groupby("curriculum_stage").size().reindex(range(11), fill_value=0).to_numpy() * 2048
            require(list(actual_stages) == m["curriculum"]["stage_steps"], f"Stage count mismatch: {group}/{seed}")
            for stage, count in enumerate(actual_stages):
                evaluations = e[e.stage == stage]
                stages.append({"group": group, "seed": seed, "stage": stage, "p": stage / 10,
                               "steps": int(count), "evaluations": len(evaluations),
                               "mean_return": evaluations.mean_return.mean(),
                               "max_return": evaluations.mean_return.max()})
            for transition in m["curriculum"]["transitions"]:
                step = transition["global_step"]
                before = e[e.global_step <= step].tail(2)
                after = e[e.global_step > step].iloc[0]
                require(len(before) == 2 and np.allclose(before.sensor_failure_probability, transition["from_probability"])
                        and (before.mean_return >= 200).all() and (before.score_200_rate >= .8).all(),
                        f"Gate not passed: {group}/{seed}/{step}")
                require(np.isclose(after.sensor_failure_probability, transition["to_probability"]), "Skipped stage evaluation")
                transitions.append({"group": group, "seed": seed, **transition,
                                    "gate_first_return": before.mean_return.iloc[0],
                                    "gate_second_return": before.mean_return.iloc[1],
                                    "gate_first_score_rate": before.score_200_rate.iloc[0],
                                    "gate_second_score_rate": before.score_200_rate.iloc[1],
                                    "first_after_return": after.mean_return,
                                    "first_after_score_rate": after.score_200_rate,
                                    "return_drop": after.mean_return - before.mean_return.iloc[-1]})
        if group == "PO02":
            for _, r in e[e["update"].between(3500, 4000)].iterrows():
                fixed_cliff.append({"seed": seed, "update": int(r["update"]), "p": r.sensor_failure_probability,
                                    "mean_return": r.mean_return, "score_200_rate": r.score_200_rate})

    aggregate_metrics = [c for c in heldout.columns if c not in ("group", "seed", "p", "policy")]
    grouped = heldout.groupby(["group", "p", "policy"])[aggregate_metrics].agg(["mean", "std"])
    grouped.columns = ["_".join(c) for c in grouped.columns]
    grouped = grouped.reset_index()
    train_grouped = training.groupby("group").agg({
        c: ["mean", "std"] for c in training.select_dtypes(include="number").columns if c != "seed"})
    train_grouped.columns = ["_".join(c) for c in train_grouped.columns]
    train_grouped = train_grouped.reset_index()
    paired = []
    for group in GROUPS:
        for probability in (0, .5, 1):
            sub = heldout[(heldout.group == group) & (heldout.p == probability)]
            for seed in SEEDS:
                a = sub[(sub.seed == seed) & (sub.policy == "sample")].iloc[0]
                b = sub[(sub.seed == seed) & (sub.policy == "argmax")].iloc[0]
                paired.append({"group": group, "seed": seed, "p": probability,
                               "sample_minus_argmax_return": a.mean_return - b.mean_return,
                               "sample_minus_argmax_score_rate": a.score_200_rate - b.score_200_rate})

    for name, frame in {"training-seeds": training, "training-groups": train_grouped,
                        "heldout-seeds": heldout, "heldout-groups": grouped,
                        "heldout-episodes": episodes, "periodic-evaluations": periodic,
                        "transitions": pd.DataFrame(transitions), "performance-stages": pd.DataFrame(stages),
                        "fixed-cliff": pd.DataFrame(fixed_cliff), "policy-differences": pd.DataFrame(paired),
                        "artifact-audit": pd.DataFrame(manifest)}.items():
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.12g")

    tz = timezone(timedelta(hours=8))
    report_audit = {
        "batch": args.batch, "baseline": frozen["baseline_batch"],
        "new_tasks": len(new_entries), "reference_tasks": len(references),
        "new_task_statuses": sorted({t["status"] for t in tasks_of(state)}),
        "selected_attempts": sorted({a["number"] for _, _, a in entries}),
        "unique_verified_artifacts": len(manifest), "artifact_hash_mismatches": 0,
        "source_fingerprint": current_source, "source_matches_frozen": True,
        "new_training_runs": len(training[training.group != "FO04"]),
        "new_training_steps": int(training[training.group != "FO04"].steps.sum()),
        "comparison_training_runs": len(training), "comparison_training_log_rows": len(training) * 4882,
        "periodic_evaluations": len(periodic), "periodic_episode_records": len(periodic) * 20,
        "heldout_evaluations": len(heldout), "heldout_episode_records": len(episodes),
        "episode_records_recomputed": episode_checks, "scalar_comparisons_passed": comparison_checks,
        "batch_started_beijing": datetime.fromisoformat(state["created_at"]).astimezone(tz).isoformat(),
        "batch_finished_beijing": max(datetime.fromisoformat(a["finished_at"]) for _, _, a in new_entries).astimezone(tz).isoformat(),
        "consistency_warnings": int(training.consistency_warnings.sum()),
        "max_logprob_error": training.max_logprob_error.max(),
        "environment": frozen["environment"],
        "analysis_environment": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "new_training_elapsed_hours": training[training.group != "FO04"].elapsed_hours.sum(),
        "new_training_evaluation_seconds": training[training.group != "FO04"].evaluation_seconds.sum(),
        "new_heldout_evaluation_seconds": heldout[heldout.group != "FO04"].evaluation_seconds.sum(),
        "note": "All checks read stored artifacts; no retraining, new evaluation, checkpoint deserialization, or writes to runs.",
    }
    (output / "audit.json").write_text(json.dumps(report_audit, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})

    def save(fig, name):
        fig.savefig(assets / f"po-stage2-{name}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Thin lines / dots expose training seeds; error bars are sample SD across 3 seeds.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, group in zip(axes.flat, GROUPS):
        twin = ax.twinx()
        for seed, style in zip(SEEDS, ("-", "--", ":")):
            e = eval_frames[group, seed]
            ax.plot(e.global_step / 1e6, e.mean_return, style, color=COLORS[group], lw=1.5, label=f"seed {seed}")
            twin.step(e.global_step / 1e6, e.sensor_failure_probability, where="post", color="#444444", alpha=.25, lw=1)
        twin.set_ylim(-.02, 1.05)
        twin.set_ylabel("Evaluation p", color="#666666")
        twin.grid(False)
        ax.axhline(200, color="#a13c3c", ls="--", lw=1)
        ax.set(title=LABELS[group], xlabel="Training steps (millions)", ylabel="Periodic mean return")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Learning curves at the current curriculum difficulty (20 episodes / point)")
    save(fig, "learning-curves")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for ax, policy in zip(axes, ("sample", "argmax")):
        for j, group in enumerate(GROUPS):
            data = grouped[(grouped.group == group) & (grouped.policy == policy)].sort_values("p")
            ax.errorbar(data.p, data.mean_return_mean, yerr=data.mean_return_std, color=COLORS[group],
                        marker="o", capsize=4, label=group, lw=1.8)
            for seed in SEEDS:
                raw = heldout[(heldout.group == group) & (heldout.policy == policy) & (heldout.seed == seed)].sort_values("p")
                ax.scatter(raw.p + (j - 1.5) * .013, raw.mean_return, color=COLORS[group], alpha=.4, s=13)
        ax.axhline(200, color="#a13c3c", ls="--", lw=1)
        ax.set(title=policy, xlabel="Sensor failure probability p", xticks=[0, .5, 1])
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Held-out mean return (mean +/- seed SD)")
    fig.suptitle("Final checkpoints: matched held-out evaluation (100 episodes / training seed)")
    save(fig, "generalization")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
    for ax, seed in zip(axes, SEEDS):
        e = eval_frames["PO03", seed]
        switch = metadata["PO03", seed]["curriculum"]["transitions"][-1]["global_step"]
        data = e[e.global_step >= switch - 1228800]
        for p, marker, color in ((.8, "s", "#7b8794"), (.9, "o", "#dd8c27"), (1., "^", "#3b9b72")):
            part = data[np.isclose(data.sensor_failure_probability, p)]
            ax.plot((part.global_step - switch) / 1e6, part.mean_return, marker=marker, color=color, label=f"p={p:g}")
        ax.axvline(0, color="#444444", ls=":")
        ax.axhline(200, color="#a13c3c", ls="--", lw=1)
        ax.set(title=f"PO03 / seed {seed}", xlabel="Steps from p=1 transition (millions)")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("Periodic mean return")
    fig.suptitle("Passing the p=0.9 gate did not transfer to p=1")
    save(fig, "transition-cliff")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, constrained_layout=True)
    for ax, policy in zip(axes, ("sample", "argmax")):
        for group in GROUPS:
            x = np.sort(episodes[(episodes.group == group) & (episodes.p == 1) & (episodes.policy == policy)]["return"])
            ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COLORS[group], lw=2, label=group)
        ax.axvline(200, color="#a13c3c", ls="--", lw=1)
        ax.axvline(-300, color="#777777", ls=":", lw=1)
        minimum = episodes[episodes.p == 1]["return"].min()
        lower = 100 * np.floor(minimum / 100)
        ax.set(title=f"p=1 / {policy}", xlabel="Episode return", ylim=(0, 1), xlim=(lower, 350))
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Cumulative episode fraction")
    fig.suptitle("Pooled empirical distributions: descriptive only, 3 x 100 episodes per curve")
    save(fig, "p1-return-distributions")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for ax, metric, title in zip(axes, ("positive_terminal_rate_mean", "timeout_rate_mean"),
                                 ("Terminal reward +100", "Timeout")):
        x = np.arange(3)
        for j, group in enumerate(GROUPS):
            data = grouped[(grouped.group == group) & (grouped.policy == "sample")].sort_values("p")
            ax.bar(x + (j - 1.5) * .19, data[metric] * 100, width=.18, color=COLORS[group], label=group)
        ax.set(title=title, xticks=x, xticklabels=["p=0", "p=0.5", "p=1"], ylabel="Episode fraction (%)", ylim=(0, 105))
        ax.legend(fontsize=8)
    fig.suptitle("Final checkpoints / sample: outcome indicators from episode logs")
    save(fig, "outcomes")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    names, starts, spans = [], [], []
    for group in ("PO01", "PO02", "PO03"):
        for seed in SEEDS:
            row = training[(training.group == group) & (training.seed == seed)].iloc[0]
            names.append(f"{group} / {seed}")
            starts.append((row.p1_first_rollout_step - 2048) / 1e6)
            spans.append(row.p1_full_rollout_steps / 1e6)
    y = np.arange(len(names))
    axes[0].barh(y, 9.998336, color="#e8ebee")
    axes[0].barh(y, spans, left=starts, color=[COLORS[n.split()[0]] for n in names])
    axes[0].set(yticks=y, yticklabels=names, xlabel="Training steps (millions)", title="Rollouts entirely at p=1", xlim=(0, 10.2))
    axes[0].invert_yaxis()
    for group in ("PO01", "PO02", "PO03"):
        for seed in SEEDS:
            t = train_frames[group, seed]
            if seed == 42 and group == "PO01":
                axes[1].plot(t.global_step / 1e6, t.learning_rate / .0003 * 100, color="#444444", lw=2, label="Shared LR schedule")
            p1 = t[t.sensor_failure_probability_start == 1].iloc[0]
            axes[1].scatter((p1.global_step - 2048) / 1e6, p1.learning_rate / .0003 * 100,
                            color=COLORS[group], marker={42: "o", 43: "s", 44: "^"}[seed], s=60,
                            label=f"{group}/{seed}")
    axes[1].set(xlabel="Training steps (millions)", ylabel="Learning rate / initial (%)", title="Learning rate on entering p=1")
    axes[1].legend(fontsize=8, ncol=2)
    save(fig, "p1-budget")

    table = ["# Recomputed held-out results", "", "Mean +/- sample SD over 3 training seeds.", "",
             "| Group | p | Policy | Return | >=200 | +100 terminal | Timeout |", "|---|---:|---|---:|---:|---:|---:|"]
    for _, r in grouped.iterrows():
        table.append(f"| {r.group} | {r.p:g} | {r.policy} | {r.mean_return_mean:.2f} +/- {r.mean_return_std:.2f} | "
                     f"{r.score_200_rate_mean:.1%} | {r.positive_terminal_rate_mean:.1%} | {r.timeout_rate_mean:.1%} |")
    (output / "tables.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print(json.dumps(report_audit, ensure_ascii=True, indent=2))
    print("\nP=1 training-seed metrics:")
    print(training.loc[training.group != "FO04", ["group", "seed", "p1_full_rollout_steps", "p1_entry_lr_fraction",
                                                 "p1_lr_mass_fraction", "late_return", "final_return"]].round(6).to_string(index=False))
    print("\nFinal transition to p=1:")
    print(pd.DataFrame(transitions).query("to_probability == 1").round(4).to_string(index=False))
    print(f"\nWrote analysis to {output} and 6 figures to {assets}")


if __name__ == "__main__":
    main()
