"""Recompute the Encoder-Decoder exploratory report from registered attempts.

Run from the repository root:
    python -B lunarlander-ppo-pomdp/reports/scripts/analyze_encoder_decoder.py

Requires numpy, pandas, matplotlib. Reads stored JSON/CSV/checkpoint hashes only;
does not import training code, deserialize checkpoints, or write under runs/.
Historical PO01/PO03 retain their own source fingerprint.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_stage2 import close, digest, episode_metrics, read_json, require, selected_attempt, sha256, tasks_of


GROUPS = ("PO01", "ED_PO01", "PO03", "ED_PO03")
SEEDS = (42, 43, 44)
COLORS = dict(zip(GROUPS, ("#277da8", "#e47826", "#36916b", "#9b59b6")))
LABELS = {"PO01": "Separate / direct", "ED_PO01": "ED / direct",
          "PO03": "Separate / curriculum", "ED_PO03": "ED / curriculum"}


def aggregate(frame, keys, exclude=()):
    columns = [c for c in frame.select_dtypes(include="number").columns if c not in (*keys, *exclude)]
    result = frame.groupby(keys)[columns].agg(["mean", "std"])
    result.columns = ["_".join(c) for c in result.columns]
    return result.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    project = args.project_dir.resolve()
    output, assets = project / "reports/encoder-decoder-analysis", project / "reports/assets"
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    batches = {"historical": project / "runs/experiments/course-10m-v1",
               "new": project / "runs/experiments/encoder-decoder-10m-v1"}
    states = {name: read_json(path / "state.json") for name, path in batches.items()}
    require(states["new"]["environment"] == states["historical"]["environment"], "Changed run environment")
    require(len(tasks_of(states["new"])) == 42, "Unexpected new task count")
    files = list(project.glob("*.py")) + list((project.parent / "shared/src").rglob("*.py"))
    current_source = digest({p.relative_to(project.parent).as_posix(): sha256(p)
                             for p in sorted(files) if "__pycache__" not in p.parts})
    # An analysis should still work after training code evolves; record this fact,
    # and validate each artifact against its own batch rather than current code.
    source_matches_new = current_source == states["new"]["source_fingerprint"]
    entries, manifest, checked = [], [], {}
    for origin, batch in batches.items():
        wanted = ("PO01", "PO03") if origin == "historical" else ("ED_PO01", "ED_PO03")
        for task in tasks_of(states[origin]):
            spec = task["spec"]
            if spec["group"] not in wanted:
                continue
            require(task["status"] == "completed", f"Incomplete: {spec['key']}")
            attempt = selected_attempt(task)
            require(attempt["exit_code"] == 0, f"Failed: {spec['key']}")
            run = batch / attempt["path"] / "run"
            for relative, expected in attempt["artifact_hashes"].items():
                path = (run / relative).resolve()
                require(path.is_relative_to(batch), f"Artifact outside batch: {path}")
                if path not in checked:
                    actual = sha256(path)
                    require(actual == expected, f"Hash mismatch: {path}")
                    checked[path] = actual
                    manifest.append({"origin": origin, "path": path.relative_to(project).as_posix(),
                                     "sha256": actual})
                require(checked[path] == expected, f"Conflicting hash: {path}")
            entries.append((origin, run, spec, attempt))
    require(len(entries) == 84, "Missing comparison tasks")
    lookup = {(s["group"], s["seed"]): (r, a) for _, r, s, a in entries if s["kind"] == "train"}
    meta, train_frames, eval_frames = {}, {}, {}
    training_rows, heldout_rows, periodic_rows, episode_frames, diagnostic_frames = [], [], [], [], []
    scalar_checks = episode_checks = 0
    for origin, run, spec, attempt in entries:
        m = read_json(run / "metadata.json")
        group, seed = spec["group"], spec["seed"]
        require(m["status"] == "completed" and m["run_id"] == attempt["run_id"]
                and m["experiment_id"] == group, f"Wrong run metadata: {run}")
        require(m["source_fingerprint"] == states[origin]["source_fingerprint"], f"Wrong source: {run}")
        require(digest(m["parameters"]) == m["config_fingerprint"], f"Wrong config fingerprint: {run}")
        for key in ("precision", "device", "python", "dependencies", "hardware", "evaluation_protocol"):
            require(m[key] == states[origin]["environment"][key], f"Wrong environment {key}: {run}")
        if spec["kind"] == "train":
            require(m["parameters"] == spec["parameters"], f"Wrong training config: {run}")
            t, e, ep, d = (pd.read_csv(run / name) for name in
                          ("training.csv", "evaluation.csv", "evaluation_episodes.csv", "diagnostics.csv"))
            require(len(t) == 4882 and np.array_equal(t["update"], np.arange(1, 4883)), f"Bad updates: {run}")
            require(np.array_equal(t.global_step, t["update"] * 2048), f"Bad steps: {run}")
            require(m["training_budget_completed"] and m["global_step"] == 9998336, f"Bad budget: {run}")
            require(list(e["update"]) == list(range(100, 4801, 100)) + [4882], f"Bad evaluations: {run}")
            require(list(d["update"]) == [1] + list(range(50, 4851, 50)) + [4882], f"Bad diagnostics: {run}")
            require(len(ep) == 980, f"Bad periodic episodes: {run}")
            require(t.consistency_finite.eq(1).all(), f"Nonfinite consistency: {run}")
            for _, ev in e.iterrows():
                part = ep[ep["update"] == ev["update"]]
                require(len(part) == 20 and sorted(part.seed) == list(range(10000, 10020)), f"Wrong seeds: {run}")
                require(np.allclose(part.sensor_failure_probability, ev.sensor_failure_probability), f"Wrong p: {run}")
                for metric, value in episode_metrics(part).items():
                    if metric in e:
                        close(ev[metric], value, f"Periodic {group}/{seed}/{metric}")
                        scalar_checks += 1
                episode_checks += len(part)
            p1 = t[t.sensor_failure_probability_start == 1]
            p1_eval = e[e.sensor_failure_probability == 1]
            qualified = (p1_eval.mean_return >= 200) & (p1_eval.score_200_rate >= .8)
            solved = bool((qualified & qualified.shift(1, fill_value=False)).any())
            require(solved == m["final_task_solved"], f"Wrong solved flag: {run}")
            row = {"origin": origin, "group": group, "seed": seed, "steps": int(t.global_step.iloc[-1]),
                   "updates": len(t), "final_task_solved": solved,
                   "p1_full_rollout_steps": len(p1) * 2048,
                   "p1_entry_step": int(p1.global_step.iloc[0] - 2048),
                   "p1_entry_lr": p1.learning_rate.iloc[0],
                   "p1_lr_mass_fraction": p1.learning_rate.sum() / t.learning_rate.sum(),
                   "peak_p1_periodic_return": p1_eval.mean_return.max(),
                   "last_periodic_return": e.mean_return.iloc[-1],
                   "last5_periodic_return": e.mean_return.tail(5).mean(),
                   "p1_qualified_evaluations": int(qualified.sum()),
                   "elapsed_hours": m["elapsed_seconds"] / 3600,
                   "attempt_wall_hours": (datetime.fromisoformat(attempt["finished_at"]) -
                                          datetime.fromisoformat(attempt["started_at"])).total_seconds() / 3600,
                   "evaluation_seconds": m["evaluation_seconds"],
                   "kl_stop_rate": t.ppo_early_stop.mean(), "sample_reuse": t.sample_reuse_count.mean(),
                   "unique_coverage": t.unique_sample_coverage.mean(),
                   "clip_rate": t.gradient_clip_fraction.mean(),
                   "consistency_warnings": int(t.consistency_warning.sum()),
                   "max_logprob_error": t.consistency_max_logprob_error.max(),
                   "late_explained_variance": t.explained_variance.tail(500).mean(),
                   "late_value_loss": t.value_loss.tail(500).mean(),
                   "late_behavior_entropy": t.behavior_entropy.tail(500).mean(),
                   "cache_bytes": int(t.cache_bytes.iloc[-1]) if "cache_bytes" in t else None,
                   "model_parameters": m["model"]["parameters"]["total"],
                   "shared_parameters": m["model"]["parameters"]["shared"],
                   "diagnostic_probes": len(d)}
            if group.startswith("ED_"):
                for col in ("encoder_policy_grad_norm", "encoder_value_grad_norm", "cross_attention_policy_grad_norm"):
                    require(np.isfinite(d[col]).all() and (d[col] > 0).all(), f"Bad gradient probe: {run}/{col}")
                    row[col + "_median"] = d[col].median()
                ratio = d.encoder_value_grad_norm / d.encoder_policy_grad_norm
                row["encoder_value_policy_norm_ratio_median"] = ratio.median()
                row["shared_negative_cosine_fraction"] = (d.shared_gradient_cosine < 0).mean()
                row["cross_gate_final_min"] = d.cross_gate_min.iloc[-1]
                row["cross_gate_final_max"] = d.cross_gate_max.iloc[-1]
            training_rows.append(row)
            meta[group, seed], train_frames[group, seed], eval_frames[group, seed] = m, t, e
            periodic_rows.extend(e.assign(group=group, training_seed=seed).to_dict("records"))
            diagnostic_frames.append(d.assign(group=group, training_seed=seed))
        else:
            result = read_json(run / "evaluation.json")
            ep, csv_ep = pd.DataFrame(result["episodes"]), pd.read_csv(run / "episodes.csv")
            require(len(ep) == len(csv_ep) == 100 and sorted(ep.seed) == list(range(20000, 20100)), f"Bad episodes: {run}")
            for col in ep:
                left, right = ep.sort_values("episode")[col], csv_ep.sort_values("episode")[col]
                if pd.api.types.is_numeric_dtype(left):
                    require(np.allclose(left, right, rtol=1e-8, atol=1e-7), f"JSON/CSV differs: {run}/{col}")
                else:
                    require(list(left) == list(right), f"JSON/CSV differs: {run}/{col}")
            for field, value in spec["parameters"].items():
                actual = m["device"] if field == "device" else m["parameters"][field]
                require(actual == value, f"Wrong evaluation config: {run}/{field}")
            p, policy = spec["parameters"]["sensor_failure_probability"], spec["parameters"]["policy"]
            trun, ta = lookup[group, seed]
            tm = read_json(trun / "metadata.json")
            dep = attempt["dependency"]
            require(dep["run_id"] == tm["run_id"] and dep["attempt"] == ta["number"] and
                    dep["checkpoint_sha256"] == checked[(trun / "checkpoints/latest.pt").resolve()], f"Wrong dependency: {run}")
            for field in ("run_id", "experiment_id", "config_fingerprint", "source_fingerprint", "evaluation_protocol"):
                require(m["training_provenance"][field] == tm[field], f"Wrong provenance: {run}/{field}")
            require(result["training_provenance"] == m["training_provenance"] and
                    m["parameters"]["training_parameters"] == tm["parameters"] and
                    m["parameters"]["training_seed"] == seed and m["checkpoint_global_step"] == 9998336 and
                    m["parameters"]["checkpoint_name"] == "latest.pt", f"Wrong checkpoint: {run}")
            require(result["sensor_failure_probability"] == p and result["policy"] == policy and
                    result["evaluation_protocol"] == m["evaluation_protocol"], f"Wrong result protocol: {run}")
            recomputed = episode_metrics(ep)
            for metric, value in recomputed.items():
                if metric in result:
                    close(result[metric], value, f"Heldout {group}/{seed}/{p}/{policy}/{metric}")
                    scalar_checks += 1
            heldout_rows.append({"origin": origin, "group": group, "seed": seed, "p": p, "policy": policy,
                                 **recomputed, "policy_entropy": result["policy_entropy"],
                                 "main_engine_probability": result["main_engine_probability"],
                                 "evaluation_seconds": m["evaluation_seconds"]})
            episode_frames.append(ep.assign(group=group, training_seed=seed, p=p, policy=policy))
            episode_checks += len(ep)

    # Only architecture keys differ in corresponding registered training recipes.
    config_differences = []
    for seed in SEEDS:
        for old, new in (("PO01", "ED_PO01"), ("PO03", "ED_PO03")):
            a, b = (copy.deepcopy(meta[g, seed]["parameters"]) for g in (old, new))
            config_differences.append({"old": old, "new": new, "seed": seed,
                                       "old_agent": a["agent"], "new_agent": b["agent"]})
            a["agent"].pop("backbone_mode")
            b["agent"].pop("backbone_mode")
            require(b["agent"].pop("decoder_layers") == 4, "Wrong decoder layers")
            require(a == b, f"Non-architecture config differs: {old}/{new}/{seed}")

    training, heldout = pd.DataFrame(training_rows), pd.DataFrame(heldout_rows)
    periodic, episodes = pd.DataFrame(periodic_rows), pd.concat(episode_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
    grouped = aggregate(heldout, ["group", "p", "policy"], ["seed"])
    training_groups = aggregate(training, ["group"], ["seed"])
    for origin, batch in batches.items():
        summary = read_json(batch / "summary/summary.json")
        for (group, p, policy), part in heldout[heldout.origin == origin].groupby(["group", "p", "policy"]):
            require(sorted(part.seed) == list(SEEDS), "Missing/duplicate seeds")
            saved = next(g for g in summary["groups"] if g["kind"] == "evaluate" and
                         g["experiment_id"] == group and g["sensor_probability"] == p and g["policy"] == policy)
            require(saved["valid"], f"Invalid stored summary: {group}/{p}/{policy}")
            for metric in ("mean_return", "score_200_rate", "timeout_rate"):
                close(part[metric].mean(), saved[metric + "_mean"], "Saved summary mean")
                close(part[metric].std(ddof=1), saved[metric + "_std"], "Saved summary SD")
                scalar_checks += 2

    differences, difference_groups = [], []
    for old, new in (("PO01", "ED_PO01"), ("PO03", "ED_PO03")):
        for p in (0, .5, 1):
            for policy in ("sample", "argmax"):
                delta = []
                for seed in SEEDS:
                    a = heldout.query("group == @old and seed == @seed and p == @p and policy == @policy").iloc[0]
                    b = heldout.query("group == @new and seed == @seed and p == @p and policy == @policy").iloc[0]
                    delta.append(b.mean_return - a.mean_return)
                    differences.append({"old": old, "new": new, "p": p, "policy": policy, "seed": seed,
                                        "return_delta": delta[-1], "score_rate_delta": b.score_200_rate - a.score_200_rate})
                difference_groups.append({"old": old, "new": new, "p": p, "policy": policy,
                                          "mean_delta": np.mean(delta), "sd_delta": np.std(delta, ddof=1),
                                          "positive_seed_deltas": sum(d > 0 for d in delta),
                                          "min_seed_delta": min(delta), "max_seed_delta": max(delta),
                                          "leave_one_seed_out_min": min((sum(delta) - d) / 2 for d in delta),
                                          "leave_one_seed_out_max": max((sum(delta) - d) / 2 for d in delta)})

    transitions, stages = [], []
    for group in ("PO03", "ED_PO03"):
        for seed in SEEDS:
            t, e, m = train_frames[group, seed], eval_frames[group, seed], meta[group, seed]
            actual = t.groupby("curriculum_stage").size().reindex(range(11), fill_value=0).to_numpy() * 2048
            require(list(actual) == m["curriculum"]["stage_steps"], "Wrong stage steps")
            recorded = pd.read_csv(lookup[group, seed][0] / "curriculum_transitions.csv")
            require(recorded.global_step.tolist() == [r["global_step"] for r in m["curriculum"]["transitions"]], "Wrong transitions")
            for stage, steps in enumerate(actual):
                stages.append({"group": group, "seed": seed, "stage": stage, "p": stage / 10, "steps": int(steps)})
            for transition in m["curriculum"]["transitions"]:
                before = e[e.global_step <= transition["global_step"]].tail(2)
                after = e[e.global_step > transition["global_step"]].iloc[0]
                require(len(before) == 2 and np.allclose(before.sensor_failure_probability, transition["from_probability"])
                        and (before.mean_return >= 200).all() and (before.score_200_rate >= .8).all(), "Unqualified gate")
                require(np.isclose(after.sensor_failure_probability, transition["to_probability"]), "Skipped stage")
                transitions.append({"group": group, "seed": seed, **transition,
                                    "gate_first_return": before.mean_return.iloc[0],
                                    "gate_second_return": before.mean_return.iloc[1],
                                    "first_after_return": after.mean_return,
                                    "first_after_score_rate": after.score_200_rate,
                                    "return_delta": after.mean_return - before.mean_return.iloc[-1]})

    tables = {"training-seeds": training, "training-groups": training_groups,
              "heldout-seeds": heldout, "heldout-groups": grouped, "heldout-episodes": episodes,
              "periodic-evaluations": periodic, "diagnostics": diagnostics,
              "seed-differences": pd.DataFrame(differences), "difference-groups": pd.DataFrame(difference_groups),
              "transitions": pd.DataFrame(transitions), "performance-stages": pd.DataFrame(stages),
              "artifact-audit": pd.DataFrame(manifest)}
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False, float_format="%.12g")
    tz = timezone(timedelta(hours=8))
    new_attempts = [a for origin, _, _, a in entries if origin == "new"]
    audit = {"new_batch": batches["new"].name, "historical_batch": batches["historical"].name,
             "new_tasks": len(new_attempts), "historical_tasks": len(entries) - len(new_attempts),
             "selected_attempts": sorted({a["number"] for _, _, _, a in entries}),
             "unique_verified_artifacts": len(manifest), "artifact_hash_mismatches": 0,
             "source_fingerprints": {k: s["source_fingerprint"] for k, s in states.items()},
             "current_source_fingerprint": current_source, "current_source_matches_new_batch": source_matches_new,
             "environment": states["new"]["environment"], "config_differences": config_differences,
             "new_training_steps": int(training[training.origin == "new"].steps.sum()),
             "comparison_training_rows": int(training.updates.sum()),
             "periodic_evaluations": len(periodic), "heldout_evaluations": len(heldout),
             "episode_records_recomputed": episode_checks, "scalar_comparisons_passed": scalar_checks,
             "new_started_beijing": min(datetime.fromisoformat(a["started_at"]) for a in new_attempts).astimezone(tz).isoformat(),
             "new_finished_beijing": max(datetime.fromisoformat(a["finished_at"]) for a in new_attempts).astimezone(tz).isoformat(),
             "new_training_elapsed_hours": training[training.origin == "new"].elapsed_hours.sum(),
             "new_attempt_wall_hours": sum((datetime.fromisoformat(a["finished_at"]) - datetime.fromisoformat(a["started_at"])).total_seconds() / 3600 for a in new_attempts),
             "new_heldout_evaluation_seconds": heldout[heldout.origin == "new"].evaluation_seconds.sum(),
             "new_diagnostic_probes": int(training[training.origin == "new"].diagnostic_probes.sum()),
             "new_consistency_warnings": int(training[training.origin == "new"].consistency_warnings.sum()),
             "new_max_logprob_error": training[training.origin == "new"].max_logprob_error.max(),
             "final_task_solved_count": int(training.final_task_solved.sum()),
             "analysis_environment": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
             "note": "Historical sources differ intentionally. Equal seed labels do not imply matched model initializations. No training, inference, checkpoint loading, or runs writes."}
    (output / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    lines = ["# Recomputed heldout results", "", "Mean +/- sample SD across three training seeds.", ""]
    for policy in ("sample", "argmax"):
        lines += [f"## {policy}", "", "| Group | p=0 return | >=200 | p=.5 return | >=200 | p=1 return | >=200 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for group in GROUPS:
            cells = []
            for p in (0, .5, 1):
                r = grouped.query("group == @group and p == @p and policy == @policy").iloc[0]
                cells += [f"{r.mean_return_mean:.2f} +/- {r.mean_return_std:.2f}", f"{r.score_200_rate_mean:.1%}"]
            lines.append("| " + " | ".join([group] + cells) + " |")
        lines.append("")
    (output / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    make_figures(tables, assets)
    print(json.dumps({k: audit[k] for k in ("new_tasks", "historical_tasks", "unique_verified_artifacts",
                     "current_source_matches_new_batch", "episode_records_recomputed", "scalar_comparisons_passed")}, indent=2))
    print(tables["difference-groups"].query("p == 1").to_string(index=False))


def make_figures(tables, assets):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2})
    grouped, heldout = tables["heldout-groups"], tables["heldout-seeds"]
    training, periodic, episodes = tables["training-seeds"], tables["periodic-evaluations"], tables["heldout-episodes"]

    def save(fig, name):
        fig.savefig(assets / f"ed-{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, constrained_layout=True)
    for ax, policy in zip(axes, ("sample", "argmax")):
        for j, group in enumerate(GROUPS):
            d = grouped[(grouped.group == group) & (grouped.policy == policy)].sort_values("p")
            ax.errorbar(d.p + (j - 1.5) * .016, d.mean_return_mean, yerr=d.mean_return_std,
                        marker="o", capsize=3, color=COLORS[group], label=LABELS[group])
        ax.axhline(200, color="#a13c3c", ls="--", lw=1)
        ax.set(title=policy, xlabel="Sensor failure probability p", xticks=[0, .5, 1], ylabel="Heldout mean return")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Final checkpoints: means and sample SD across 3 training seeds")
    save(fig, "generalization")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True, constrained_layout=True)
    for ax, policy in zip(axes, ("sample", "argmax")):
        for x, (old, new) in enumerate((("PO01", "ED_PO01"), ("PO03", "ED_PO03"))):
            for seed, style in zip(SEEDS, ("o-", "s--", "^:")):
                pair = []
                for offset, group in enumerate((old, new)):
                    r = heldout.query("group == @group and seed == @seed and p == 1 and policy == @policy").iloc[0]
                    pair.append(r.mean_return)
                    ax.scatter(x * 3 + offset, r.mean_return, marker=style[0], color=COLORS[group], s=45, zorder=3)
                ax.plot([x * 3, x * 3 + 1], pair, style, color="#777777", alpha=.55, lw=1)
        for seed, style in zip(SEEDS, ("o-", "s--", "^:")):
            ax.plot([], [], style, color="#777777", label=f"seed {seed}")
        ax.legend(fontsize=8, loc="lower left")
        ax.axhline(200, color="#a13c3c", ls="--", lw=1)
        ax.set(title=policy, ylabel="p=1 heldout mean return", xticks=[0, 1, 3, 4],
               xticklabels=["PO01", "ED_PO01", "PO03", "ED_PO03"])
    fig.suptitle("Seed-level comparison (labels link seed numbers, not identical initializations)")
    save(fig, "p1-seeds")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, group in zip(axes.flat, GROUPS):
        twin = ax.twinx()
        for seed, style in zip(SEEDS, ("-", "--", ":")):
            e = periodic[(periodic.group == group) & (periodic.training_seed == seed)]
            ax.plot(e.global_step / 1e6, e.mean_return, style, color=COLORS[group], label=f"seed {seed}")
            twin.step(e.global_step / 1e6, e.sensor_failure_probability, where="post", color="#555555", alpha=.2)
        twin.set(ylim=(-.02, 1.05), ylabel="Evaluation p")
        twin.grid(False)
        ax.axhline(200, color="#a13c3c", ls="--", lw=1)
        ax.set(title=f"{group}: {LABELS[group]}", xlabel="Training steps (millions)", ylabel="Periodic mean return")
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Periodic sample evaluation: 20 episodes at current difficulty; unsmoothed")
    save(fig, "learning-curves")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    stages = tables["performance-stages"]
    stage_colors = plt.cm.viridis(np.linspace(.05, .95, 11))
    for i, (group, seed) in enumerate((g, s) for g in ("PO03", "ED_PO03") for s in SEEDS):
        part = stages[(stages.group == group) & (stages.seed == seed)].sort_values("stage")
        left = 0
        for stage, steps in zip(part.stage, part.steps):
            axes[0].barh(i, steps / 1e6, left=left, color=stage_colors[stage], label=f"p={stage/10:g}" if i == 0 else None)
            left += steps / 1e6
        entry = training[(training.group == group) & (training.seed == seed)].iloc[0].p1_entry_step
        e = periodic[(periodic.group == group) & (periodic.training_seed == seed) & (periodic.sensor_failure_probability == 1)]
        axes[1].plot((e.global_step - entry) / 1e6, e.mean_return, ("-", "--", ":")[seed - 42],
                     color=COLORS[group], label=f"{group}/{seed}")
    axes[0].set(yticks=range(6), yticklabels=[f"{g}/{s}" for g in ("PO03", "ED_PO03") for s in SEEDS],
                xlabel="Training steps (millions)", title="Recorded curriculum stage budgets")
    axes[0].legend(fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.13))
    axes[1].axhline(200, color="#a13c3c", ls="--", lw=1)
    axes[1].set(xlabel="Steps since p=1 entry (millions)", ylabel="Periodic mean return", title="Final stage: progress after entry")
    axes[1].legend(fontsize=8)
    save(fig, "curriculum")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    p1_returns = episodes.loc[episodes.p == 1, "return"]
    return_limits = (np.floor(p1_returns.min() / 100) * 100, np.ceil(p1_returns.max() / 100) * 100)
    for ax, policy in zip(axes[:2], ("sample", "argmax")):
        for group in GROUPS:
            data = np.sort(episodes.query("group == @group and p == 1 and policy == @policy")["return"].to_numpy())
            ax.step(data, np.arange(1, len(data) + 1) / len(data), where="post", color=COLORS[group], label=group)
        ax.axvline(200, color="#a13c3c", ls="--", lw=1)
        ax.set(title=f"p=1 / {policy}", xlabel="Episode return", ylabel="Empirical cumulative fraction", xlim=return_limits)
        ax.legend(fontsize=8)
    x = np.arange(8)
    bottom = np.zeros(8)
    labels = []
    for metric, color, label in (("positive_terminal_rate_mean", "#409974", "+100 terminal"),
                                 ("terminal_penalty_rate_mean", "#d06b57", "-100 terminal"),
                                 ("timeout_rate_mean", "#b2b7bd", "Timeout")):
        vals = []
        for group in GROUPS:
            for policy in ("sample", "argmax"):
                vals.append(grouped.query("group == @group and p == 1 and policy == @policy").iloc[0][metric])
        axes[2].bar(x, vals, bottom=bottom, color=color, label=label)
        bottom += vals
    labels = [f"{g}\n{p}" for g in GROUPS for p in ("sample", "argmax")]
    axes[2].set(xticks=x, xticklabels=labels, ylabel="Mean episode fraction", ylim=(0, 1.03), title="p=1 terminal outcomes")
    axes[2].tick_params(axis="x", labelsize=7, rotation=50)
    axes[2].legend(fontsize=8, loc="lower left")
    fig.suptitle("Return distributions pool 300 episodes for description, not 300 independent training runs")
    save(fig, "outcomes")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for group in GROUPS:
        t = training[training.group == group]
        axes[0, 0].bar(group, t.elapsed_hours.mean(), yerr=t.elapsed_hours.std(ddof=1), capsize=4, color=COLORS[group])
    axes[0, 0].set(ylabel="Training metadata elapsed hours", title="Equal environment-step budget, different compute")
    d = tables["diagnostics"]
    for group in ("ED_PO01", "ED_PO03"):
        for seed, style in zip(SEEDS, ("-", "--", ":")):
            part = d[(d.group == group) & (d.training_seed == seed)]
            axes[0, 1].plot(part.global_step / 1e6, part.encoder_value_grad_norm / part.encoder_policy_grad_norm,
                            style, color=COLORS[group], lw=1, alpha=.8, label=f"{group}/{seed}")
            axes[1, 0].bar(f"{group}\n{seed}", (part.shared_gradient_cosine < 0).mean(), color=COLORS[group])
            axes[1, 1].plot(part.global_step / 1e6, part.cross_attention_policy_grad_norm, style, color=COLORS[group], lw=1, alpha=.65)
    axes[0, 1].set(yscale="log", xlabel="Training steps (millions)", ylabel="Encoder value / policy gradient norm", title="Fixed probe; value loss includes vf_coef")
    axes[0, 1].legend(fontsize=7, ncol=2)
    axes[1, 0].set(ylabel="Fraction of probes with negative cosine", ylim=(0, .65), title="Shared gradients: opposing directions in 39%-55% of probes")
    axes[1, 0].tick_params(axis="x", labelsize=8)
    axes[1, 1].set(yscale="log", xlabel="Training steps (millions)", ylabel="Cross-attention policy gradient norm", title="Nonzero gradient is not an ablation of memory use")
    save(fig, "diagnostics-cost")


if __name__ == "__main__":
    main()
