"""Read experiment artifacts; report incomplete/incompatible runs without selecting a duplicate."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def csv_rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def number(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def training_arguments(parameters):
    """Freeze all public training arguments except budget/seed/sensor condition."""
    settings = {**parameters, **parameters["agent"]}
    for key in ("agent", "sensors", "seed", "total_timesteps"):
        settings.pop(key)
    arguments = []
    for key, value in sorted(settings.items()):
        if key == "use_previous_action":
            arguments.append("--previous-action" if value else "--no-previous-action")
        elif key == "normalize_advantage":
            arguments.append("--normalize-advantage" if value else "--no-normalize-advantage")
        else:
            arguments.extend(["--compile" if key == "compile_mode" else "--" + key.replace("_", "-"), str(value)])
    return arguments


def read_run(path):
    metadata = json.loads(path.read_text(encoding="utf-8"))
    parameters = metadata["parameters"]
    kind = metadata["kind"]
    if kind not in ("train", "evaluate"):
        return None
    issues = []
    record = {
        "path": str(path.parent.resolve()), "run_id": metadata.get("run_id"), "kind": kind,
        "experiment_id": metadata.get("experiment_id") or "unlabelled",
        "status": metadata.get("status"), "seed": parameters.get("seed"),
        "source_fingerprint": metadata.get("source_fingerprint"),
        "config_fingerprint": metadata.get("config_fingerprint"),
        "evaluation_protocol": metadata.get("evaluation_protocol"),
        "execution_fingerprint": digest({key: metadata.get(key) for key in (
            "precision", "device", "dependencies", "hardware")}),
        "training_parameters": parameters if kind == "train" else parameters.get("training_parameters", {}),
        "effective_total_timesteps": metadata.get("effective_total_timesteps") if kind == "train"
                                      else metadata.get("checkpoint_global_step"),
        "elapsed_seconds": metadata.get("elapsed_seconds"),
        "evaluation_fraction": metadata.get("evaluation_fraction"),
        "entered_final": metadata.get("entered_final"), "final_task_solved": metadata.get("final_task_solved"),
        "sensor_probability": None, "policy": None, "mean_return": None, "score_200_rate": None,
        "late_mean_return": None, "late_score_200_rate": None, "gate_pass": False, "timeout_rate": None,
        "sensor_mode": (parameters if kind == "train" else parameters.get("training_parameters", {}))
                        .get("sensors", {}).get("mode"),
    }
    if metadata.get("status") != "completed":
        issues.append(f"status={metadata.get('status')}")
    if not metadata.get("experiment_id"):
        issues.append("missing experiment-id")
    if not all(metadata.get(key) for key in ("source_fingerprint", "config_fingerprint", "evaluation_protocol")):
        issues.append("missing provenance")
    if metadata.get("config_fingerprint") != digest(parameters):
        issues.append("config fingerprint mismatch")
    if kind == "train":
        if not metadata.get("training_budget_completed") or not (path.parent / "checkpoints/latest.pt").is_file():
            issues.append("missing completed budget/checkpoint")
        evaluations = csv_rows(path.parent / "evaluation.csv")
        training = csv_rows(path.parent / "training.csv")
        episodes = csv_rows(path.parent / "episodes.csv")
        if not evaluations:
            issues.append("missing periodic evaluation")
        else:
            evaluations.sort(key=lambda row: int(row["update"]))
            late = evaluations[-5:]
            record.update(mean_return=number(evaluations[-1], "mean_return"),
                          score_200_rate=number(evaluations[-1], "score_200_rate"),
                          late_mean_return=statistics.mean(number(row, "mean_return") for row in late),
                          late_score_200_rate=statistics.mean(number(row, "score_200_rate") for row in late),
                          timeout_rate=number(evaluations[-1], "timeout_rate"),
                          gate_pass=len(evaluations) >= 2 and all(
                              number(row, "mean_return") >= 200 and number(row, "score_200_rate") >= 0.8
                              for row in evaluations[-2:]))
            if any(row.get("evaluation_protocol") != metadata.get("evaluation_protocol") for row in evaluations):
                issues.append("mixed evaluation protocol")
        record.update(total_optimizer_steps=sum(int(row["optimizer_steps"]) for row in training),
                      mean_unique_sample_coverage=statistics.mean(float(row["unique_sample_coverage"]) for row in training)
                                                   if training else None,
                      training_last50_return=statistics.mean(float(row["return"]) for row in episodes[-50:])
                                             if episodes else None)
        signature_parameters = {key: value for key, value in parameters.items() if key != "seed"}
    else:
        result_path = path.parent / "evaluation.json"
        if not result_path.exists():
            issues.append("missing evaluation.json")
            result = {}
        else:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        record.update(seed=parameters.get("training_seed"), sensor_probability=parameters.get("sensor_failure_probability"),
                      policy=parameters.get("policy"), mean_return=number(result, "mean_return"),
                      score_200_rate=number(result, "score_200_rate"), timeout_rate=number(result, "timeout_rate"))
        signature_parameters = {key: value for key, value in parameters.items()
                                if key not in ("checkpoint", "training_seed", "training_parameters")}
        signature_parameters["training_parameters"] = {key: value for key, value in record["training_parameters"].items()
                                                       if key != "seed"}
        provenance = metadata.get("training_provenance") or {}
        if not provenance or result.get("training_provenance") != provenance:
            issues.append("missing/inconsistent training provenance")
        if result.get("evaluation_protocol") != metadata.get("evaluation_protocol"):
            issues.append("evaluation protocol mismatch")
        signature_parameters["training_source"] = provenance.get("source_fingerprint")
    record["comparison_fingerprint"] = digest({
        "parameters": signature_parameters, "source": record["source_fingerprint"],
        "protocol": record["evaluation_protocol"], "precision": metadata.get("precision"),
        "device": metadata.get("device"), "dependencies": metadata.get("dependencies"),
        "hardware": metadata.get("hardware"), "budget": record["effective_total_timesteps"],
    })
    record["issues"] = issues
    return record


def aggregate(records, expected_seeds):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["kind"], record["experiment_id"], record["sensor_probability"], record["policy"])].append(record)
    groups = []
    for (kind, experiment, probability, policy), runs in sorted(grouped.items(), key=lambda pair: str(pair[0])):
        seeds = [run["seed"] for run in runs]
        problems = sorted({issue for run in runs for issue in run["issues"]})
        missing = sorted(set(expected_seeds) - set(seeds))
        duplicates = sorted({seed for seed in seeds if seeds.count(seed) > 1}, key=str)
        unexpected = sorted(set(seeds) - set(expected_seeds), key=str)
        if missing:
            problems.append(f"missing seeds: {missing}")
        if duplicates:
            problems.append(f"duplicate seeds: {duplicates}")
        if unexpected:
            problems.append(f"unexpected seeds: {unexpected}")
        if len({run["comparison_fingerprint"] for run in runs}) != 1:
            problems.append("inconsistent config/source/budget/protocol/hardware")
        group = {"kind": kind, "experiment_id": experiment, "sensor_probability": probability,
                 "policy": policy, "run_count": len(runs), "seeds": seeds, "valid": not problems,
                 "issues": problems, "qualifying_seeds": sum(run["gate_pass"] for run in runs),
                 "runs": [run["path"] for run in runs]}
        # Invalid groups remain visible but never receive pooled statistics.
        for key in ("mean_return", "score_200_rate", "late_mean_return", "late_score_200_rate", "timeout_rate"):
            values = [run[key] for run in runs if run.get(key) is not None]
            valid = not problems and len(values) == len(runs)
            group[f"{key}_mean"] = statistics.mean(values) if valid else None
            group[f"{key}_std"] = statistics.stdev(values) if valid and len(values) > 1 else 0.0 if valid else None
        groups.append(group)
    return groups


def select_recipe(groups, records):
    incomplete_full = [group for group in groups if group["kind"] == "train" and not group["valid"]
                       and any(run["sensor_mode"] == "full" for run in records if run["path"] in group["runs"])]
    if incomplete_full:
        return {"eligible": False, "reason": "incomplete/invalid full-observation conditions; resolve before selection"}
    full = [group for group in groups if group["kind"] == "train" and group["valid"]
            and all(run["sensor_mode"] == "full" for run in records if run["path"] in group["runs"])]
    comparability = set()
    for group in full:
        run = next(run for run in records if run["path"] == group["runs"][0])
        config = run["training_parameters"]
        comparability.add(digest({"source": run["source_fingerprint"], "budget": run["effective_total_timesteps"],
                                  "execution": run["execution_fingerprint"],
                                  "protocol": run["evaluation_protocol"], "seeds": sorted(group["seeds"]),
                                  **{key: config[key] for key in ("eval_interval", "eval_episodes", "eval_seed",
                                                               "eval_policy", "eval_num_envs", "max_episode_steps")}}))
    candidates = [group for group in full if group["qualifying_seeds"] >= 2]
    if len(comparability) > 1:
        return {"eligible": False, "reason": "full-observation batches have inconsistent source/budget/evaluation"}
    if not candidates:
        return {"eligible": False, "reason": "no complete full-observation condition passes the two-seed gate"}
    best = min(candidates, key=lambda group: (-group["late_mean_return_mean"], -group["late_score_200_rate_mean"],
                                             group["experiment_id"]))
    run = next(run for run in records if run["path"] == best["runs"][0])
    return {"eligible": True, "experiment_id": best["experiment_id"], "parameters": run["training_parameters"],
            "source_fingerprint": run["source_fingerprint"], "evaluation_protocol": run["evaluation_protocol"],
            "training_seeds": sorted(best["seeds"]), "training_run_paths": best["runs"],
            "arguments": training_arguments(run["training_parameters"]),
            "rule": "at least 2 seeds pass last two evaluations; rank last-five mean, score200 rate, then ID"}


def write_csv(path, records):
    fields = list(dict.fromkeys(key for row in records for key in row))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def summarize(roots, output_dir, expected_seeds=(42, 43, 44), conditions=None, *,
              run_dirs=None, run_issues=None):
    """Use exact run_dirs for managed attempts; roots keeps the original recursive CLI behavior."""
    if len(set(expected_seeds)) != len(expected_seeds) or not expected_seeds:
        raise ValueError("expected-seeds must be a nonempty unique list")
    if run_dirs is not None and roots:
        raise ValueError("roots and run_dirs are mutually exclusive")
    metadata_files = set()
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            raise ValueError(f"root does not exist: {root}")
        metadata_files.update(path.resolve() for path in root.rglob("metadata.json"))
    if run_dirs is not None:
        metadata_files.update((Path(path) / "metadata.json").resolve() for path in run_dirs)
    run_issues = {str(Path(path).resolve()): issues for path, issues in (run_issues or {}).items()}
    records, unreadable = [], []
    for path in sorted(metadata_files):
        try:
            record = read_run(path)
            if record is not None and (not conditions or record["experiment_id"] in conditions):
                record["issues"].extend(run_issues.get(str(path.parent), []))
                records.append(record)
        except (OSError, ValueError, KeyError, TypeError) as error:
            unreadable.append({"path": str(path), "error": str(error)})
    groups = aggregate(records, expected_seeds)
    selection = select_recipe(groups, records)
    # A leftover held-out result cannot stand in for a missing training condition.
    condition_records = [record for record in records if record["kind"] == "train"]
    if not condition_records:
        condition_records = records
    missing_conditions = sorted(set(conditions or []) - {record["experiment_id"] for record in condition_records})
    if missing_conditions:
        selection = {"eligible": False, "reason": f"missing conditions: {missing_conditions}"}
    if unreadable:
        selection = {"eligible": False, "reason": "unreadable artifacts; resolve before selection"}
    result = {"expected_seeds": list(expected_seeds), "runs": records, "groups": groups,
              "unreadable": unreadable, "missing_conditions": missing_conditions, "selection": selection}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_csv(output / "runs.csv", records)
    write_csv(output / "groups.csv", groups)
    lines = ["# Experiment summary", "", "Statistics across training seeds; invalid groups are not pooled.", "",
             "| Kind | Condition | p / policy | Seeds | Return mean ± std | Score ≥200 | Issues |",
             "|---|---|---|---|---|---|---|"]
    for group in groups:
        mean, std = group["mean_return_mean"], group["mean_return_std"]
        score = group["score_200_rate_mean"]
        lines.append(f"| {group['kind']} | {group['experiment_id']} | {group['sensor_probability']} / {group['policy']} | "
                     f"{group['seeds']} | {f'{mean:.2f} ± {std:.2f}' if mean is not None else 'pending'} | "
                     f"{f'{score:.1%}' if score is not None else 'pending'} | {'; '.join(group['issues'])} |")
    lines.extend(["", "Selection: " + (selection.get("experiment_id") if selection["eligible"] else selection["reason"]),
                  "", f"Unreadable artifacts: {len(unreadable)}; details in summary.json."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--conditions", nargs="+")
    args = parser.parse_args()
    try:
        result = summarize(args.root, args.output_dir, args.expected_seeds, args.conditions)
        print(f"{len(result['runs'])} runs; {len(result['groups'])} groups; "
              f"{len(result['unreadable'])} unreadable; selection eligible={result['selection']['eligible']}")
        return 0
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
