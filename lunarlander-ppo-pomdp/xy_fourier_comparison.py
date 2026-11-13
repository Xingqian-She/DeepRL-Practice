"""Read registered historical attempts without requiring their source to equal today's.

Only writes into the new batch summary; historical files are always read-only.
"""

from copy import deepcopy
import json
from pathlib import Path
import statistics

from summarize_experiments import digest


GROUP_PAIRS = {"XY_PO01": "PO01", "XY_PO03": "PO03"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalized(parameters):
    result = deepcopy(parameters)
    result["agent"].setdefault("decoder_layers", 4)
    result["agent"].pop("xy_fourier", None)
    return result


def write_comparison(root, state, historical_root=None):
    """Compare completed held-outs and retain incomplete/error states explicitly."""
    from run_experiments import current_attempt, file_hash

    root = Path(root)
    historical_root = Path(historical_root or root.parent / "course-10m-v1")
    result = {"historical_batch": historical_root.name, "new_batch": root.name,
              "source_fingerprints": {"new": state["source_fingerprint"]},
              "rows": [], "issues": [], "complete": False}

    def checked(batch, batch_state, task):
        if task["status"] != "completed":
            raise ValueError(f"Incomplete task: {task['spec']['key']}")
        attempt = current_attempt(task)
        run = (batch / attempt["path"] / "run").resolve()
        if not run.is_relative_to(batch.resolve()):
            raise ValueError("Attempt path outside batch")
        hashes = attempt.get("artifact_hashes", {})
        required = ("metadata.json", "evaluation.json") if task["spec"]["kind"] == "evaluate" else ("metadata.json",)
        for name in required:
            if name not in hashes or file_hash(run / name) != hashes[name]:
                raise ValueError(f"Artifact hash mismatch: {run / name}")
        meta = read_json(run / "metadata.json")
        if (meta["source_fingerprint"] != batch_state["source_fingerprint"]
                or meta["config_fingerprint"] != digest(meta["parameters"])
                or meta["status"] != "completed" or meta["run_id"] != attempt["run_id"]
                or meta["experiment_id"] != task["spec"]["group"]):
            raise ValueError(f"Metadata mismatch: {run}")
        if task["spec"]["kind"] == "train" and meta["parameters"] != task["spec"]["parameters"]:
            raise ValueError(f"Training parameters mismatch: {run}")
        return run, meta

    try:
        historical = read_json(historical_root / "state.json")
        result["source_fingerprints"]["historical"] = historical["source_fingerprint"]
        if historical["contract"]["smoke"] or state["contract"]["smoke"]:
            raise ValueError("Smoke batches cannot provide formal comparisons")
        if (historical["contract"]["seeds"] != state["contract"]["seeds"]
                or historical["contract"]["evaluation"] != state["contract"]["evaluation"]):
            raise ValueError("Historical evaluation contract differs")
        for new_group, old_group in GROUP_PAIRS.items():
            for seed in state["contract"]["seeds"]:
                old_task = historical["tasks"][f"train--{old_group}--seed{seed}"]
                new_task = state["tasks"].get(f"train--{new_group}--seed{seed}")
                if new_task is None:
                    continue
                if normalized(old_task["spec"]["parameters"]) != normalized(new_task["spec"]["parameters"]):
                    raise ValueError(f"Unexpected recipe difference: {new_group}/seed{seed}")
                if not new_task["spec"]["parameters"]["agent"].get("xy_fourier"):
                    raise ValueError("New batch does not enable XY Fourier features")
                _, old_train_meta = checked(historical_root, historical, old_task)
                if new_task["status"] == "completed":
                    _, new_train_meta = checked(root, state, new_task)
                    for key in ("precision", "device", "dependencies", "hardware", "evaluation_protocol"):
                        if old_train_meta[key] != new_train_meta[key]:
                            raise ValueError(f"Historical environment differs: {key}")
                for probability in state["contract"]["evaluation"]["probabilities"]:
                    for policy in state["contract"]["evaluation"]["policies"]:
                        suffix = f"seed{seed}--p{probability:g}--{policy}"
                        old_eval = historical["tasks"][f"eval--{old_group}--{suffix}"]
                        new_eval = state["tasks"].get(f"eval--{new_group}--{suffix}")
                        if new_eval is None or new_eval["status"] != "completed":
                            continue
                        for batch, bs, task in ((historical_root, historical, old_eval), (root, state, new_eval)):
                            run, _ = checked(batch, bs, task)
                            evaluation = read_json(run / "evaluation.json")
                            result["rows"].append({"group": task["spec"]["group"], "seed": seed,
                                                   "p": probability, "policy": policy,
                                                   "mean_return": evaluation["mean_return"],
                                                   "score_200_rate": evaluation["score_200_rate"],
                                                   "source_fingerprint": bs["source_fingerprint"]})
        selected = {t["spec"]["group"] for t in state["tasks"].values() if t["spec"]["kind"] == "train"}
        evaluation = state["contract"]["evaluation"]
        expected = len(selected) * len(state["contract"]["seeds"]) * len(evaluation["probabilities"]) * len(evaluation["policies"]) * 2
        result["complete"] = len(result["rows"]) == expected
    except (OSError, ValueError, KeyError, TypeError) as error:
        result["issues"].append(str(error))
        result["rows"] = []  # Do not publish partial evidence after a validation failure.

    groups = {}
    for row in result["rows"]:
        groups.setdefault((row["group"], row["p"], row["policy"]), []).append(row)
    lines = ["# XY Fourier 与历史 Stage2 对照", "",
             "探索性比较：各批次按自己的源码指纹校验；源码不同，不能视为严格同期对照。",
             "主要指标为 p=1/sample；均值和样本标准差按三个训练种子计算。留出种子已用于历史分析。", "",
             f"完整：{result['complete']}；历史批次：{historical_root.name}。",
             f"源码指纹：{json.dumps(result['source_fingerprints'], ensure_ascii=False)}", "",
             "| Group | p | Policy | Seeds | Return mean ± sample SD | ≥200 mean ± sample SD |",
             "|---|---:|---|---|---:|---:|"]
    for (group, probability, policy), rows in sorted(groups.items()):
        def fmt(key, percent=False):
            values = [r[key] for r in rows]
            if len(values) != len(state["contract"]["seeds"]):
                return "pending"
            mean, sd = statistics.mean(values), statistics.stdev(values)
            return f"{mean:.1%} ± {sd:.1%}" if percent else f"{mean:.2f} ± {sd:.2f}"
        lines.append(f"| {group} | {probability:g} | {policy} | {sorted(r['seed'] for r in rows)} | "
                     f"{fmt('mean_return')} | {fmt('score_200_rate', True)} |")
    lines += ["", "课程推进、进入 p=1 时的学习率和 p=1 训练预算见 curriculum.csv；"
              "相位最大值及阈值次数见 training.csv、evaluation.csv 和各 evaluation.json。"]
    if not result["complete"]:
        lines += ["", "尚未完成全部对照；已完成的种子不代替完整三种子结果。"]
    lines += ["", *result["issues"]]
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    (output / "xy-comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "xy-comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
