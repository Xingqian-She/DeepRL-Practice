"""Read-only historical ED reference; write evidence only into the new summary."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import statistics

from config import AgentConfig


GROUP_PAIRS = {"DYT_SNAKE_ED_PO01": "ED_PO01", "DYT_SNAKE_ED_PO03": "ED_PO03"}
COMPONENT_FIELDS = ("norm_type", "ffn_type", "dyt_alpha_init", "snake_alpha_init", "snake_eps")


def normalized_recipe(parameters):
    result = deepcopy(parameters)
    result["agent"] = asdict(AgentConfig(**result["agent"]))
    for name in COMPONENT_FIELDS:
        result["agent"].pop(name)
    return result


def write_comparison(root, state, historical_root=None):
    from run_experiments import attempt_path, current_attempt, read_json, validate_attempt, write_json

    root = Path(root)
    historical_root = Path(historical_root or root.parent / "encoder-decoder-10m-v1")
    result = {"historical_batch": historical_root.name, "new_batch": root.name,
              "source_fingerprints": {"new": state["source_fingerprint"]},
              "rows": [], "training": [], "differences": [], "issues": [], "complete": False}
    checked_tasks = {}

    def checked(batch, batch_state, task):
        key = (str(batch), task["spec"]["key"])
        if key not in checked_tasks:
            attempt = current_attempt(task)
            if task["status"] != "completed" or attempt is None:
                raise ValueError(f"Incomplete task: {key}")
            required = ({"metadata.json", "checkpoints/latest.pt"} if task["spec"]["kind"] == "train"
                        else {"metadata.json", "evaluation.json", "episodes.csv"})
            if not required.issubset(attempt.get("artifact_hashes", {})):
                raise ValueError(f"Missing registered artifact hashes: {key}")
            metadata, _ = validate_attempt(batch, batch_state, task)
            checked_tasks[key] = (attempt_path(batch, attempt) / "run", metadata)
        return checked_tasks[key]

    selected = [group for group in GROUP_PAIRS if any(
        t["spec"]["group"] == group for t in state["tasks"].values())]
    try:
        historical = read_json(historical_root / "state.json")
        result["source_fingerprints"]["historical"] = historical["source_fingerprint"]
        if historical["contract"]["smoke"] or state["contract"]["smoke"]:
            raise ValueError("Smoke batches cannot provide formal comparisons")
        for field in ("seeds", "evaluation"):
            if historical["contract"][field] != state["contract"][field]:
                raise ValueError(f"Historical contract differs: {field}")
        if historical["environment"] != state["environment"]:
            raise ValueError("Historical environment differs")
        for new_group in selected:
            old_group = GROUP_PAIRS[new_group]
            for seed in state["contract"]["seeds"]:
                new_train = state["tasks"][f"train--{new_group}--seed{seed}"]
                old_train = historical["tasks"][f"train--{old_group}--seed{seed}"]
                new_parameters, old_parameters = (t["spec"]["parameters"] for t in (new_train, old_train))
                if normalized_recipe(new_parameters) != normalized_recipe(old_parameters):
                    raise ValueError(f"Unexpected recipe difference: {new_group}/seed{seed}")
                if ((new_parameters["agent"]["norm_type"], new_parameters["agent"]["ffn_type"])
                        != ("dyt", "swi_snake_glu")):
                    raise ValueError("New reference comparison requires DyT + Swi-Snake-GLU")
                old_agent = AgentConfig(**old_parameters["agent"])
                if (old_agent.norm_type, old_agent.ffn_type) != ("rmsnorm", "swiglu"):
                    raise ValueError("Historical reference must use RMSNorm + SwiGLU")
                for batch, bs, task in ((historical_root, historical, old_train), (root, state, new_train)):
                    if batch == root and task["status"] != "completed":
                        continue
                    _, meta = checked(batch, bs, task)
                    result["training"].append({"group": task["spec"]["group"], "seed": seed,
                        "run_id": meta["run_id"], "source_fingerprint": bs["source_fingerprint"],
                        "elapsed_seconds": meta.get("elapsed_seconds"),
                        "parameters": meta.get("model", {}).get("parameters"),
                        "p1_training_steps": meta.get("p1_training_steps"),
                        "p1_entry_global_step": meta.get("p1_entry_global_step"),
                        "p1_entry_learning_rate": meta.get("p1_entry_learning_rate")})
                for probability in state["contract"]["evaluation"]["probabilities"]:
                    for policy in state["contract"]["evaluation"]["policies"]:
                        suffix = f"seed{seed}--p{probability:g}--{policy}"
                        new_eval = state["tasks"][f"eval--{new_group}--{suffix}"]
                        if new_eval["status"] != "completed":
                            continue
                        old_eval = historical["tasks"][f"eval--{old_group}--{suffix}"]
                        for batch, bs, train_task, task in (
                            (historical_root, historical, old_train, old_eval), (root, state, new_train, new_eval)
                        ):
                            # Certify both the checkpoint dependency and the held-out data.
                            checked(batch, bs, train_task)
                            run, meta = checked(batch, bs, task)
                            evaluation = read_json(run / "evaluation.json")
                            result["rows"].append({"group": task["spec"]["group"], "seed": seed,
                                "p": probability, "policy": policy, "run_id": meta["run_id"],
                                "source_fingerprint": bs["source_fingerprint"],
                                "mean_return": evaluation["mean_return"],
                                "score_200_rate": evaluation["score_200_rate"]})
                        old, new = result["rows"][-2:]
                        result["differences"].append({"group": new_group, "reference": old_group,
                            "seed": seed, "p": probability, "policy": policy,
                            "mean_return_delta": new["mean_return"] - old["mean_return"],
                            "score_200_rate_delta": new["score_200_rate"] - old["score_200_rate"]})
        evaluation = state["contract"]["evaluation"]
        expected = len(selected) * len(state["contract"]["seeds"]) * len(evaluation["probabilities"]) * len(evaluation["policies"])
        result["complete"] = bool(selected) and len(result["differences"]) == expected
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(rows=[], training=[], differences=[], issues=[str(error)])

    def fmt(values, percent=False):
        if len(values) != len(state["contract"]["seeds"]):
            return "pending"
        mean, sd = statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0
        return f"{mean:.1%} ± {sd:.1%}" if percent else f"{mean:.2f} ± {sd:.2f}"

    lines = ["# DyT + Snake 与历史 Encoder–Decoder 对照", "",
             "探索性历史比较，不能分离 DyT 与 Snake 的贡献，也不视为同期消融。",
             "主要指标为 p=1/sample；均值和样本标准差按独立训练种子计算，评估 seed=20000 已用于历史分析。",
             f"完整：{result['complete']}；历史批次：{historical_root.name}。",
             f"来源指纹：{result['source_fingerprints']}", "",
             "| Group | p | Policy | Seeds | Return mean ± sample SD | ≥200 mean ± sample SD |",
             "|---|---:|---|---|---:|---:|"]
    groups = {}
    for row in result["rows"]:
        groups.setdefault((row["group"], row["p"], row["policy"]), []).append(row)
    for (group, p, policy), rows in sorted(groups.items()):
        lines.append(f"| {group} | {p:g} | {policy} | {sorted(r['seed'] for r in rows)} | "
                     f"{fmt([r['mean_return'] for r in rows])} | {fmt([r['score_200_rate'] for r in rows], True)} |")
    lines += ["", "| Group | Seed | Seconds | Parameters | p=1 steps | p=1 entry step | Entry LR |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["training"]:
        count = (row["parameters"] or {}).get("total")
        lines.append(f"| {row['group']} | {row['seed']} | {row['elapsed_seconds']} | {count} | "
                     f"{row['p1_training_steps']} | {row['p1_entry_global_step']} | {row['p1_entry_learning_rate']} |")
    lines += ["", "逐种子差值和训练来源见 dyt-snake-comparison.json；课程轨迹另见 curriculum.csv。"]
    if not result["complete"]:
        lines += ["", "对照尚不完整；不将已完成的种子作为完整三种子结果。", *result["issues"]]
    output = root / "summary"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "dyt-snake-comparison.json", result)
    (output / "dyt-snake-comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
