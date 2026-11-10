"""Freeze the validated FO04 recipe and run the three observation curricula."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import subprocess
import sys

import run_experiments as managed
from summarize_experiments import aggregate, read_run, select_recipe, write_csv


PROJECT = managed.PROJECT
TRAINING_BUDGET = 10_000_000
SEEDS = [42, 43, 44]
FO_GROUPS = [f"FO{i:02d}" for i in range(1, 8)]
EVALUATION = {"episodes": 100, "seed": 20000, "num_envs": 8,
              "probabilities": [0, .5, 1], "policies": ["sample", "argmax"]}
COURSES = {
    "PO01": {"mode": "partial", "schedule": "fixed", "start_fraction": .1, "end_fraction": .8},
    "PO02": {"mode": "curriculum", "schedule": "fixed", "start_fraction": .1, "end_fraction": .8},
    "PO03": {"mode": "curriculum", "schedule": "performance", "start_fraction": .1, "end_fraction": .8},
}


def coordinator_fingerprint():
    paths = list(Path(__file__).parent.glob("*.py")) + [
        PROJECT / "run_stage2.ps1", PROJECT / "run_experiments.ps1"]
    return managed.digest({p.relative_to(PROJECT).as_posix(): managed.file_hash(p) for p in sorted(paths)})


@contextmanager
def baseline_lock(root):
    """Read-only shared lease; the stage-one writer uses an exclusive flock."""
    import fcntl

    with (root / ".lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("第一阶段批次正在运行或分析，请等待其完成") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def coordinator_lock(root):
    """Cover snapshot creation and reporting as well as the managed task loop."""
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".stage2.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("该第二阶段批次已有调度器运行") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def baseline_parameters(base, group, seed):
    value = deepcopy(base)
    value["seed"] = seed
    if group in ("FO03", "FO04"):
        value["agent"]["backbone_mode"] = "separate"
    if group in ("FO02", "FO04"):
        value["lr_schedule"] = "linear"
    if group == "FO05":
        value["normalize_advantage"] = False
    if group == "FO06":
        value["value_clip_coef"] = .2
    if group == "FO07":
        value["ent_coef_schedule"] = "linear"
    return value


def inspect_baseline(root):
    """Validate exact current attempts without editing the first-stage state or reports."""
    from runtime import source_fingerprint

    state = managed.load_state(root)
    contract = state["contract"]
    if contract["smoke"] or contract["seeds"] != SEEDS or contract["evaluation"] != EVALUATION:
        raise ValueError("第一阶段须为正式七组对照、seed=42/43/44 和约定的 100 回合留出协议")
    if state["source_fingerprint"] != source_fingerprint():
        raise ValueError("第一阶段与当前训练源码指纹不同，不能直接复用参考成绩")
    base = state["tasks"]["train--FO01--seed42"]["spec"]["parameters"]
    if (base["total_timesteps"] != TRAINING_BUDGET or base["sensors"]["mode"] != "full"
            or base["agent"]["backbone_mode"] != "shared" or base["lr_schedule"] != "constant"
            or not base["normalize_advantage"] or base["value_clip_coef"] != 0
            or base["ent_coef_schedule"] != "constant"):
        raise ValueError("第一阶段 FO01 预算或组件基线不符合阶段约定")
    evidence, references, records = [], [], []

    def checked(key):
        task = state["tasks"].get(key)
        if task is None or task["status"] != "completed":
            raise ValueError(f"第一阶段缺少已完成任务：{key}")
        metadata, hashes = managed.validate_attempt(root, state, task)
        attempt = managed.current_attempt(task)
        if attempt.get("run_id") != metadata["run_id"]:
            raise ValueError(f"第一阶段任务来源不一致：{key}")
        path = managed.attempt_path(root, attempt) / "run"
        record = read_run(path / "metadata.json")
        entry = {"task": key, "attempt": attempt["number"], "path": path.relative_to(root).as_posix(),
                 "run_id": metadata["run_id"], "spec_fingerprint": managed.digest(task["spec"]),
                 "artifact_hashes": hashes}
        return task, entry, record

    for group in FO_GROUPS:
        for seed in SEEDS:
            task, entry, record = checked(f"train--{group}--seed{seed}")
            if task["spec"]["parameters"] != baseline_parameters(base, group, seed):
                raise ValueError(f"第一阶段组件对照参数不一致：{group}/seed{seed}")
            evidence.append(entry)
            records.append(record)
            if group == "FO04":
                references.append(entry)
    selection = select_recipe(aggregate(records, SEEDS), records)
    if not selection["eligible"] or selection.get("experiment_id") != "FO04":
        raise ValueError(f"第一阶段未按既定规则选出 FO04：{selection.get('reason', selection.get('experiment_id'))}")
    for seed in SEEDS:
        for probability in EVALUATION["probabilities"]:
            for policy in EVALUATION["policies"]:
                key = f"eval--FO04--seed{seed}--p{probability:g}--{policy}"
                task, entry, _ = checked(key)
                expected = {"episodes": EVALUATION["episodes"], "seed": EVALUATION["seed"],
                            "num_envs": EVALUATION["num_envs"], "policy": policy,
                            "sensor_failure_probability": probability, "device": base["device"]}
                if (task["spec"]["parameters"] != expected or task["spec"]["seed"] != seed
                        or task["spec"]["depends_on"] != f"train--FO04--seed{seed}"):
                    raise ValueError(f"FO04 留出协议不一致：{key}")
                references.append(entry)
    return {"schema_version": 1, "stage": 2, "baseline_batch": root.name, "experiment_id": "FO04",
            "parameters": selection["parameters"], "training_seeds": list(SEEDS),
            "source_fingerprint": state["source_fingerprint"], "environment": state["environment"],
            "evaluation": deepcopy(EVALUATION), "courses": deepcopy(COURSES),
            "selection_rule": selection["rule"], "training_evidence": evidence,
            "reference_runs": references, "coordinator_fingerprint": coordinator_fingerprint()}


def build_plan(recipe, groups=None, *, smoke=False, device=None):
    from config import AgentConfig, TrainConfig
    from sensors import SensorConfig

    selected = list(COURSES) if groups is None else groups
    if not selected or len(set(selected)) != len(selected) or any(g not in COURSES for g in selected):
        raise ValueError("第二阶段组别只能为不重复的 PO01、PO02、PO03")
    defaults = deepcopy(recipe["parameters"])
    defaults.pop("seed")
    evaluation = deepcopy(recipe["evaluation"])
    if smoke:
        defaults = managed.merge(defaults, {
            "agent": {"hidden_dim": 32, "num_layers": 2, "num_heads": 2, "context_length": 4},
            "total_timesteps": 64, "num_envs": 2, "num_steps": 16, "sequence_length": 6,
            "sequences_per_minibatch": 3, "update_epochs": 2, "max_episode_steps": 16,
            "save_interval": 1, "diagnostic_interval": 1, "eval_interval": 1,
            "eval_episodes": 2, "eval_num_envs": 2, "device": "cpu", "compile_mode": "off"})
        evaluation.update(episodes=2, num_envs=2)
    if device:
        defaults["device"] = device
    if not smoke and defaults["device"] != recipe["environment"]["device"]:
        raise ValueError("正式第二阶段的设备必须与第一阶段参考一致；CPU 烟测请使用 -Smoke")

    def normalize(value, seed):
        value = deepcopy(value)
        return asdict(TrainConfig(agent=AgentConfig(**value.pop("agent")),
                                  sensors=SensorConfig(**value.pop("sensors")), seed=seed, **value))

    training, heldout = [], []
    for group in selected:
        for seed in recipe["training_seeds"]:
            config = normalize({**defaults, "sensors": recipe["courses"][group]}, seed)
            key = f"train--{group}--seed{seed}"
            training.append({"key": key, "kind": "train", "group": group, "seed": seed, "parameters": config})
            for probability in evaluation["probabilities"]:
                for policy in evaluation["policies"]:
                    heldout.append({"key": f"eval--{group}--seed{seed}--p{probability:g}--{policy}",
                                    "kind": "evaluate", "group": group, "seed": seed, "depends_on": key,
                                    "training_parameters": config,
                                    "parameters": {"episodes": evaluation["episodes"], "seed": evaluation["seed"],
                                                   "num_envs": evaluation["num_envs"], "policy": policy,
                                                   "sensor_failure_probability": probability, "device": config["device"]}})
    contract = {"seeds": recipe["training_seeds"], "training_defaults": normalize(defaults, SEEDS[0]),
                "evaluation": evaluation, "smoke": smoke,
                "stage2": {"baseline_batch": recipe["baseline_batch"], "recipe_fingerprint": managed.digest(recipe),
                           "coordinator_fingerprint": recipe["coordinator_fingerprint"]}}
    return {"contract": contract, "groups": list(selected), "tasks": training + heldout}


def recipe_path(root, smoke):
    return root / ("smoke-recipe.json" if smoke else "frozen-recipe.json")


def check_snapshot(root, recipe, plan):
    path = recipe_path(root, plan["contract"]["smoke"])
    if (root / "state.json").exists():
        state = managed.load_state(root)
        if "stage2" not in state["contract"]:
            raise ValueError("目标目录不是第二阶段批次；请使用 -Batch 新名称")
        if not path.is_file():
            raise ValueError("第二阶段已有状态但缺少配置快照；拒绝重建来源，请使用新批次")
        # Work on a copy: validation never appends tasks or alters the saved state.
        managed.register_plan(deepcopy(state), plan, recipe["source_fingerprint"], state["environment"])
    if path.exists():
        frozen = managed.read_json(path)
        frozen.pop("frozen_at", None)
        if frozen != recipe:
            raise ValueError("冻结配置、参考产物或协调代码已变化；请使用 -Batch 新名称")


def freeze(root, recipe, plan):
    check_snapshot(root, recipe, plan)
    path = recipe_path(root, plan["contract"]["smoke"])
    if not path.exists():
        managed.write_json(path, {**recipe, "frozen_at": managed.now()})


def curriculum_record(path, metadata):
    config, progress = metadata["parameters"], metadata.get("curriculum", {})
    phase_steps = progress.get("stage_steps")
    basis = "recorded performance stages"
    if config["sensors"]["schedule"] != "performance":
        # Count the exact scheduled p for each collected vector step, including partial runs.
        from sensors import SensorConfig
        sensors = SensorConfig(**config["sensors"])
        counts = {"p0": 0, "ramp": 0, "p1": 0}
        budget = config["total_timesteps"] // (config["num_envs"] * config["num_steps"]) * config["num_envs"] * config["num_steps"]
        for step in range(0, metadata.get("last_completed_step", metadata.get("global_step", 0)), config["num_envs"]):
            probability = sensors.failure_probability(step, budget)
            counts["p0" if probability == 0 else "p1" if probability == 1 else "ramp"] += config["num_envs"]
        phase_steps, basis = counts, "derived from configured schedule and completed vector steps"
    return {"experiment_id": metadata["experiment_id"], "seed": config["seed"],
            "status": metadata["status"], "path": str(path), "global_step": metadata.get("global_step"),
            "sensor_mode": config["sensors"]["mode"], "schedule": config["sensors"]["schedule"],
            "final_probability": progress.get("probability"), "entered_final": metadata.get("entered_final"),
            "final_task_solved": metadata.get("final_task_solved"), "stage_steps": phase_steps,
            "stage_steps_basis": basis, "transitions": progress.get("transitions", [])}


def report(root, state, recipe, baseline):
    """Compare common held-out conditions; periodic scores never rank different terminal p."""
    records, unreadable, curricula, task_rows = [], [], [], []
    smoke = state["contract"]["smoke"]

    def include(path, origin, issues=()):
        try:
            record = read_run(path / "metadata.json")
            record["origin"] = origin
            record["issues"].extend(issues)
            records.append(record)
            if record["kind"] == "train" and origin == "new":
                curricula.append(curriculum_record(path, managed.read_json(path / "metadata.json")))
        except (OSError, ValueError, KeyError, TypeError) as error:
            unreadable.append({"path": str(path), "error": str(error)})

    if not smoke:
        for entry in recipe["reference_runs"]:
            path = managed.attempt_path(baseline, {"path": entry["path"]})
            problems = []
            for name, expected in entry["artifact_hashes"].items():
                target = (path / name).resolve()
                if not target.is_relative_to(path.resolve()) or not target.is_file() or managed.file_hash(target) != expected:
                    problems.append(f"reference artifact changed: {name}")
            include(path, "reused", problems)
    for key, task in state["tasks"].items():
        attempt = managed.current_attempt(task)
        path, metadata = None, {}
        if attempt:
            path = managed.attempt_path(root, attempt) / "run"
            include(path, "new", [] if task["status"] == "completed" else ["managed task: " + task["reason"]])
            try:
                metadata = managed.read_json(path / "metadata.json")
            except (OSError, ValueError):
                pass
        task_rows.append({"task": key, "kind": task["spec"]["kind"], "group": task["spec"]["group"],
                          "training_seed": task["spec"]["seed"], "status": task["status"],
                          "attempt": task["current_attempt"], "global_step": metadata.get("global_step"),
                          "elapsed_seconds": metadata.get("elapsed_seconds"), "reason": task["reason"],
                          "run_dir": str(path) if path else None})
    groups = aggregate(records, state["contract"]["seeds"])
    conditions = sorted({task["spec"]["group"] for task in state["tasks"].values()})
    if not smoke:
        conditions.insert(0, "FO04")
    selection = {"eligible": True, "experiment_id": "FO04", "role": "frozen stage-one configuration, not a course ranking",
                 "rule": recipe["selection_rule"], "recipe_fingerprint": managed.digest(recipe)}
    result = {"stage": 2, "smoke": smoke, "expected_seeds": state["contract"]["seeds"], "conditions": conditions,
              "reference_batch": baseline.name, "reference_run_count": 0 if smoke else len(recipe["reference_runs"]),
              "runs": records, "groups": groups, "curriculum": curricula, "unreadable": unreadable,
              "missing_conditions": sorted(set(conditions) - {r["experiment_id"] for r in records if r["kind"] == "train"}),
              "selection": selection}
    output = root / "summary"
    output.mkdir(exist_ok=True)
    for name, value in (("summary.json", result), ("selection.json", selection)):
        managed.write_json(output / name, value)
    for name, values in (("runs.csv", records), ("groups.csv", groups), ("curriculum.csv", curricula), ("tasks.csv", task_rows)):
        write_csv(output / name, values)
    counts = Counter(task["status"] for task in state["tasks"].values())
    managed.write_json(output / "status.json", {"batch": root.name, "updated_at": managed.now(),
                       "counts": dict(counts), "reference_run_count": result["reference_run_count"], "tasks": task_rows})
    lines = ["# 第二阶段实验概况", "", f"批次：`{root.name}`；新增任务已完成 **{counts['completed']}/{len(task_rows)}**。",
             f"参考：`{baseline.name}` 的 FO04；复用产物 **{result['reference_run_count']}** 条，不计入新增任务。",
             "", "配置已冻结；运行完成不代表课程到达 p=1 或任务达标。训练中断后从头重跑该任务，不是 checkpoint 续训。",
             "", "成绩见 [summary.md](summary.md)，课程进度见 [curriculum.csv](curriculum.csv)，冻结配置见 "
             f"[配置快照](../{recipe_path(root, smoke).name})。"]
    if smoke:
        lines += ["", "**短程烟测：小模型、64 步、2 回合；正式 FO04 不参与烟测成绩汇总。**"]
    lines += ["", "| 任务 | 状态 | attempt | 实际步数 | 说明 |", "|---|---|---:|---:|---|"]
    for row in task_rows:
        reason = str(row["reason"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['task']} | {row['status']} | {row['attempt'] or '-'} | {row['global_step'] or '-'} | {reason} |")
    (output / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines = ["# 第二阶段统一条件评估", "", "配置沿用冻结 FO04；不依据不同结束 p 的周期分数排序课程。",
             "均值 ± 样本标准差跨三个训练种子计算；100 个留出回合不是 100 个训练种子。", "",
             "| 条件 | p | 策略 | 均分 ± SD | ≥200 分比例 | 超时比例 | 状态/问题 |", "|---|---:|---|---:|---:|---:|---|"]
    for condition in conditions:
        for probability in state["contract"]["evaluation"]["probabilities"]:
            for policy in state["contract"]["evaluation"]["policies"]:
                group = next((g for g in groups if (g["kind"], g["experiment_id"], g["sensor_probability"], g["policy"])
                              == ("evaluate", condition, probability, policy)), None)
                if group and group["valid"]:
                    values = (f"{group['mean_return_mean']:.2f} ± {group['mean_return_std']:.2f}",
                              f"{group['score_200_rate_mean']:.1%}", f"{group['timeout_rate_mean']:.1%}", "有效")
                else:
                    values = ("pending", "pending", "pending", "; ".join(group["issues"]) if group else "尚未完成")
                lines.append(f"| {condition} | {probability:g} | {policy} | " + " | ".join(values) + " |")
    lines += ["", "## 课程进度", "", "阶段步数和转换明细保存在 curriculum.csv / summary.json；固定课程步数按配置与已完成采样步数推导。",
              "", "| 条件 | seed | 最终 p | 进入 p=1 | 最终任务达标 | 阶段步数 | 转换次数 |", "|---|---:|---:|---|---|---|---:|"]
    for row in curricula:
        lines.append(f"| {row['experiment_id']} | {row['seed']} | {row['final_probability']} | {row['entered_final']} | "
                     f"{row['final_task_solved']} | {row['stage_steps']} | {len(row['transitions'])} |")
    lines += ["", f"不可读产物：{len(unreadable)}；不合并无效组成绩。",
              "配置来源为第一阶段选型；selection.json 仅记录该冻结配置，不是第二阶段课程胜者。"]
    if smoke:
        lines += ["", "**烟测成绩不用于标准任务结论，正式 FO04 已排除。**"]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"第二阶段汇总：{output / 'overview.md'}", flush=True)
    return result


def run_stage2(root, baseline, cancellation, *, groups=None, smoke=False, device=None,
               dry_run=False, analyze_only=False, executor=managed.execute, environment=None):
    if root.resolve() == baseline.resolve():
        raise ValueError("第二阶段批次不能与第一阶段来源相同")
    with baseline_lock(baseline):
        recipe = inspect_baseline(baseline)
        plan = build_plan(recipe, groups, smoke=smoke, device=device)
        check_snapshot(root, recipe, plan)
        counts = Counter(task["kind"] for task in plan["tasks"])
        print(f"第二阶段：{root.name} | 冻结 FO04 | {counts['train']} 次新训练，{counts['evaluate']} 次新评估", flush=True)
        if dry_run:
            print("只读预览；不创建批次或冻结文件。参考产物：" + ("烟测不合并" if smoke else "3 次训练 + 18 次评估"))
            for task in plan["tasks"]:
                print(task["key"] + (f" | requested steps={task['parameters']['total_timesteps']}" if task["kind"] == "train"
                                     else f" | episodes={task['parameters']['episodes']}"))
            return 0
        if analyze_only:
            if not (root / "state.json").exists():
                raise ValueError("第二阶段批次尚未建立，无法分析")
            with coordinator_lock(root), managed.batch_lock(root):
                check_snapshot(root, recipe, plan)
                state = managed.load_state(root)
                managed.reconcile(root, state)
                managed.write_json(root / "state.json", state)
                report(root, state, recipe, baseline)
            return 0
        environment = managed.preflight(plan) if environment is None else environment
        if not smoke and environment != recipe["environment"]:
            raise ValueError("正式第二阶段环境与第一阶段参考不一致，拒绝混用成绩")
        with coordinator_lock(root):
            freeze(root, recipe, plan)

            def guarded_execute(command, log, stop):
                if coordinator_fingerprint() != recipe["coordinator_fingerprint"]:
                    raise RuntimeError("第二阶段协调代码运行期间发生变化，请使用新批次")
                check_snapshot(root, recipe, plan)
                return executor(command, log, stop)

            try:
                return managed.run_batch(root, plan, cancellation, executor=guarded_execute, environment=environment)
            finally:
                if (root / "state.json").is_file():
                    with managed.batch_lock(root):
                        report(root, managed.load_state(root), recipe, baseline)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=managed.identifier)
    parser.add_argument("--baseline-batch", type=managed.identifier, default="baseline-v1")
    parser.add_argument("--groups", nargs="+")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--heartbeat-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--heartbeat-timeout", type=float, default=30, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    root = PROJECT / "runs/experiments" / (args.batch or ("course-smoke-v1" if args.smoke else "course-10m-v1"))
    baseline = PROJECT / "runs/experiments" / args.baseline_batch
    try:
        if args.status:
            if not (root / "state.json").exists():
                print(f"第二阶段批次尚未建立：{root}")
                return 0
            state = managed.load_state(root)
            if "stage2" not in state["contract"]:
                raise ValueError("这不是第二阶段批次")
            print(f"批次：{root}；参考：{state['contract']['stage2']['baseline_batch']}")
            for key, task in state["tasks"].items():
                print(f"{task['status']:12s} {key} | {task['reason']}")
            print("以上为最后保存的状态；启动或 AnalyzeOnly 时复核产物。")
            return 0
        if not math.isfinite(args.heartbeat_timeout) or args.heartbeat_timeout <= 0:
            raise ValueError("heartbeat-timeout 必须为有限正数")
        groups = [g for value in (args.groups or []) for g in value.split(",") if g] or None
        with managed.Cancellation(args.heartbeat_file, args.heartbeat_timeout) as stop:
            return run_stage2(root, baseline, stop, groups=groups, smoke=args.smoke, device=args.device,
                              dry_run=args.dry_run, analyze_only=args.analyze_only)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"第二阶段启动失败：{type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1
