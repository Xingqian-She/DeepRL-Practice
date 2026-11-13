"""Managed sequential experiments. On Windows use run_experiments.ps1; --help lists container options."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import signal
import statistics
import subprocess
import sys
import threading
import time


PROJECT = Path(__file__).resolve().parent
REPOSITORY = PROJECT.parent
SHARED = REPOSITORY / "shared" / "src"
sys.path.insert(0, str(SHARED))
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")
ARTIFACT_ERRORS = (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError, IndexError, pickle.UnpicklingError)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON: {value}")))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def dependency_lock_hash(path):
    """Preserve exact lock content while tolerating Git's Windows CRLF checkout."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge(base, changes):
    result = deepcopy(base)
    for key, value in changes.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"无效名称 {value!r}：请使用英文字母、数字、下划线和连字符，长度不超过 80")
    return value


def build_plan(config_path, groups=None, *, smoke=False, device=None):
    from config import AgentConfig, TrainConfig
    from sensors import SensorConfig

    settings = read_json(config_path)
    if settings.get("schema_version") != 1 or set(settings) - {
            "schema_version", "default_groups", "seeds", "training", "evaluation", "groups"}:
        raise ValueError("不支持的 experiments.json 版本或字段")
    seeds = settings["seeds"]
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds 必须是非空、不重复的 [0, 2**32) 整数列表")
    selected = groups or settings["default_groups"]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("实验组必须非空且不能重复")
    for group in selected:
        identifier(group)
        if group not in settings["groups"]:
            raise ValueError(f"未知实验组 {group}；可选：{', '.join(settings['groups'])}")
    evaluation = deepcopy(settings["evaluation"])
    if set(evaluation) != {"episodes", "seed", "num_envs", "probabilities", "policies"}:
        raise ValueError("evaluation 字段必须包含 episodes/seed/num_envs/probabilities/policies")
    for name in ("episodes", "num_envs"):
        if type(evaluation[name]) is not int or evaluation[name] < 1:
            raise ValueError(f"evaluation.{name} 必须是正整数")
    if type(evaluation["seed"]) is not int or not 0 <= evaluation["seed"] < 2**32:
        raise ValueError("evaluation.seed 超出范围")
    probabilities, policies = evaluation["probabilities"], evaluation["policies"]
    if (not probabilities or len(set(probabilities)) != len(probabilities)
            or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities)):
        raise ValueError("probabilities 必须是非空、不重复的 [0, 1] 数值列表")
    if not policies or len(set(policies)) != len(policies) or any(p not in ("sample", "argmax") for p in policies):
        raise ValueError("policies 必须是非空、不重复的 sample/argmax 列表")

    overrides = {}
    if smoke:
        overrides = {"agent": {"hidden_dim": 32, "num_layers": 2, "num_heads": 2, "context_length": 4},
                     "total_timesteps": 64, "num_envs": 2, "num_steps": 16, "sequence_length": 6,
                     "sequences_per_minibatch": 3, "update_epochs": 2, "max_episode_steps": 16,
                     "save_interval": 1, "diagnostic_interval": 1, "eval_interval": 1,
                     "eval_episodes": 2, "eval_num_envs": 2, "device": "cpu", "compile_mode": "off"}
        evaluation.update(episodes=2, num_envs=2)
    if device:
        overrides["device"] = device

    def normalized(parameters, seed):
        parameters = merge(parameters, overrides)
        if smoke and parameters.get("agent", {}).get("backbone_mode") == "encoder_decoder":
            parameters["agent"]["decoder_layers"] = 2
            parameters["max_episode_steps"] = 19
        if "seed" in parameters:
            raise ValueError("请在顶层 seeds 配置种子，不要在 training 内配置 seed")
        agent, sensors = parameters.pop("agent", {}), parameters.pop("sensors", {})
        config = TrainConfig(agent=AgentConfig(**agent), sensors=SensorConfig(**sensors), seed=seed, **parameters)
        if config.eval_interval == 0:
            raise ValueError("统一入口需要周期评估以生成基本统计，eval_interval 必须大于 0")
        return asdict(config)

    defaults = normalized(deepcopy(settings["training"]), seeds[0])
    contract = {"seeds": seeds, "training_defaults": defaults, "evaluation": evaluation, "smoke": smoke}
    training, heldout = [], []
    for group in selected:
        entry = settings["groups"][group]
        if set(entry) - {"description", "training"}:
            raise ValueError(f"实验组 {group} 有未知字段")
        parameters = merge(settings["training"], entry.get("training", {}))
        for seed in seeds:
            config = normalized(deepcopy(parameters), seed)
            key = f"train--{group}--seed{seed}"
            training.append({"key": key, "kind": "train", "group": group, "seed": seed, "parameters": config})
            for probability in probabilities:
                for policy in policies:
                    heldout.append({"key": f"eval--{group}--seed{seed}--p{probability:g}--{policy}",
                                    "kind": "evaluate", "group": group, "seed": seed, "depends_on": key,
                                    "training_parameters": config,
                                    "parameters": {"episodes": evaluation["episodes"], "seed": evaluation["seed"],
                                                   "num_envs": evaluation["num_envs"], "policy": policy,
                                                   "sensor_failure_probability": probability, "device": config["device"]}})
    return {"contract": contract, "groups": list(selected), "tasks": training + heldout}


def preflight(plan):
    import gymnasium as gym
    import torch
    from runtime import EVALUATION_PROTOCOL, PRECISION, configure_runtime, dependency_version

    if sys.platform != "linux":
        raise RuntimeError("训练调度器请在 Linux 容器运行；Windows 请执行 run_experiments.ps1")
    if (sys.version_info.major, sys.version_info.minor) != (3, 11):
        raise RuntimeError("项目锁定 Python 3.11，请更新项目镜像")
    installed_lock = Path("/opt/deeprl/requirements.lock")
    if not installed_lock.is_file() or dependency_lock_hash(installed_lock) != dependency_lock_hash(REPOSITORY / "requirements.lock"):
        raise RuntimeError("镜像依赖锁与仓库不一致，请通过 PowerShell 入口重建镜像")
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    devices = {str(configure_runtime(task["seed"], task["parameters"]["device"]))
               for task in plan["tasks"] if task["kind"] == "train"}
    if len(devices) != 1:
        raise ValueError("同一批次须统一运行设备，请为不同设备建立独立批次")
    device = devices.pop()
    env = gym.make("LunarLander-v3")
    try:
        env.reset(seed=42)
        env.step(0)
    finally:
        env.close()
    result = {"precision": PRECISION, "device": device, "python": platform.python_version(),
              "dependencies": {name: dependency_version(name) for name in ("torch", "gymnasium", "numpy", "box2d-py")},
              "hardware": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                           "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu"},
              "requirements_sha256": dependency_lock_hash(installed_lock), "evaluation_protocol": EVALUATION_PROTOCOL}
    print(f"环境就绪：Python {result['python']} | {device} | {result['hardware']['device_name']}", flush=True)
    return result


@contextmanager
def batch_lock(root):
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("该批次已有调度器运行，请先等待或停止原任务") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load_state(root):
    state = read_json(root / "state.json")
    if state.get("schema_version") != 1 or not isinstance(state.get("tasks"), dict):
        raise ValueError(f"不支持或损坏的批次状态：{root / 'state.json'}")
    return state


def register_plan(state, plan, source, environment):
    if (state["contract"] != plan["contract"] or state["source_fingerprint"] != source
            or state["environment"] != environment):
        raise ValueError("此批次的公共参数、源码或运行环境已改变；请使用 -Batch 新名称，原结果已保留")
    for spec in plan["tasks"]:
        old = state["tasks"].get(spec["key"])
        if old and old["spec"] != spec:
            raise ValueError(f"既有任务 {spec['key']} 的参数已改变，请使用 -Batch 新名称")
    for spec in plan["tasks"]:
        state["tasks"].setdefault(spec["key"], {"spec": spec, "status": "pending", "reason": "尚未运行",
                                                "current_attempt": None, "attempts": []})


def attempt_path(root, attempt):
    target = (root / attempt["path"]).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError("任务路径超出当前批次目录")
    return target


def current_attempt(task):
    return next((attempt for attempt in task["attempts"] if attempt["number"] == task["current_attempt"]), None)


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def dependency_reference(state, spec):
    task = state["tasks"][spec["depends_on"]]
    if task["status"] != "completed":
        raise ValueError("依赖的训练尚未完整完成")
    attempt = current_attempt(task)
    return {"task": spec["depends_on"], "attempt": attempt["number"], "run_id": attempt["run_id"],
            "checkpoint_sha256": attempt["artifact_hashes"]["checkpoints/latest.pt"]}


def validate_attempt(root, state, task):
    """Validate actual artifacts, never a child exit code or the cached task status alone."""
    import torch
    from runtime import load_checkpoint
    from summarize_experiments import read_run

    attempt, spec = current_attempt(task), task["spec"]
    if attempt is None:
        raise ValueError("尚未运行")
    run = attempt_path(root, attempt) / "run"
    for name, expected in attempt.get("artifact_hashes", {}).items():
        path = (run / name).resolve()
        if not path.is_relative_to(run.resolve()) or file_hash(path) != expected:
            raise ValueError(f"产物校验失败：{name}")
    metadata = read_json(run / "metadata.json")
    if metadata["status"] != "completed":
        raise ValueError(f"运行状态为 {metadata['status']}")
    if (metadata["kind"] != spec["kind"] or metadata["experiment_id"] != spec["group"]
            or metadata["source_fingerprint"] != state["source_fingerprint"]
            or metadata["config_fingerprint"] != digest(metadata["parameters"])):
        raise ValueError("运行标识、源码或配置指纹不匹配")
    for field in ("precision", "device", "dependencies", "hardware", "evaluation_protocol"):
        if metadata[field] != state["environment"][field]:
            raise ValueError(f"运行环境不匹配：{field}")
    record = read_run(run / "metadata.json")
    if record["issues"]:
        raise ValueError("; ".join(record["issues"]))
    if spec["kind"] == "train":
        parameters = spec["parameters"]
        size = parameters["num_envs"] * parameters["num_steps"]
        updates = parameters["total_timesteps"] // size
        steps = updates * size
        if (metadata["parameters"] != parameters or metadata.get("global_step") != steps
                or metadata.get("effective_total_timesteps") != steps or metadata.get("completed_updates") != updates
                or not metadata.get("training_budget_completed")):
            raise ValueError("训练配置或完成预算不匹配")
        training = csv_rows(run / "training.csv")
        evaluations = csv_rows(run / "evaluation.csv")
        if (len(training) != updates or [int(row["update"]) for row in training] != list(range(1, updates + 1))
                or any(int(row["global_step"]) != int(row["update"]) * size for row in training)
                or not evaluations or int(evaluations[-1]["global_step"]) != steps):
            raise ValueError("训练或周期评估记录不完整")
        # Once validated, unchanged content hashes also certify the previously loaded weights.
        if not attempt.get("artifact_hashes"):
            agent, saved = load_checkpoint(run / "checkpoints/latest.pt", torch.device("cpu"))
            if (saved["train_config"] != parameters or saved["global_step"] != steps
                    or saved.get("provenance") != {key: metadata[key] for key in (
                        "run_id", "experiment_id", "config_fingerprint", "source_fingerprint", "evaluation_protocol")}
                    or not all(torch.isfinite(value).all().item() for value in agent.state_dict().values())):
                raise ValueError("checkpoint 的权重、预算或训练来源不匹配")
        required = ["metadata.json", "training.csv", "evaluation.csv", "checkpoints/latest.pt"]
    else:
        reference = dependency_reference(state, spec)
        if attempt.get("dependency") != reference:
            raise ValueError("评估依赖的训练 attempt 或 checkpoint 已改变")
        parameters = metadata["parameters"]
        training_task = state["tasks"][spec["depends_on"]]
        train_run = attempt_path(root, current_attempt(training_task)) / "run"
        provenance = read_json(train_run / "metadata.json")
        provenance = {key: provenance[key] for key in (
            "run_id", "experiment_id", "config_fingerprint", "source_fingerprint", "evaluation_protocol")}
        expected = {key: value for key, value in spec["parameters"].items() if key != "device"}
        if (any(parameters.get(key) != value for key, value in expected.items())
                or parameters.get("training_seed") != spec["seed"]
                or parameters.get("training_parameters") != spec["training_parameters"]
                or Path(parameters["checkpoint"]).resolve() != (train_run / "checkpoints/latest.pt").resolve()
                or parameters.get("mode") != "none"
                or parameters.get("effective_use_previous_action") != spec["training_parameters"]["agent"]["use_previous_action"]
                or metadata.get("training_provenance") != provenance):
            raise ValueError("评估配置或训练来源不匹配")
        result = read_json(run / "evaluation.json")
        episodes = result["episodes"]
        count = spec["parameters"]["episodes"]
        budget = spec["training_parameters"]
        steps = budget["total_timesteps"] // (budget["num_envs"] * budget["num_steps"]) * budget["num_envs"] * budget["num_steps"]
        if (len(episodes) != count or metadata.get("completed_episodes") != count
                or len(csv_rows(run / "episodes.csv")) != count
                or metadata.get("checkpoint_global_step") != steps or result.get("checkpoint_global_step") != steps
                or result.get("parameters") != parameters or result.get("training_provenance") != provenance
                or [row["seed"] for row in episodes] != [(expected["seed"] + i) % 2**32 for i in range(count)]
                or any(not math.isfinite(row["return"]) or not 0 < row["length"] <= budget["max_episode_steps"] for row in episodes)):
            raise ValueError("评估回合数、预算或逐回合数据不完整")
        if (not math.isclose(result["mean_return"], statistics.mean(row["return"] for row in episodes), abs_tol=1e-7)
                or not math.isclose(result["score_200_rate"], sum(row["return"] >= 200 for row in episodes) / count)
                or not 0 <= result["timeout_rate"] <= 1):
            raise ValueError("评估统计与逐回合结果不一致")
        required = ["metadata.json", "evaluation.json", "episodes.csv"]
    optional = ["episodes.csv", "evaluation_episodes.csv", "diagnostics.csv", "curriculum_transitions.csv"]
    hashes = {name: file_hash(run / name) for name in dict.fromkeys(required + optional) if (run / name).is_file()}
    return metadata, hashes


def reconcile(root, state):
    # Training first: invalidated weights must invalidate their dependent held-out results.
    ordered = sorted(state["tasks"].values(), key=lambda task: task["spec"]["kind"] != "train")
    for task in ordered:
        attempt = current_attempt(task)
        if attempt is None:
            task.update(status="pending", reason="尚未运行")
            continue
        try:
            metadata, hashes = validate_attempt(root, state, task)
            attempt.update(artifact_hashes=hashes, run_id=metadata["run_id"])
            task.update(status="completed", reason="产物和来源校验通过")
        except ARTIFACT_ERRORS as error:
            previous = task["status"]
            stopped = attempt.get("interruption_reason")
            task.update(status="interrupted" if previous in ("running", "interrupted") else "failed",
                        reason=f"{stopped}; {error}" if stopped else str(error))


class Cancellation:
    def __init__(self, heartbeat=None, timeout=30):
        self.event = threading.Event()
        self.reason = ""
        self.heartbeat, self.timeout = heartbeat, timeout
        self.closed = threading.Event()

    def request(self, reason):
        if not self.event.is_set():
            self.reason = reason
            self.event.set()

    def __enter__(self):
        self.handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
        for number in self.handlers:
            signal.signal(number, lambda signum, frame: self.request(f"收到停止信号 {signum}"))
        if self.heartbeat:
            self.thread = threading.Thread(target=self._watch, daemon=True)
            self.thread.start()
        return self

    def _watch(self):
        last, changed = None, time.monotonic()
        while not self.closed.wait(min(1, self.timeout / 4)):
            try:
                value = Path(self.heartbeat).read_bytes()
                if value and value != last:
                    last, changed = value, time.monotonic()
            except OSError:
                pass
            if time.monotonic() - changed > self.timeout:
                self.request("启动窗口心跳失联，停止前台任务")
                return

    def __exit__(self, *args):
        self.closed.set()
        if hasattr(self, "thread"):
            self.thread.join(timeout=2)
        for number, handler in self.handlers.items():
            signal.signal(number, handler)


def execute(command, log_path, cancellation):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SHARED) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "2")
    env["PYTHONUNBUFFERED"] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                 start_new_session=True)

        def copy_output():
            for line in child.stdout:
                log.write(line)
                log.flush()
                try:
                    print(line, end="", flush=True)
                except (BrokenPipeError, OSError):
                    pass

        output = threading.Thread(target=copy_output, daemon=True)
        output.start()
        try:
            while child.poll() is None:
                if cancellation.event.wait(.1):
                    for signum, wait in ((signal.SIGINT, 10), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
                        if child.poll() is not None:
                            break
                        try:
                            os.killpg(child.pid, signum)
                            child.wait(timeout=wait)
                        except ProcessLookupError:
                            break
                        except subprocess.TimeoutExpired:
                            continue
                    break
            return child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            output.join()
            child.stdout.close()


def command_for(root, state, spec, run):
    from summarize_experiments import training_arguments

    if spec["kind"] == "train":
        config = spec["parameters"]
        sensors = config["sensors"]
        arguments = training_arguments(config) + ["--seed", str(spec["seed"]), "--total-timesteps", str(config["total_timesteps"]),
            "--sensor-mode", sensors["mode"], "--curriculum-schedule", sensors["schedule"],
            "--curriculum-start-fraction", str(sensors["start_fraction"]),
            "--curriculum-end-fraction", str(sensors["end_fraction"])]
        script = "train.py"
    else:
        dependency = state["tasks"][spec["depends_on"]]
        checkpoint = attempt_path(root, current_attempt(dependency)) / "run/checkpoints/latest.pt"
        arguments = ["--checkpoint", str(checkpoint)]
        for key, value in spec["parameters"].items():
            arguments += ["--" + key.replace("_", "-"), str(value)]
        script = "evaluate.py"
    return [sys.executable, "-B", str(PROJECT / script), *arguments, "--experiment-id", spec["group"], "--run-dir", str(run)]


def report(root, state):
    from curriculum import curriculum_record
    from summarize_experiments import summarize, write_csv

    directories, issues, rows, curricula = [], {}, [], []
    for key, task in state["tasks"].items():
        attempt = current_attempt(task)
        metadata, run = {}, None
        if attempt:
            run = attempt_path(root, attempt) / "run"
            directories.append(run)
            if task["status"] != "completed":
                issues[str(run)] = ["managed task: " + task["reason"]]
            try:
                metadata = read_json(run / "metadata.json")
            except (OSError, ValueError):
                pass
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("kind") == "train":
            curricula.append(curriculum_record(run, metadata))
        elapsed = metadata.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed):
            elapsed = None
        rows.append({"task": key, "kind": task["spec"]["kind"], "group": task["spec"]["group"],
                     "training_seed": task["spec"]["seed"], "status": task["status"],
                     "attempt": task["current_attempt"], "global_step": metadata.get("global_step"),
                     "elapsed_seconds": elapsed, "reason": task["reason"],
                     "run_dir": str(run) if run else None})
    conditions = sorted({task["spec"]["group"] for task in state["tasks"].values()})
    result = summarize([], root / "summary", state["contract"]["seeds"], conditions,
                       run_dirs=directories, run_issues=issues)
    counts = Counter(task["status"] for task in state["tasks"].values())
    output = root / "summary"
    write_csv(output / "tasks.csv", rows)
    write_csv(output / "curriculum.csv", curricula)
    write_json(output / "status.json", {"batch": state["batch"], "updated_at": now(), "counts": dict(counts), "tasks": rows})
    lines = ["# 实验运行概况", "", f"批次：`{state['batch']}`；已完成 {counts['completed']}/{len(rows)} 个任务。",
             "", "运行完成表示预算和产物完整，不表示已学会着陆。跨种子标准差采用样本标准差；留出回合不是独立训练种子。",
             "", "本入口不自动冻结课程配置；selection.json 只描述当前已登记实验组，不代表已完成手册的七组对照。"]
    if state["contract"]["smoke"]:
        lines += ["", "**这是短程烟测，成绩不用于判断标准任务是否达标。**"]
    if state["contract"]["training_defaults"]["agent"].get("xy_fourier"):
        lines += ["", "**固定 XY Fourier 实验：x 四档、y 两档，输入 32 维。相位统计只记录，不调整频率。**"]
        if not state["contract"]["smoke"]:
            from xy_fourier_comparison import write_comparison
            write_comparison(root, state)
            lines += ["", "历史 PO01/PO03 参考与源码差异见 [xy-comparison.md](xy-comparison.md)。"]
    if state["contract"]["training_defaults"]["agent"]["backbone_mode"] == "encoder_decoder":
        lines += ["", "**Encoder–Decoder 探索实验：以最终统一条件评估为主要依据；"
                  "容量、感受野和共享梯度均有变化，结果不能单独归因于交叉注意力。**",
                  "", "PO03 的最终难度预算、进入时学习率和课程轨迹见 [curriculum.csv](curriculum.csv)。",
                  "历史 PO01/PO03 仅作独立参考，保留原来源：[第二阶段报告](../../../../reports/PO01-PO03-stage2-analysis.md)。"
                  "评估 seed=20000 已用于历史分析，本轮为探索性比较。"]
    lines += ["", "成绩及达标比例见 [summary.md](summary.md)，逐种子数据见 [runs.csv](runs.csv)，跨种子统计见 [groups.csv](groups.csv)。",
              "", "| 任务 | 状态 | attempt | 实际步数 | 耗时（秒） | 说明 |", "|---|---|---:|---:|---:|---|"]
    for row in rows:
        elapsed = row["elapsed_seconds"]
        reason = str(row["reason"]).replace("|", "/").replace("\n", " ")
        lines.append(f"| {row['task']} | {row['status']} | {row['attempt'] or '-'} | {row['global_step'] or '-'} | "
                     f"{f'{elapsed:.2f}' if elapsed is not None else '-'} | {reason} |")
    (output / "overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"统计已保存：{output / 'overview.md'}", flush=True)
    return result


def run_batch(root, plan, cancellation, *, executor=execute, environment=None, source=None):
    from runtime import source_fingerprint

    check_source = source is None
    with batch_lock(root):
        environment = preflight(plan) if environment is None else environment
        source = source_fingerprint() if source is None else source
        if (root / "state.json").exists():
            state = load_state(root)
        else:
            state = {"schema_version": 1, "batch": root.name, "created_at": now(), "contract": plan["contract"],
                     "source_fingerprint": source, "environment": environment, "tasks": {}}
        register_plan(state, plan, source, environment)
        reconcile(root, state)
        write_json(root / "state.json", state)
        exit_code = 0
        try:
            for index, spec in enumerate(plan["tasks"], 1):
                if cancellation.event.is_set():
                    exit_code = 130
                    break
                if check_source and source != source_fingerprint():
                    raise RuntimeError("执行期间源码发生改变；已停止后续任务，请使用新批次")
                task = state["tasks"][spec["key"]]
                if task["status"] == "completed":
                    print(f"[{index}/{len(plan['tasks'])}] 跳过已完成：{spec['key']}", flush=True)
                    continue
                # A hard stop can leave a directory behind before state.json records its allocation.
                on_disk = [int(match[1]) for path in (root / "tasks" / spec["key"]).glob("attempt-*")
                           if (match := re.fullmatch(r"attempt-(\d+)", path.name))]
                number = max([attempt["number"] for attempt in task["attempts"]] + on_disk, default=0) + 1
                relative = f"tasks/{spec['key']}/attempt-{number:04d}"
                attempt = {"number": number, "path": relative, "started_at": now()}
                if spec["kind"] == "evaluate":
                    attempt["dependency"] = dependency_reference(state, spec)
                folder = attempt_path(root, attempt)
                folder.mkdir(parents=True, exist_ok=False)
                task["attempts"].append(attempt)
                task.update(current_attempt=number, status="running", reason="正在运行")
                write_json(root / "state.json", state)
                print(f"[{index}/{len(plan['tasks'])}] 开始：{spec['key']}，attempt {number}", flush=True)
                code = executor(command_for(root, state, spec, folder / "run"), folder / "console.log", cancellation)
                attempt.update(finished_at=now(), exit_code=code)
                if cancellation.event.is_set():
                    attempt["interruption_reason"] = cancellation.reason
                    task.update(status="interrupted", reason=cancellation.reason)
                    exit_code = 130
                    break
                try:
                    metadata, hashes = validate_attempt(root, state, task)
                    # Valid artifacts can be recovered next time even if a client exited abnormally.
                    if code != 0:
                        raise RuntimeError(f"子任务退出码 {code}；重新启动时将复核产物")
                    attempt.update(artifact_hashes=hashes, run_id=metadata["run_id"])
                    task.update(status="completed", reason="产物和来源校验通过")
                except ARTIFACT_ERRORS as error:
                    task.update(status="failed", reason=str(error))
                    print(f"任务失败，队列已停止：{task['reason']}", file=sys.stderr, flush=True)
                    exit_code = 1
                    break
                write_json(root / "state.json", state)
        except Exception as error:
            print(f"队列已停止：{type(error).__name__}: {error}", file=sys.stderr, flush=True)
            state["error"] = str(error)
            exit_code = 1
        finally:
            state.update(updated_at=now(), last_exit_code=exit_code)
            write_json(root / "state.json", state)
            # Reconcile dependent results after any new training attempt, including a failed one.
            reconcile(root, state)
            write_json(root / "state.json", state)
            report(root, state)
        if exit_code == 130:
            print(f"已停止：{cancellation.reason}。再次执行相同命令即可检查并补跑。", flush=True)
        return exit_code


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "experiments.json")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--batch", type=identifier)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--status", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="独立小模型/64 步烟测，默认 CPU，不用于收敛结论")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--heartbeat-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--heartbeat-timeout", type=float, default=30, help=argparse.SUPPRESS)
    return parser


def main(argv=None):
    args = make_parser().parse_args(argv)
    root = PROJECT / "runs" / "experiments" / (args.batch or ("smoke-v1" if args.smoke else "baseline-v1"))
    try:
        if args.status or args.analyze_only:
            if not (root / "state.json").exists():
                print(f"批次尚未建立：{root}")
                return 0 if args.status else 1
            state = load_state(root)
            if args.status:
                print(f"批次：{root}")
                for key, task in state["tasks"].items():
                    print(f"{task['status']:12s} {key} | {task['reason']}")
                print("以上为最后保存的状态；启动或 AnalyzeOnly 时会重新核对产物。")
            else:
                with batch_lock(root):
                    state = load_state(root)
                    reconcile(root, state)
                    write_json(root / "state.json", state)
                    report(root, state)
            return 0
        groups = [group for value in (args.groups or []) for group in value.split(",") if group] or None
        plan = build_plan(args.config, groups, smoke=args.smoke, device=args.device)
        counts = Counter(task["kind"] for task in plan["tasks"])
        print(f"批次：{root}\n实验组：{', '.join(plan['groups'])} | {counts['train']} 次训练，{counts['evaluate']} 次留出评估", flush=True)
        if args.dry_run:
            for spec in plan["tasks"]:
                budget = spec["parameters"].get("total_timesteps")
                print(f"{spec['key']}" + (f" | requested steps={budget}" if budget else f" | episodes={spec['parameters']['episodes']}"))
            return 0
        if args.heartbeat_timeout <= 0 or not math.isfinite(args.heartbeat_timeout):
            raise ValueError("heartbeat-timeout 必须是有限正数")
        with Cancellation(args.heartbeat_file, args.heartbeat_timeout) as cancellation:
            return run_batch(root, plan, cancellation)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
