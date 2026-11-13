"""运行设施：精度设置、独立目录、元数据、CSV/TensorBoard 和 checkpoint。"""

import csv
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import uuid

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import AgentConfig, ENV_ID
from model import Agent
from sensors import input_spec
from acceleration import eager_status


PROJECT_DIR = Path(__file__).resolve().parent
PRECISION = "float32; math SDPA; TF32/AMP disabled"
EVALUATION_PROTOCOL = "batched-policy-v2"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def source_fingerprint():
    files = list(PROJECT_DIR.glob("*.py")) + list((PROJECT_DIR.parent / "shared" / "src").rglob("*.py"))
    return fingerprint({path.relative_to(PROJECT_DIR.parent).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(files) if "__pycache__" not in path.parts})


def dependency_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def configure_runtime(seed, device="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("明确请求了 CUDA，但容器中 CUDA 不可用")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device(device)


def git_info():
    root = PROJECT_DIR.parent
    try:
        def git(*args):
            return subprocess.check_output(
                ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root), *args],
                text=True, stderr=subprocess.DEVNULL, timeout=10,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
            ).strip()
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Run:
    def __init__(self, kind, parameters, device, output_dir=None, *, run_dir=None, experiment_id=None):
        if output_dir is not None and run_dir is not None:
            raise ValueError("run-dir 与 output-dir 不能同时使用")
        created = datetime.now(timezone.utc)
        identifier = f"{created:%Y%m%dT%H%M%S%fZ}-{kind}-seed{parameters['seed']}-{uuid.uuid4().hex[:8]}"
        self.path = (Path(run_dir).resolve() if run_dir is not None
                     else Path(output_dir or PROJECT_DIR / "runs").resolve() / identifier)
        self.path.mkdir(parents=True, exist_ok=False)
        self._csv_fields = {}
        self.metadata = {
            "schema_version": 3, "experiment_id": experiment_id,
            "config_fingerprint": fingerprint(parameters), "source_fingerprint": source_fingerprint(),
            "evaluation_protocol": EVALUATION_PROTOCOL,
            "run_id": identifier, "created_at": created.isoformat(), "kind": kind,
            "environment_id": ENV_ID, "parameters": parameters, "precision": PRECISION,
            "device": str(device), "python": platform.python_version(), "git": git_info(),
            "dependencies": {name: dependency_version(name) for name in ("torch", "gymnasium", "numpy", "box2d-py")},
            "hardware": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                         "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"},
            "status": "running", "global_step": 0,
            "acceleration": eager_status(parameters.get("compile_mode", "off"),
                                         "initializing" if kind == "train" else "evaluation uses eager"),
        }
        self.writer = SummaryWriter(str(self.path / "tensorboard")) if kind == "train" else None
        self.save()
        print(f"Run directory: {self.path}", flush=True)

    def save(self):
        write_json(self.path / "metadata.json", self.metadata)

    def append_csv(self, filename, row):
        target = self.path / filename
        new = not target.exists()
        if filename not in self._csv_fields:
            if new:
                self._csv_fields[filename] = list(row)
            else:
                with target.open(newline="", encoding="utf-8") as file:
                    self._csv_fields[filename] = next(csv.reader(file))
        fields = self._csv_fields[filename]
        if set(fields) != set(row):
            raise ValueError(f"CSV 字段变化：{filename}")
        with target.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerow(row)

    def close(self):
        try:
            if self.writer is not None:
                self.writer.close()
        finally:
            self.save()


@contextmanager
def preserve_rng_state(device):
    """Evaluation and diagnostics must not advance training RNG streams."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if device.type == "cuda" or torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def save_checkpoint(path, agent, config, global_step, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save({
        "format_version": 2, "environment_id": ENV_ID, "agent_config": asdict(agent.config),
        "train_config": asdict(config), "state_dict": agent.state_dict(),
        "global_step": global_step, "precision": PRECISION,
        "input_spec": input_spec(xy_fourier=agent.config.xy_fourier),
        "acceleration": getattr(agent, "acceleration", eager_status()),
        "training_state": training_state,
        "provenance": getattr(agent, "provenance", None),
    }, temporary)
    temporary.replace(path)


def load_checkpoint(path, device, use_previous_action=None):
    saved = torch.load(Path(path), map_location=device, weights_only=True)
    if saved.get("format_version") == 1:
        raise ValueError("旧版 6 维输入 checkpoint 与当前 20 维输入不兼容，请重新训练")
    if saved.get("format_version") != 2 or saved.get("environment_id") != ENV_ID:
        raise ValueError("不是本项目支持的 LunarLander checkpoint")
    config = dict(saved["agent_config"])
    expected_config = AgentConfig(**config)
    if saved.get("input_spec") != input_spec(xy_fourier=expected_config.xy_fourier):
        raise ValueError("checkpoint 的传感器输入布局与当前项目不一致")
    if use_previous_action is not None:
        config["use_previous_action"] = use_previous_action
    agent = Agent(AgentConfig(**config)).to(device=device, dtype=torch.float32)
    agent.load_state_dict(saved["state_dict"], strict=True)
    agent.provenance = saved.get("provenance")
    agent.eval()
    return agent, saved
