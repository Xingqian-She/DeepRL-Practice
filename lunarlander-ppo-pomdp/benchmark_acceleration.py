"""真实训练入口对照：默认两种模型、3 轮预热 + 10 轮计时、各重复 3 次。"""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import torch

from runtime import PROJECT_DIR, PRECISION, git_info, write_json


MODELS = {"d256": (256, 4, 4), "d512": (512, 8, 8)}


def benchmark(output, models=("d256", "d512"), warmup_updates=3, timed_updates=10, repeats=3):
    if min(warmup_updates, timed_updates, repeats) < 1:
        raise ValueError("warmup_updates、timed_updates、repeats 必须为正数")
    if not torch.cuda.is_available():
        raise RuntimeError("性能对照需要 CUDA 容器")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifact_root = PROJECT_DIR / "runs" / ("acceleration-benchmark-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    artifact_root.mkdir(parents=True)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(), "torch": str(torch.__version__),
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(), "precision": PRECISION,
        "git": git_info(), "warmup_updates": warmup_updates, "timed_updates": timed_updates,
        "repeats": repeats, "artifact_root": str(artifact_root),
        "cache_policy": "isolated empty Inductor/Triton caches per model; first auto trial cold, later auto trials reuse",
        "timing": "steady throughput: collect + PPO, CUDA synchronized; process wall includes imports, compile, logging, checkpoint and exit",
        "trials": [], "summary": {},
    }
    for name in models:
        hidden, layers, heads = MODELS[name]
        # 仅隔离基准的编译缓存；不清理用户已有缓存或实验目录。
        with tempfile.TemporaryDirectory(prefix="lunarlander-compile-benchmark-") as cache_root:
            for repeat in range(repeats):
                # 交替先后顺序，减小温度/运行顺序的偏差。
                for mode in (("off", "auto") if repeat % 2 == 0 else ("auto", "off")):
                    label = f"{name}-{mode}-{repeat + 1}"
                    parent = artifact_root / label
                    log_path = artifact_root / (label + ".log")
                    command = [sys.executable, "-B", str(PROJECT_DIR / "train.py"), "--device", "cuda",
                               "--compile", mode, "--hidden-dim", str(hidden), "--num-layers", str(layers),
                               "--num-heads", str(heads), "--total-timesteps", str((warmup_updates + timed_updates) * 2048),
                               "--output-dir", str(parent)]
                    environment = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": str(Path(cache_root) / "inductor"),
                                   "TRITON_CACHE_DIR": str(Path(cache_root) / "triton")}
                    print(f"Benchmark {label}: {warmup_updates}+{timed_updates} updates", flush=True)
                    started = time.perf_counter()
                    with log_path.open("w", encoding="utf-8") as log:
                        subprocess.run(command, cwd=PROJECT_DIR, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, check=True)
                    wall_seconds = time.perf_counter() - started
                    metadata_files = list(parent.glob("*/metadata.json"))
                    if len(metadata_files) != 1:
                        raise RuntimeError(f"expected one run: {parent}")
                    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
                    if metadata["status"] != "completed" or metadata["acceleration"]["enabled"] != (mode == "auto"):
                        raise RuntimeError(f"run failed or unexpected fallback: {metadata_files[0]}")
                    with (metadata_files[0].parent / "training.csv").open(encoding="utf-8", newline="") as file:
                        rows = list(csv.DictReader(file))
                    timed = rows[warmup_updates:]
                    if len(timed) != timed_updates:
                        raise RuntimeError("incomplete timed updates")
                    seconds = [float(row["collect_seconds"]) + float(row["ppo_seconds"]) for row in timed]
                    trial = {
                        "model": name, "agent": metadata["parameters"]["agent"], "mode": mode, "repeat": repeat + 1,
                        "run": str(metadata_files[0].parent), "process_wall_seconds": wall_seconds,
                        "training_elapsed_seconds": metadata["elapsed_seconds"],
                        "compile_warmup_seconds": metadata["acceleration"]["warmup_seconds"],
                        "peak_allocated_mib": metadata["peak_allocated_mib"],
                        "steady_update_seconds": seconds,
                        "steady_steps_per_second": timed_updates * 2048 / sum(seconds),
                        "max_consistency_error": max(float(row["consistency_max_logprob_error"]) for row in rows),
                    }
                    result["trials"].append(trial)
                    write_json(output, result)
                    print(f"  {trial['steady_steps_per_second']:.1f} steps/s | wall {wall_seconds:.1f}s | "
                          f"compile {trial['compile_warmup_seconds']:.1f}s | peak {trial['peak_allocated_mib']:.1f} MiB", flush=True)
        summary = {}
        for mode in ("off", "auto"):
            trials = [row for row in result["trials"] if row["model"] == name and row["mode"] == mode]
            summary[mode] = {
                "median_steps_per_second": statistics.median(row["steady_steps_per_second"] for row in trials),
                "median_process_wall_seconds": statistics.median(row["process_wall_seconds"] for row in trials),
                "max_peak_allocated_mib": max(row["peak_allocated_mib"] for row in trials),
            }
        summary["steady_speedup"] = summary["auto"]["median_steps_per_second"] / summary["off"]["median_steps_per_second"]
        summary["cold_compile_warmup_seconds"] = next(row["compile_warmup_seconds"] for row in result["trials"]
                                                      if row["model"] == name and row["mode"] == "auto")
        result["summary"][name] = summary
        write_json(output, result)
        print(f"{name}: steady speedup {summary['steady_speedup']:.2f}x", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/acceleration-benchmark.json"))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--timed-updates", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    benchmark(**vars(parser.parse_args()))
