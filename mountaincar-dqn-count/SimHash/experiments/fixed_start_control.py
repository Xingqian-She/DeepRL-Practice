"""同源码、同环境的 seed0 配对对照：仅通过环境 wrapper 固定每回合起点。"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args):
    import gymnasium as gym
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    sys.path.insert(0, str(args.source.resolve()))
    import train
    from config import Config
    from reset_study import VARIANTS, versions, write_json
    beta, append = VARIANTS[args.variant]
    config = Config(seed=0, total_steps=100_000, beta=beta, append_constant=append)
    original_make = gym.make
    positions = []

    class FixedReset(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.file = (args.run_dir / "resets.csv").open("w", newline="", encoding="utf-8")
            self.writer = csv.DictWriter(self.file, fieldnames=[
                "episode_index", "requested_seed", "effective_seed", "position", "velocity"])
            self.writer.writeheader()

        def reset(self, **kwargs):
            requested = kwargs.get("seed")
            kwargs["seed"] = config.seed
            obs, info = super().reset(**kwargs)
            positions.append(float(obs[0]))
            self.writer.writerow(dict(episode_index=len(positions) - 1, requested_seed=requested,
                effective_seed=config.seed, position=float(obs[0]), velocity=float(obs[1])))
            self.file.flush()
            return obs, info

        def close(self):
            try:
                super().close()
            finally:
                self.file.close()

    train.gym.make = lambda *a, **kw: FixedReset(original_make(*a, **kw))
    started = time.monotonic()
    train.train(config, args.run_dir, torch.device("cuda"))
    torch.cuda.synchronize()
    assert len(positions) > 1 and len(set(positions)) == 1
    write_json(args.run_dir / "complete.json", dict(
        reset_mode="fixed_every_episode", elapsed_seconds=time.monotonic() - started,
        checkpoint_sha256=sha(args.run_dir / "q_network.pt"), reset_count=len(positions),
        unique_initial_positions=len(set(positions)), initial_position=positions[0], versions=versions()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--variant", choices=["baseline", "simhash", "append"])
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.root / "fixed_start_control.py")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), seed=0, steps=100_000,
        reset_mode="fixed_every_episode", worker_sha256=sha(Path(__file__)),
        training_source_sha256={p.name: sha(p) for p in args.source.glob("*.py")})
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1")
    processes = []
    try:
        for variant in ["baseline", "simhash", "append"]:
            log = (args.root / f"{variant}-seed0.log").open("w", encoding="utf-8")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                       "--source", str(args.source), "--variant", variant,
                       "--run-dir", str(args.root / f"{variant}-seed0")]
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            processes.append((variant, process, log))
            print(f"START fixed {variant} pid={process.pid}", flush=True)
        (args.root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        failed = []
        for variant, process, log in processes:
            code = process.wait()
            log.close()
            print(f"FINISH fixed {variant} exit={code}", flush=True)
            if code:
                failed.append(variant)
        if failed:
            raise RuntimeError(f"Fixed controls failed: {failed}")
    finally:
        for _, process, log in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
            log.close()


if __name__ == "__main__":
    main()
