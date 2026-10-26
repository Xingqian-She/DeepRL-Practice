"""Validate CartPole rendering, video and TensorBoard without training a model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import uuid

import gymnasium as gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def seed_value(value: str) -> int:
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("must be between 0 and 2**32 - 1")
    return number


def git_info() -> dict:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={root.as_posix()}", *args],
            cwd=root, text=True, stderr=subprocess.DEVNULL, timeout=10,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        ).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"),
                "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("none", "human", "video"), default="none")
    parser.add_argument("--episodes", type=positive_int, default=3)
    parser.add_argument("--seed", type=seed_value, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs",
        help="Parent directory; a unique run subdirectory is always created.",
    )
    args = parser.parse_args()
    if args.mode == "human" and not os.environ.get("DISPLAY"):
        parser.error("human mode requires WSLg DISPLAY; use compose.gui.yaml from WSL")
    if args.mode == "video":
        # SDL surface rendering works without an X11/Wayland server.
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    created = datetime.now(timezone.utc)
    run_id = f"{created:%Y%m%dT%H%M%S%fZ}-seed{args.seed}-{uuid.uuid4().hex[:8]}"
    run_dir = args.output_dir.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "run_id": run_id,
        "created_at": created.isoformat(),
        "environment_id": "CartPole-v1",
        "policy": "random",
        "parameters": {"mode": args.mode, "episodes": args.episodes, "seed": args.seed},
        "output_dir": str(run_dir),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "device": {"policy": "cpu", "cuda_available": torch.cuda.is_available(),
                   "torch_cuda_version": torch.version.cuda},
        "git": git_info(),
        "status": "running",
        "episodes": [],
    }

    def save_metadata() -> None:
        target = run_dir / "metadata.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)

    save_metadata()
    print(f"Run directory: {run_dir}", flush=True)
    env = None
    writer = None
    try:
        render_mode = {"none": None, "human": "human", "video": "rgb_array"}[args.mode]
        env = gym.make("CartPole-v1", render_mode=render_mode)
        if args.mode == "video":
            env = gym.wrappers.RecordVideo(
                env, video_folder=str(run_dir / "videos"),
                episode_trigger=lambda episode: True, name_prefix="cartpole-random",
                disable_logger=True,
            )
        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
        global_step = 0
        for episode in range(args.episodes):
            # Offset seeds without reseeding each step; seed actions as well as resets.
            episode_seed = (args.seed + episode) % 2**32
            env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            total_reward, length = 0.0, 0
            while True:
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                total_reward += float(reward)
                length += 1
                global_step += 1
                if terminated or truncated:
                    break
            result = {"episode": episode + 1, "seed": episode_seed,
                      "return": total_reward, "length": length, "global_step": global_step}
            metadata["episodes"].append(result)
            writer.add_scalar("episode/return", total_reward, global_step)
            writer.add_scalar("episode/length", length, global_step)
            writer.flush()
            save_metadata()
            print(f"Episode {episode + 1}: return={total_reward:g}, length={length}", flush=True)
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            if env is not None:
                env.close()  # RecordVideo flushes its final MP4 here.
        except BaseException as error:
            metadata["status"] = "failed"
            metadata["error"] = f"close: {type(error).__name__}: {error}"
            raise
        finally:
            try:
                if writer is not None:
                    writer.close()
            except BaseException as error:
                metadata["status"] = "failed"
                metadata["error"] = f"writer close: {type(error).__name__}: {error}"
                raise
            finally:
                if metadata["status"] == "running":
                    metadata["status"] = "completed"
                metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
                save_metadata()


if __name__ == "__main__":
    main()
