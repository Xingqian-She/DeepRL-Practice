"""Compare the former serial evaluator with reusable batched evaluation on real Box2D."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import statistics
import time

import numpy as np
import torch

from acceleration import synchronize
from config import AgentConfig
from envs import make_env
from evaluation_runner import EvaluationRunner
from model import Agent
from runtime import configure_runtime, preserve_rng_state, source_fingerprint, write_json
from sensors import SensorProcessor


@torch.no_grad()
def legacy_evaluate(agent, episodes, seed, policy, probability, max_episode_steps):
    """The pre-refactor hot path, retained only as a performance reference."""
    device = next(agent.parameters()).device
    with preserve_rng_state(device):
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        reference = Agent(agent.config).to(device=device, dtype=torch.float32)
        reference.load_state_dict(agent.state_dict(), strict=True)
        reference.eval()
        env = make_env(max_episode_steps)
        returns, steps, entropy_sum, main_sum = [], 0, 0.0, 0.0
        try:
            for episode in range(episodes):
                episode_seed = (seed + episode) % 2**32
                observation, _ = env.reset(seed=episode_seed)
                sensors = SensorProcessor(episode_seed)
                cache = reference.create_cache(1)
                previous = torch.full((1,), -1, dtype=torch.long, device=device)
                starts = torch.ones(1, dtype=torch.bool, device=device)
                total = 0.0
                while True:
                    processed = sensors.process(observation, probability)
                    inputs = torch.as_tensor(processed, dtype=torch.float32, device=device).unsqueeze(0)
                    logits, _ = reference.step_outputs(inputs, previous, cache, starts)
                    distribution = torch.distributions.Categorical(logits=logits)
                    action = logits.argmax(-1) if policy == "argmax" else distribution.sample()
                    index = int(action.item())
                    entropy_sum += distribution.entropy().item()
                    main_sum += distribution.probs[0, 2].item()
                    observation, reward, terminated, truncated, _ = env.step(index)
                    total += float(reward)
                    steps += 1
                    if terminated or truncated:
                        break
                    previous = action.clone()
                    starts.zero_()
                returns.append(total)
        finally:
            env.close()
        return {"environment_steps": steps, "mean_return": float(np.mean(returns)),
                "score_200_rate": float(np.mean(np.asarray(returns) >= 200)),
                "policy_entropy": entropy_sum / steps, "main_engine_probability": main_sum / steps}


def benchmark(output, *, device="cuda", episodes=(20, 100), repeats=3, num_envs=8,
              policy="sample", probability=0.0, max_episode_steps=1000):
    if repeats < 1 or num_envs < 1 or not episodes or min(episodes) < 1:
        raise ValueError("repeats, num_envs and episodes must be positive")
    device = configure_runtime(42, device)
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "torch": str(torch.__version__),
              "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "source_fingerprint": source_fingerprint(), "num_envs": num_envs, "repeats": repeats,
              "policy": policy, "sensor_failure_probability": probability, "max_episode_steps": max_episode_steps,
              "weights": "fixed untrained D256/L4/H4 initialization, seed42; not a learning result",
              "timing": "CUDA synchronized wall time, first batched call includes snapshot creation; later calls reuse it",
              "sampling": "legacy global Torch RNG vs episode-local NumPy RNG; sample trajectories may differ; compare steps/s too",
              "trials": [], "summary": {}}
    for mode in ("shared", "separate"):
        configure_runtime(42, str(device))
        agent = Agent(AgentConfig(backbone_mode=mode)).to(device)
        result.setdefault("models", {})[mode] = {"config": asdict(agent.config), "parameters": agent.parameter_counts()}
        for count in episodes:
            runner = None
            try:
                for repeat in range(repeats):
                    for backend in (("serial", "batched") if repeat % 2 == 0 else ("batched", "serial")):
                        synchronize(device)
                        started = time.perf_counter()
                        if backend == "serial":
                            summary = legacy_evaluate(agent, count, 10000, policy, probability, max_episode_steps)
                        else:
                            if runner is None:
                                runner = EvaluationRunner(agent, num_envs=num_envs, max_episode_steps=max_episode_steps)
                            evaluated = runner.evaluate(agent, episodes=count, seed=10000, policy=policy,
                                                        sensor_failure_probability=probability)
                            summary = {key: evaluated[key] for key in ("mean_return", "score_200_rate", "policy_entropy", "main_engine_probability")}
                            summary["environment_steps"] = sum(row["length"] for row in evaluated["episodes"])
                        synchronize(device)
                        elapsed = time.perf_counter() - started
                        trial = {"backbone_mode": mode, "episodes": count, "backend": backend, "repeat": repeat + 1,
                                 "seconds": elapsed, "steps_per_second": summary["environment_steps"] / elapsed, **summary}
                        result["trials"].append(trial)
                        write_json(output, result)
                        print(f"{mode} episodes={count} {backend} repeat={repeat + 1}: "
                              f"{elapsed:.3f}s, {trial['environment_steps']} steps, {trial['steps_per_second']:.1f} steps/s", flush=True)
            finally:
                if runner is not None:
                    runner.close()
            selected = [row for row in result["trials"] if row["backbone_mode"] == mode and row["episodes"] == count]
            serial = [row for row in selected if row["backend"] == "serial"]
            batched = [row for row in selected if row["backend"] == "batched"]
            result["summary"][f"{mode}-{count}"] = {
                "serial_median_seconds": statistics.median(row["seconds"] for row in serial),
                "batched_median_seconds": statistics.median(row["seconds"] for row in batched),
                "batched_first_seconds": batched[0]["seconds"],
                "batched_reused_median_seconds": statistics.median(row["seconds"] for row in batched[1:]) if len(batched) > 1 else None,
                "wall_speedup": statistics.median(row["seconds"] for row in serial) / statistics.median(row["seconds"] for row in batched),
                "throughput_speedup": statistics.median(row["steps_per_second"] for row in batched) / statistics.median(row["steps_per_second"] for row in serial),
            }
            write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/validation/evaluation-benchmark.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--episodes", type=int, nargs="+", default=[20, 100])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--policy", choices=("sample", "argmax"), default="sample")
    parser.add_argument("--probability", type=float, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=1000)
    benchmark(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
