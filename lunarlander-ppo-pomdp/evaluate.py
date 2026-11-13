"""Independent batched policy evaluation, shared by training and the CLI."""

import argparse
import os
from pathlib import Path
import sys

from evaluation_runner import EvaluationRunner
from runtime import Run, configure_runtime, load_checkpoint, write_json
from sensors import input_spec, validate_probability


def evaluate_agent(agent, *, episodes=20, seed=10000, sensor_failure_probability=1.0,
                   policy="sample", max_episode_steps=1000, env_factory=None,
                   mode="none", video_dir=None, on_episode=None, num_envs=8, runner=None):
    """Pass a runner to reuse its policy snapshot/environment pool across evaluations."""
    owned = runner is None
    if owned:
        runner = EvaluationRunner(agent, num_envs=num_envs, max_episode_steps=max_episode_steps,
                                  env_factory=env_factory, mode=mode, video_dir=video_dir)
    try:
        return runner.evaluate(agent, episodes=episodes, seed=seed, policy=policy,
                               sensor_failure_probability=sensor_failure_probability, on_episode=on_episode)
    finally:
        if owned:
            runner.close()


def run_evaluation(checkpoint, *, episodes=10, seed=10000, device="auto", mode="none",
                   policy="argmax", use_previous_action=None, output_dir=None, run_dir=None,
                   experiment_id=None, num_envs=8, sensor_failure_probability=1.0, env_factory=None):
    if type(episodes) is not int or episodes < 1 or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("episodes 必须为正整数，seed 必须在 [0, 2**32) 内")
    if type(num_envs) is not int or num_envs < 1:
        raise ValueError("num_envs 必须为正整数")
    if mode not in ("none", "human", "video") or policy not in ("argmax", "sample"):
        raise ValueError("无效的评估 mode 或 policy")
    validate_probability(sensor_failure_probability)
    if mode == "human" and not os.environ.get("DISPLAY"):
        raise ValueError("human 模式需要 WSLg DISPLAY，请从 WSL 使用 compose.gui.yaml")
    if mode == "video":
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    device = configure_runtime(seed, device)
    agent, saved = load_checkpoint(checkpoint, device, use_previous_action)
    parameters = {"checkpoint": str(Path(checkpoint).resolve()), "episodes": episodes, "seed": seed,
                  "num_envs": num_envs, "mode": mode, "policy": policy,
                  "training_use_previous_action": saved["agent_config"]["use_previous_action"],
                  "effective_use_previous_action": agent.config.use_previous_action,
                  "sensor_failure_probability": sensor_failure_probability,
                  "training_sensors": saved["train_config"]["sensors"],
                  "input_spec": input_spec(xy_fourier=agent.config.xy_fourier)}
    parameters.update(training_seed=saved["train_config"]["seed"], training_parameters=saved["train_config"],
                      checkpoint_name=Path(checkpoint).name)
    run = Run("evaluate", parameters, device, output_dir, run_dir=run_dir, experiment_id=experiment_id)
    runner = None
    try:
        run.metadata.update(training_provenance=saved.get("provenance"),
                            checkpoint_global_step=saved["global_step"])
        runner = EvaluationRunner(agent, num_envs=num_envs, max_episode_steps=saved["train_config"]["max_episode_steps"],
                                  env_factory=env_factory, mode=mode, video_dir=run.path / "videos")
        completed = 0
        def record(row):
            nonlocal completed
            completed += 1
            run.append_csv("episodes.csv", row)
            print(f"completed {completed}/{episodes} | episode {row['episode']} | "
                  f"return {row['return']:.2f} | length {row['length']}", flush=True)
        summary = evaluate_agent(agent, episodes=episodes, seed=seed, policy=policy,
                                 sensor_failure_probability=sensor_failure_probability, runner=runner, on_episode=record)
        summary.update(parameters=parameters, performance=runner.last_performance,
                       training_provenance=saved.get("provenance"), checkpoint_global_step=saved["global_step"])
        write_json(run.path / "evaluation.json", summary)
        run.metadata.update(status="completed", completed_episodes=episodes,
                            global_step=sum(row["length"] for row in summary["episodes"]),
                            **runner.last_performance)
        print(f"Mean return: {summary['mean_return']:.2f} | score>=200: {summary['score_200_rate']:.0%} | "
              f"evaluation: {runner.last_performance['evaluation_seconds']:.2f}s", flush=True)
    except KeyboardInterrupt:
        run.metadata["status"] = "interrupted"
        raise
    except BaseException as error:
        run.metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if runner is not None:
                runner.close()
        finally:
            run.close()
    return run.path


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mode", choices=("none", "human", "video"), default="none",
                        help="video/human 实际使用 1 个环境")
    parser.add_argument("--policy", choices=("argmax", "sample"), default="argmax")
    parser.add_argument("--previous-action", dest="use_previous_action", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--sensor-failure-probability", type=float, default=1.0)
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output-dir")
    outputs.add_argument("--run-dir", help="准确运行目录；存在则拒绝覆盖")
    parser.add_argument("--experiment-id")
    return parser


def main():
    try:
        run_evaluation(**vars(make_parser().parse_args()))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
