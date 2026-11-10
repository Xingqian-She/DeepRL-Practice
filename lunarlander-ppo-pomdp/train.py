"""PPO 传感器退化实验：python train.py；参数帮助：python train.py --help。"""

import argparse
from collections import deque
from dataclasses import asdict
import sys
import time

import numpy as np
import torch

from acceleration import configure_acceleration, synchronize
from config import AgentConfig, TrainConfig
from curriculum import Curriculum
from diagnose_consistency import save_snapshot
from envs import make_vector_env
from evaluate import evaluate_agent
from evaluation_runner import EvaluationRunner
from model import Agent
from ppo import ConsistencyError, PPOTrainer
from rollout import Collector
from runtime import Run, configure_runtime, save_checkpoint
from sensors import SensorConfig, input_spec


def run_training(config, output_dir=None, *, run_dir=None, experiment_id=None):
    started = time.perf_counter()
    device = configure_runtime(config.seed, config.device)
    run = Run("train", asdict(config), device, output_dir, run_dir=run_dir, experiment_id=experiment_id)
    run.metadata.update(effective_total_timesteps=config.effective_total_timesteps, input_spec=input_spec(),
                        completed_updates=0, last_completed_step=0, sampled_global_step=0,
                        training_budget_completed=False, consistency_warning_count=0,
                        evaluation_seconds=0.0)
    envs = None
    evaluator = None
    recent = deque(maxlen=50)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        agent = Agent(config.agent).to(device)
        agent.provenance = {key: run.metadata[key] for key in (
            "run_id", "experiment_id", "config_fingerprint", "source_fingerprint", "evaluation_protocol")}
        run.metadata["model"] = {"backbone_mode": config.agent.backbone_mode,
                                 "parameters": agent.parameter_counts()}
        # 在正式环境 reset/采样和 optimizer 更新前完成编译，失败回退不会污染轨迹。
        agent.acceleration = run.metadata["acceleration"]
        try:
            configure_acceleration(agent, config, device)
        finally:
            run.metadata["acceleration"] = agent.acceleration
            run.save()
        envs = make_vector_env(config.num_envs, config.max_episode_steps)
        trainer = PPOTrainer(agent, config)
        curriculum = Curriculum(config)
        best_evaluations = {}
        last_evaluation = None
        collector = Collector(envs, agent, device, config.seed,
                              sensor_config=curriculum, total_timesteps=config.effective_total_timesteps)
        run.metadata["curriculum"] = curriculum.state(0)

        def training_state(update):
            return {"completed_updates": update, "curriculum": curriculum.state(collector.global_step),
                    "last_evaluation": last_evaluation, "best_evaluations": dict(best_evaluations)}

        def evidence(report, severity, update, buffer):
            path = run.path / "diagnostics" / f"consistency-{severity}-update{update:06d}.pt"
            save_snapshot(path, agent, buffer, config, update=update, global_step=collector.global_step,
                          report=report, environment=dict(run.metadata), severity=severity)
            run.metadata[f"{severity}_snapshot"] = str(path.relative_to(run.path))
            print(f"Consistency {severity}: {report['consistency_max_logprob_error']}; snapshot: {path}", flush=True)

        # A. 采样 → B. 概率核对和 GAE → C. PPO 更新 → D. 记录与保存。
        for update in range(1, config.num_updates + 1):
            synchronize(device)
            collect_started = time.perf_counter()
            buffer = collector.collect(config.num_steps)
            run.metadata["sampled_global_step"] = collector.global_step
            synchronize(device)
            collect_seconds = time.perf_counter() - collect_started
            ppo_started = time.perf_counter()
            def warning(report):
                run.metadata["consistency_warning_count"] += 1
                if "warning_snapshot" not in run.metadata:
                    evidence(report, "warning", update, buffer)
            try:
                metrics = trainer.update(buffer, on_consistency_warning=warning, update_index=update)
            except ConsistencyError as error:
                run.metadata.update(failed_sampled_step=collector.global_step, failed_update=update)
                evidence(error.report, "failure", update, buffer)
                raise
            synchronize(device)
            ppo_seconds = time.perf_counter() - ppo_started
            metrics.update(collect_seconds=collect_seconds, ppo_seconds=ppo_seconds,
                           steps_per_second=config.num_envs * config.num_steps / (collect_seconds + ppo_seconds))
            collector.invalidate_cache()
            for episode in buffer.episodes:
                recent.append(episode)
                run.append_csv("episodes.csv", episode)
                run.writer.add_scalar("episode/return", episode["return"], episode["global_step"])
                run.writer.add_scalar("episode/length", episode["length"], episode["global_step"])
            sensor_metrics = buffer.sensor_metrics()
            trained_stage = curriculum.stage
            curriculum.record_steps(config.num_envs * config.num_steps, collector.global_step)
            row = {"update": update, "global_step": collector.global_step, "curriculum_stage": trained_stage,
                   **metrics, **sensor_metrics,
                   "mean_recent_return": float(np.mean([e["return"] for e in recent])) if recent else None,
                   "mean_recent_length": float(np.mean([e["length"] for e in recent])) if recent else None}
            run.append_csv("training.csv", row)
            if trainer.last_diagnostics is not None:
                diagnostic = {"update": update, "global_step": collector.global_step,
                              **trainer.last_diagnostics}
                run.append_csv("diagnostics.csv", diagnostic)
                for name, value in trainer.last_diagnostics.items():
                    if value is not None:
                        run.writer.add_scalar(f"diagnostics/{name}", value, collector.global_step)
            for name, value in row.items():
                if value is not None and name not in ("update", "global_step"):
                    run.writer.add_scalar(f"train/{name}", value, collector.global_step)
            run.writer.flush()
            run.metadata.update(global_step=collector.global_step, completed_updates=update,
                                last_completed_step=collector.global_step, sensor_metrics=sensor_metrics)
            if config.eval_interval and (update % config.eval_interval == 0 or update == config.num_updates):
                evaluation_started = time.perf_counter()
                if evaluator is None:
                    evaluator = EvaluationRunner(agent, num_envs=config.eval_num_envs,
                                                  max_episode_steps=config.max_episode_steps)
                probability = collector.failure_probability
                summary = evaluate_agent(agent, episodes=config.eval_episodes, seed=config.eval_seed,
                                         policy=config.eval_policy, sensor_failure_probability=probability,
                                         max_episode_steps=config.max_episode_steps,
                                         num_envs=config.eval_num_envs, runner=evaluator)
                evaluation_seconds = time.perf_counter() - evaluation_started
                run.metadata["evaluation_seconds"] += evaluation_seconds
                evaluated_steps = sum(episode["length"] for episode in summary["episodes"])
                last_evaluation = {"update": update, "global_step": collector.global_step,
                                   "stage": curriculum.stage, "seed": config.eval_seed,
                                   "episode_count": config.eval_episodes,
                                   **{k: v for k, v in summary.items() if k != "episodes"},
                                   "evaluation_seconds": evaluation_seconds,
                                   "evaluation_environment_steps": evaluated_steps,
                                   "evaluation_steps_per_second": evaluated_steps / evaluation_seconds,
                                   "evaluation_num_envs": min(config.eval_num_envs, config.eval_episodes)}
                run.append_csv("evaluation.csv", last_evaluation)
                for episode in summary["episodes"]:
                    run.append_csv("evaluation_episodes.csv", {"update": update, "global_step": collector.global_step,
                                   "sensor_failure_probability": probability, **episode})
                for name, value in last_evaluation.items():
                    if isinstance(value, (int, float)) and name not in ("update", "global_step"):
                        run.writer.add_scalar(f"evaluation/{name}", value, collector.global_step)
                key = curriculum.checkpoint_key(probability)
                if key is not None and (key not in best_evaluations
                                        or summary["mean_return"] > best_evaluations[key]["mean_return"]):
                    best_evaluations[key] = dict(last_evaluation)
                    save_checkpoint(run.path / "checkpoints" / f"best_{key}.pt", agent, config,
                                    collector.global_step, training_state(update))
                transition = curriculum.observe(summary, collector.global_step)
                if transition:
                    run.append_csv("curriculum_transitions.csv", transition)
                    save_checkpoint(run.path / "checkpoints" / f"transition_stage_{trained_stage:02d}.pt",
                                    agent, config, collector.global_step, training_state(update))
                    collector.refresh_pending_observation()
                    print(f"Curriculum: p={transition['from_probability']:.2f} -> "
                          f"{transition['to_probability']:.2f}", flush=True)
                print(f"Evaluation p={probability:.3f}: mean={summary['mean_return']:.1f}, "
                      f"score>=200={summary['score_200_rate']:.0%}, {evaluation_seconds:.2f}s", flush=True)
            run.metadata.update(curriculum=curriculum.state(collector.global_step),
                                last_evaluation=last_evaluation, best_evaluations=dict(best_evaluations),
                                entered_final=curriculum.reached_final, final_task_solved=curriculum.final_task_solved)
            run.writer.flush()
            if update % config.save_interval == 0 or update == config.num_updates:
                save_checkpoint(run.path / "checkpoints" / "latest.pt", agent, config,
                                collector.global_step, training_state(update))
            run.save()
            score = f"{row['mean_recent_return']:.1f}" if recent else "尚无完整回合"
            print(f"update {update}/{config.num_updates} | steps {collector.global_step} | "
                  f"最近回报 {score} | p失灵 {collector.failure_probability:.3f} | "
                  f"logp误差 {metrics['consistency_max_logprob_error']:.2g} | "
                  f"{metrics['steps_per_second']:.0f} steps/s", flush=True)
        run.metadata["status"] = "completed"
        run.metadata["training_budget_completed"] = True
    except KeyboardInterrupt:
        run.metadata["status"] = "interrupted"
        raise
    except BaseException as error:
        run.metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if envs is not None:
                envs.close()
        finally:
            try:
                if evaluator is not None:
                    evaluator.close()
            finally:
                run.metadata["elapsed_seconds"] = time.perf_counter() - started
                run.metadata["evaluation_fraction"] = run.metadata["evaluation_seconds"] / run.metadata["elapsed_seconds"]
                if device.type == "cuda":
                    run.metadata["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024**2
                run.close()
    return run.path


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for name in ("total_timesteps", "num_envs", "num_steps", "sequence_length", "sequences_per_minibatch",
                 "update_epochs", "max_episode_steps", "seed", "save_interval", "eval_interval",
                 "eval_episodes", "eval_num_envs", "eval_seed", "curriculum_patience", "diagnostic_interval"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in ("learning_rate", "gamma", "gae_lambda", "clip_coef", "vf_coef", "ent_coef",
                 "max_grad_norm", "consistency_tolerance", "consistency_warning_tolerance", "target_kl",
                 "curriculum_step", "curriculum_return_threshold", "curriculum_success_rate",
                 "learning_rate_final", "ent_coef_final", "value_clip_coef"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    for name in ("hidden_dim", "num_layers", "num_heads", "context_length"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults.agent, name))
    parser.add_argument("--alpha-init", type=float, default=defaults.agent.alpha_init)
    parser.add_argument("--previous-action", dest="use_previous_action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction,
                        default=defaults.normalize_advantage)
    parser.add_argument("--lr-schedule", choices=("constant", "linear"), default=defaults.lr_schedule)
    parser.add_argument("--ent-coef-schedule", choices=("constant", "linear"), default=defaults.ent_coef_schedule)
    parser.add_argument("--backbone-mode", choices=("shared", "separate"), default=defaults.agent.backbone_mode,
                        help="separate 为策略/价值各建一套同宽同深的 Encoder/Transformer")
    parser.add_argument("--sensor-mode", choices=("curriculum", "full", "partial"), default="curriculum")
    parser.add_argument("--curriculum-start-fraction", type=float, default=0.1)
    parser.add_argument("--curriculum-end-fraction", type=float, default=0.8)
    parser.add_argument("--curriculum-schedule", choices=("fixed", "performance"), default="fixed")
    parser.add_argument("--eval-policy", choices=("sample", "argmax"), default=defaults.eval_policy)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compile", dest="compile_mode", choices=("auto", "on", "off"),
                        default=defaults.compile_mode, help="默认 auto：Linux/CUDA 预热成功后启用 Inductor")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output-dir", help="父目录；每次仍在其中创建唯一 run 子目录")
    outputs.add_argument("--run-dir", help="准确运行目录；存在则拒绝覆盖")
    parser.add_argument("--experiment-id")
    return parser


def main():
    args = vars(make_parser().parse_args())
    output_dir = args.pop("output_dir")
    run_dir, experiment_id = args.pop("run_dir"), args.pop("experiment_id")
    model_args = {name: args.pop(name) for name in (
        "hidden_dim", "num_layers", "num_heads", "context_length", "alpha_init", "use_previous_action", "backbone_mode")}
    try:
        sensors = SensorConfig(mode=args.pop("sensor_mode"),
                               start_fraction=args.pop("curriculum_start_fraction"),
                               end_fraction=args.pop("curriculum_end_fraction"),
                               schedule=args.pop("curriculum_schedule"))
        config = TrainConfig(agent=AgentConfig(**model_args), sensors=sensors, **args)
        run_training(config, output_dir, run_dir=run_dir, experiment_id=experiment_id)
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
