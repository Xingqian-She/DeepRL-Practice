"""Save/replay pre-update evidence without resampling sensors or updating weights."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from acceleration import configure_acceleration
from buffer import RolloutBuffer
from config import AgentConfig, TrainConfig
from history import Token
from model import Agent, make_model_input
from runtime import configure_runtime, write_json
from sensors import SensorConfig


def save_snapshot(path, agent, buffer, config, *, update, global_step, report, environment, severity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    numpy_rng = np.random.get_state()
    payload = {
        "snapshot_version": 1, "severity": severity, "update": update, "global_step": global_step,
        "parameter_version": update - 1, "train_config": asdict(config), "report": report,
        "environment": environment,
        "state_dict": {k: v.detach().cpu().clone() for k, v in agent.state_dict().items()},
        "buffer": {"num_steps": buffer.num_steps, "num_envs": buffer.num_envs,
                   "warmup_length": buffer.warmup_length, "step": buffer.step,
                   "sensor_probability_start": buffer.sensor_probability_start,
                   "sensor_probability_end": buffer.sensor_probability_end, "episodes": buffer.episodes,
                   "prefix": [{k: getattr(t, k).detach().cpu().clone() for k in t.__dataclass_fields__}
                              for t in buffer.prefix],
                   "tensors": {k: v.detach().cpu().clone() for k, v in vars(buffer).items()
                               if isinstance(v, torch.Tensor)}},
        "rng": {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if next(agent.parameters()).is_cuda else [],
                "numpy": {"algorithm": numpy_rng[0], "keys": numpy_rng[1].tolist(),
                          "position": numpy_rng[2], "has_gauss": numpy_rng[3], "cached_gaussian": numpy_rng[4]}},
    }
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def restore_buffer(saved, device):
    prefix = [Token(**{k: v.to(device) for k, v in token.items()}) for token in saved["prefix"]]
    buffer = RolloutBuffer(saved["num_steps"], saved["num_envs"], device, prefix, saved["warmup_length"])
    for key, value in saved["tensors"].items():
        setattr(buffer, key, value.to(device))
    for key in ("step", "episodes", "sensor_probability_start", "sensor_probability_end"):
        setattr(buffer, key, saved[key])
    return buffer


def error_summary(actual, expected, buffer):
    delta = actual.double() - expected.double()
    finite = torch.isfinite(delta)
    finite_errors = delta[finite].abs()
    magnitude = torch.nan_to_num(delta.abs(), nan=float("inf"), posinf=float("inf"))
    index = magnitude.flatten().argmax().item()
    time, env = divmod(index, buffer.num_envs)
    def number(value):
        return value.item() if torch.isfinite(value) else None
    return {"finite": bool(finite.all()), "samples": delta.numel(),
            "max_abs_error": magnitude.max().item() if finite.all() else None,
            "rms_error": delta.square().mean().sqrt().item() if finite.all() else None,
            **{f"p{q}_abs_error": torch.quantile(finite_errors, q / 100).item() if len(finite_errors) else None
               for q in (50, 95, 99)},
            "worst": {"time": time, "env": env, "action": buffer.actions[time, env].item(),
                      "position": buffer.position_ids[time, env].item(),
                      "actual_logp": number(actual[time, env]), "expected_logp": number(expected[time, env])}}


@torch.no_grad()
def replay_snapshot(path, *, device="cpu", precision="fp32", compile_mode="off"):
    if precision not in ("fp32", "fp64") or (precision == "fp64" and compile_mode != "off"):
        raise ValueError("FP64 仅支持 eager；precision 必须是 fp32 或 fp64")
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if saved.get("snapshot_version") != 1:
        raise ValueError("不支持的诊断现场版本")
    device = configure_runtime(0, device)
    settings = dict(saved["train_config"])
    settings["agent"] = AgentConfig(**settings["agent"])
    settings["sensors"] = SensorConfig(**settings["sensors"])
    config = replace(TrainConfig(**settings), device=str(device), compile_mode=compile_mode)
    dtype = torch.float64 if precision == "fp64" else torch.float32
    agent = Agent(config.agent).to(device=device, dtype=dtype)
    agent.load_state_dict(saved["state_dict"], strict=True)
    agent.eval()
    if precision == "fp32":
        configure_acceleration(agent, config, device)
    buffer = restore_buffer(saved["buffer"], device)
    shape = (buffer.num_steps, buffer.num_envs)
    serial = torch.empty(shape, dtype=dtype, device=device)
    parallel = torch.empty_like(serial)
    cache = agent.create_cache(buffer.num_envs)
    for token in buffer.prefix:
        agent.step_outputs(token.observations.to(dtype), token.previous_actions, cache, token.episode_starts)
    position_mismatches = 0
    for time in range(buffer.num_steps):
        positions = torch.where(buffer.episode_starts[time], 0, cache.next_positions)
        position_mismatches += int((positions != buffer.position_ids[time]).sum().item())
        logits, _ = agent.step_outputs(buffer.observations[time].to(dtype), buffer.previous_actions[time],
                                      cache, buffer.episode_starts[time])
        serial[time] = torch.distributions.Categorical(logits=logits, validate_args=False).log_prob(buffer.actions[time])
    specs = buffer.sequence_specs(config.sequence_length)
    for index, batch in enumerate(buffer.batches(config.sequence_length, config.sequences_per_minibatch,
                                                  shuffle=False)):
        features = make_model_input(batch["observations"].to(dtype), batch["previous_actions"],
                                    config.agent.use_previous_action)
        with sdpa_kernel(SDPBackend.MATH):
            core = agent._compiled_sequence or agent._sequence_core
            logits, _ = core(features, batch["episode_starts"], batch["valid_mask"], batch["position_ids"])
        logp = torch.distributions.Categorical(logits=logits, validate_args=False).log_prob(batch["actions"])
        group = specs[index * config.sequences_per_minibatch:(index + 1) * config.sequences_per_minibatch]
        for row, (env, begin, end) in enumerate(group):
            parallel[begin:end, env] = logp[row, batch["loss_mask"][row]]
    sampling = torch.distributions.Categorical(logits=buffer.sampling_logits, validate_args=False).log_prob(buffer.actions)
    return {"snapshot": str(Path(path).resolve()), "snapshot_update": saved["update"],
            "global_step": saved["global_step"], "device": str(device), "precision": precision,
            "compile_requested": compile_mode, "compile_enabled": bool(agent._compiled_step),
            "position_mismatches": position_mismatches,
            "sampling_logits_vs_saved": error_summary(sampling, buffer.old_log_probs, buffer),
            "serial_vs_saved": error_summary(serial, buffer.old_log_probs, buffer),
            "parallel_vs_saved": error_summary(parallel, buffer.old_log_probs, buffer),
            "parallel_vs_serial": error_summary(parallel, serial, buffer)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp32")
    parser.add_argument("--compile", dest="compile_mode", choices=("off", "auto", "on"), default="off")
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args())
    output = args.pop("output")
    try:
        result = replay_snapshot(args.pop("snapshot"), **args)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
