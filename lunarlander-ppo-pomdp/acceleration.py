"""可选 Inductor 数值核心；仅在正式采样和 optimizer 更新前允许回退。"""

import importlib.util
import math
import platform
import random
import time

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from model import cache_tensors, make_model_input
from sensors import OBSERVATION_DIM


def eager_status(requested="off", reason="disabled"):
    return {"requested": requested, "enabled": False, "paths": [], "backend": None,
            "mode": None, "cuda_graphs": False, "warmup_seconds": 0.0, "reason": reason}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _unsupported_reason(device):
    if platform.system() != "Linux" or device.type != "cuda":
        return "compile requires Linux/CUDA"
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return "CUDA build/device unavailable"
    if not callable(getattr(torch, "compile", None)) or importlib.util.find_spec("triton") is None:
        return "torch.compile or Triton unavailable"
    if torch.cuda.get_device_capability(device)[0] < 7:
        return "CUDA compute capability must be >= 7.0"
    return None


def _sequence_inputs(config, batch_size, length, device):
    # 确定性非零输入；不消耗训练、环境或传感器的随机流。
    observations = torch.arange(batch_size * length * OBSERVATION_DIM, device=device,
                                dtype=torch.float32).reshape(batch_size, length, OBSERVATION_DIM).sin()
    previous = torch.zeros(batch_size, length, dtype=torch.long, device=device)
    previous[:, 0] = -1
    features = make_model_input(observations, previous, config.agent.use_previous_action,
                                xy_fourier=config.agent.xy_fourier)
    starts = torch.zeros(batch_size, length, dtype=torch.bool, device=device)
    starts[:, 0] = True
    if length > 2:
        starts[0, length // 2] = True
    valid = torch.ones_like(starts)
    if length > 2:
        valid[-1, -1] = False
    times = torch.arange(length, device=device).expand(batch_size, length)
    positions = times - torch.where(starts, times, 0).cummax(dim=-1).values
    return features, starts, valid, positions


def _assert_outputs(actual, expected):
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=5e-4, atol=5e-5)


def _warmup(agent, config, device):
    """临时 cache；覆盖 reset、初始/完整预热序列及 grad/no-grad 两种图。"""
    with sdpa_kernel(SDPBackend.MATH):
        agent.eval()
        features, _, _, _ = _sequence_inputs(config, config.num_envs, 1, device)
        features = features[:, 0].contiguous()
        eager_cache, compiled_cache = agent.create_cache(config.num_envs), agent.create_cache(config.num_envs)
        with torch.no_grad():
            for index in range(3):
                starts = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
                if index == 0:
                    starts[:] = True
                elif index == 2:
                    starts[0] = True
                expected = agent._step_core(features, eager_cache, starts)
                actual = agent._compiled_step(features, compiled_cache, starts)
                _assert_outputs(actual, expected)
                for name, tensor in cache_tensors(compiled_cache).items():
                    torch.testing.assert_close(tensor, cache_tensors(eager_cache)[name], rtol=5e-4, atol=5e-5)

        sequence_count = config.num_envs * math.ceil(config.num_steps / config.sequence_length)
        sizes = {min(config.sequences_per_minibatch, sequence_count)}
        if sequence_count % config.sequences_per_minibatch:
            sizes.add(sequence_count % config.sequences_per_minibatch)
        lengths = {min(config.sequence_length, config.num_steps),
                   config.agent.warmup_length + min(config.sequence_length, config.num_steps)}
        for batch_size in sorted(sizes):
            for length in sorted(lengths):
                inputs = _sequence_inputs(config, batch_size, length, device)
                agent.eval()
                with torch.no_grad():
                    expected = agent._sequence_core(*inputs)
                    _assert_outputs(agent._compiled_sequence(*inputs), expected)
                agent.train()
                with torch.enable_grad():
                    agent.zero_grad(set_to_none=True)
                    outputs = agent._compiled_sequence(*inputs)
                    _assert_outputs(outputs, expected)
                    (outputs[0].square().mean() + outputs[1].square().mean()).backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in agent.parameters()):
                        raise RuntimeError("compile warmup produced missing/nonfinite gradients")
    synchronize(device)


def configure_acceleration(agent, config, device):
    """安装独立函数，预热成功才宣布启用；后续异常不捕获、不重试。"""
    agent._compiled_step = agent._compiled_sequence = None
    status = eager_status(config.compile_mode)
    agent.acceleration = status
    if config.compile_mode == "off":
        return status
    reason = _unsupported_reason(device)
    if reason:
        status["reason"] = reason
        if config.compile_mode == "on":
            raise RuntimeError(reason)
        return status

    print("Compile warmup: FP32/math SDPA, Inductor default, CUDA Graphs disabled...", flush=True)
    started = time.perf_counter()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    was_training = agent.training
    try:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            agent._compiled_step = torch.compile(
                agent._step_core, backend="inductor", fullgraph=True, dynamic=False,
                options={"triton.cudagraphs": False},
            )
            agent._compiled_sequence = torch.compile(
                agent._sequence_core, backend="inductor", fullgraph=True, dynamic=True,
                options={"triton.cudagraphs": False},
            )
            _warmup(agent, config, device)
    except Exception as error:
        agent._compiled_step = agent._compiled_sequence = None
        status["reason"] = f"warmup failed: {type(error).__name__}: {error}"
        if config.compile_mode == "on":
            raise RuntimeError(status["reason"]) from error
        print(f"Compile fallback to eager: {status['reason']}", flush=True)
    else:
        status.update(enabled=True, paths=["step", "history_replay", "ppo_sequence"],
                      backend="inductor", mode="default", reason=None)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        agent.zero_grad(set_to_none=True)
        agent.train(was_training)
        status["warmup_seconds"] = time.perf_counter() - started
    print(f"Compile enabled={status['enabled']} | warmup {status['warmup_seconds']:.2f}s", flush=True)
    return status
