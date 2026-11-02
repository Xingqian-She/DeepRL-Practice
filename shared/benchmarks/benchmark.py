"""Reproducible wall-latency measurements; no speedup target or extra dependencies."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch
from torch.profiler import profile, ProfilerActivity

from rl_shared import TransformerBackbone, TransformerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.iterations, args.warmup) <= 0 or args.seq_len < 2:
        parser.error("batch-size/iterations/warmup must be positive; seq-len must be at least 2")
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.manual_seed(1234)
    torch.set_num_threads(1)
    config = TransformerConfig(hidden_dim=args.hidden_dim, num_layers=args.layers,
                               num_heads=args.heads, context_length=args.window)
    model = TransformerBackbone(config).to(device)
    parallel = torch.compile(model, fullgraph=True) if args.compile else model
    x = torch.randn(args.batch_size, args.seq_len, args.hidden_dim, device=device)
    token = x[:, -1:].contiguous()
    reset_mask = torch.arange(args.batch_size, device=device) < max(1, math.ceil(args.batch_size * 0.1))

    def amp():
        return torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def measure(name, callback, tokens, resets=0):
        for _ in range(args.warmup):
            callback()
        synchronize()
        gc.collect()
        baseline = None
        if device.type == "cuda":
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            callback()
            synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        # Profile separately: profiler setup/teardown is not part of timings.
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            callback()
            synchronize()
        sdpa = sorted({event.key for event in prof.key_averages() if "scaled_dot_product" in event.key})
        median = statistics.median(samples)
        result = dict(name=name, median_ms=median, p95_ms=sorted(samples)[math.ceil(len(samples) * .95) - 1],
                      tokens_per_second=tokens * 1000 / median,
                      peak_allocated_mib=None if peak is None else peak / 2 ** 20,
                      incremental_peak_mib=None if peak is None else (peak - baseline) / 2 ** 20,
                      reset_environments_per_call=resets, sdpa_ops=sdpa)
        print(json.dumps(result), flush=True)
        return result

    def train():
        model.zero_grad(set_to_none=True)
        with amp():
            loss = parallel(x).square().mean()
        loss.backward()

    def prefix():
        with torch.no_grad(), amp():
            parallel(x)[:, -1:]

    results = [measure("training_forward_backward", train, x.shape[0] * x.shape[1]),
               measure("prefix_recompute_no_grad", prefix, args.batch_size)]
    model.zero_grad(set_to_none=True)

    def make_step(*, grad, backward, reset):
        cache = model.create_cache(args.batch_size, dtype=dtype)
        with amp():
            for t in range(args.seq_len - 1):
                model.step(x[:, t:t + 1], cache)

        def callback():
            if grad:
                model.zero_grad(set_to_none=True)
            with amp():
                out = model.step(token, cache, enable_grad=grad,
                                 episode_starts=reset_mask if reset else None)
                if backward:
                    loss = out.square().mean()
            if backward:
                loss.backward()
        return callback, cache.nbytes

    cache_bytes = None
    for name, grad, backward, reset in (
        ("step_no_grad", False, False, False),
        ("step_grad_forward", True, False, False),
        ("step_grad_forward_backward", True, True, False),
        ("step_reset10pct_no_grad", False, False, True),
        ("step_reset10pct_grad_forward_backward", True, True, True),
    ):
        callback, cache_bytes = make_step(grad=grad, backward=backward, reset=reset)
        reset_count = max(1, math.ceil(args.batch_size * .1)) if reset else 0
        results.append(measure(name, callback, args.batch_size, resets=reset_count))
        del callback
        model.zero_grad(set_to_none=True)

    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  torch=torch.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name() if device.type == "cuda" else platform.processor(),
                  dtype=args.dtype, compiled_parallel=args.compile, config=asdict(config),
                  batch_size=args.batch_size, prefix_length=args.seq_len, warmup=args.warmup,
                  iterations=args.iterations, cache_bytes=cache_bytes,
                  timing="wall clock, per-call CUDA synchronization; setup/warmup/profiling excluded",
                  notes="Forward+backward cases include zero_grad but no optimizer. Step history is detached. "
                        "Prefix case recomputes the full prefix with per-layer window masks; prefix length is fixed. "
                        "Reset cases reset the same ceil(10% * B) environment slots before every call.",
                  results=results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
