"""Short real-Box2D PPO validation for both XY recipes, including compiled CUDA.

python -B tests/smoke_xy_fourier.py --device cpu --output-dir runs/xy-validation/cpu
python -B tests/smoke_xy_fourier.py --device cuda --production-shape --output-dir runs/xy-validation/cuda
"""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buffer import RolloutBuffer
from config import AgentConfig, TrainConfig
from evaluate import run_evaluation
from run_experiments import PROJECT, build_plan
from runtime import configure_runtime, load_checkpoint, write_json
from sensors import SensorConfig
from train import run_training


def smoke(output, device, production_shape=False):
    output = Path(output).resolve()
    plan = build_plan(PROJECT / "xy_fourier_experiments.json", smoke=not production_shape, device=device)
    reports = []
    for group in plan["groups"]:
        recipe = next(t["parameters"] for t in plan["tasks"] if t["kind"] == "train" and t["group"] == group)
        config = TrainConfig(**{**recipe, "agent": AgentConfig(**recipe["agent"]),
                                "sensors": SensorConfig(**recipe["sensors"])})
        if production_shape:
            config = replace(config, total_timesteps=4096, eval_interval=1, eval_episodes=2,
                             eval_num_envs=2, save_interval=1, diagnostic_interval=1)
        config = replace(config, compile_mode="on" if device == "cuda" else "off")
        runtime_device = configure_runtime(config.seed, device)
        lengths = set()
        collate = RolloutBuffer.collate

        def record_collate(buffer, specs):
            batch = collate(buffer, specs)
            lengths.add(batch["observations"].shape[1])
            return batch

        with patch.object(RolloutBuffer, "collate", record_collate):
            run = run_training(config, run_dir=output / group / "run", experiment_id=group)
        agent, saved = load_checkpoint(run / "checkpoints/latest.pt", torch.device("cpu"))
        metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["status"] == "completed"
        assert saved["input_spec"]["input_dim"] == 32
        assert agent.config.xy_fourier and agent.config.backbone_mode == "separate"
        assert metadata["acceleration"]["enabled"] == (device == "cuda"), metadata["acceleration"]
        if production_shape:
            assert 189 in lengths, lengths
            assert metadata["model"]["parameters"]["total"] == 6444821
        rows = list(csv.DictReader((run / "training.csv").open(encoding="utf-8")))
        assert len(rows) == 2
        assert all(int(row["xy_phase_transitions"]) == config.num_envs * config.num_steps for row in rows)
        assert all(int(row["xy_phase_skipped_terminal_transitions"]) == 0 for row in rows)
        max_error = max(float(r["consistency_max_logprob_error"]) for r in rows)
        assert max_error < config.consistency_tolerance
        assert any(float(r["xy_x_phase_step_max_over_pi"]) > 0 for r in rows)
        evaluations = []
        for policy, probability in (("sample", 1.0), ("argmax", .5)):
            path = run_evaluation(run / "checkpoints/latest.pt", episodes=2, num_envs=2,
                                  seed=32000, device=device, policy=policy,
                                  sensor_failure_probability=probability,
                                  run_dir=output / group / f"evaluation-{policy}", experiment_id=group)
            result = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
            assert result["xy_phase_transitions"] == sum(e["length"] for e in result["episodes"])
            assert result["parameters"]["input_spec"]["input_dim"] == 32
            evaluations.append(str(path))
        reports.append({"group": group, "global_step": saved["global_step"], "model": metadata["model"],
                        "acceleration": metadata["acceleration"], "max_logprob_error": max_error,
                        "sequence_lengths": sorted(lengths), "evaluations": evaluations})
    report = {"status": "passed", "device": str(runtime_device), "production_shape": production_shape,
              "source_fingerprint": metadata["source_fingerprint"], "groups": reports}
    write_json(output / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--production-shape", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    smoke(args.output_dir, args.device, args.production_shape)
