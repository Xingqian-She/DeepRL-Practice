"""Pinned-container production-shape smoke, never a long convergence experiment.

python -B tests/smoke_encoder_decoder_cuda.py --output-dir runs/ed-validation/cuda-production
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
from model import Agent
from run_experiments import PROJECT, build_plan
from runtime import configure_runtime, load_checkpoint, write_json
from sensors import SensorConfig
from train import run_training


def smoke(output):
    output = Path(output).resolve()
    recipe = dict(build_plan(PROJECT / "encoder_decoder_experiments.json", ["ED_PO01"])["tasks"][0]["parameters"])
    recipe["agent"] = AgentConfig(**recipe["agent"])
    recipe["sensors"] = SensorConfig(**recipe["sensors"])
    config = replace(TrainConfig(**recipe), total_timesteps=4096, compile_mode="on", save_interval=1,
                     eval_interval=1, eval_episodes=2, eval_num_envs=2, diagnostic_interval=1)
    device = configure_runtime(config.seed, config.device)
    initial = {name: value.detach().cpu().clone() for name, value in Agent(config.agent).state_dict().items()}
    lengths = set()
    collate = RolloutBuffer.collate

    def record_collate(buffer, specs):
        batch = collate(buffer, specs)
        lengths.add(batch["observations"].shape[1])
        return batch

    with patch.object(RolloutBuffer, "collate", record_collate):
        run = run_training(config, run_dir=output / "run", experiment_id="ED_CUDA_SHAPE_SMOKE")
    agent, saved = load_checkpoint(run / "checkpoints/latest.pt", torch.device("cpu"))
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["acceleration"]["enabled"], metadata["acceleration"]
    assert metadata["completed_updates"] == 2 and metadata["global_step"] == 4096
    assert 313 in lengths, sorted(lengths)
    groups = {"encoder_projection": lambda name: name.startswith("encoder."),
              "encoder_stack": lambda name: name.startswith("backbone.encoder."),
              "decoder_projection": lambda name: name.startswith("decoder_projection."),
              "decoder_stack": lambda name: name.startswith("backbone.decoder."),
              "cross_attention": lambda name: ".cross_attention." in name,
              "actor": lambda name: name.startswith("actor."),
              "critic": lambda name: name.startswith("critic.")}
    changed = {group: any(not torch.equal(initial[name], value) for name, value in agent.state_dict().items()
                          if predicate(name)) for group, predicate in groups.items()}
    assert all(changed.values()), changed
    with (run / "training.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert max(float(row["consistency_max_logprob_error"]) for row in rows) < config.consistency_tolerance
    evaluations = []
    for policy, probability in (("sample", 1.0), ("argmax", 0.5)):
        evaluations.append(str(run_evaluation(run / "checkpoints/latest.pt", episodes=3, num_envs=2,
                                              seed=20000, device=str(device), policy=policy,
                                              sensor_failure_probability=probability,
                                              run_dir=output / f"heldout-{policy}",
                                              experiment_id="ED_CUDA_SHAPE_SMOKE")))
    report = {"status": "passed", "run": str(run), "global_step": saved["global_step"],
              "model": metadata["model"], "acceleration": metadata["acceleration"],
              "dependencies": metadata["dependencies"], "hardware": metadata["hardware"],
              "source_fingerprint": metadata["source_fingerprint"],
              "observed_training_sequence_lengths": sorted(lengths), "parameter_groups_changed": changed,
              "max_logprob_error": max(float(row["consistency_max_logprob_error"]) for row in rows),
              "evaluations": evaluations}
    write_json(output / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    smoke(parser.parse_args().output_dir)
