"""Encoder/decoder integration with real sequence PPO and policy-only evaluation."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from acceleration import _sequence_inputs
from config import AgentConfig
from diagnose_consistency import replay_snapshot, save_snapshot
from evaluation_runner import PolicySnapshot
from evaluate import run_evaluation
from fixtures import fixture_vector, small_config
from model import Agent, cache_tensors
from ppo import PPOTrainer, check_consistency
from rollout import Collector
from run_experiments import PROJECT, build_plan, dependency_lock_hash
from runtime import configure_runtime, load_checkpoint, save_checkpoint
from sensors import SensorConfig
from train import make_parser, run_training


def config_for_test(**changes):
    config = small_config(compile_mode="off", **changes)
    return replace(config, agent=replace(config.agent, backbone_mode="encoder_decoder", decoder_layers=2))


class EncoderDecoderAgentTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(3, "cpu")
        self.config = config_for_test()

    def test_configuration_independent_projections_and_parameter_groups(self):
        config = AgentConfig(backbone_mode="encoder_decoder")
        self.assertEqual((config.warmup_length, config.value_warmup_length), (249, 125))
        self.assertEqual(replace(config, decoder_layers=1).warmup_length, 156)
        self.assertEqual(make_parser().parse_args(["--backbone-mode", "encoder_decoder"]).decoder_layers, 4)
        agent = Agent(self.config.agent)
        self.assertNotEqual(agent.encoder[0].weight.data_ptr(), agent.decoder_projection[0].weight.data_ptr())
        policy = {id(p) for p in agent.policy_parameters()}
        value = {id(p) for p in agent.value_parameters()}
        encoder = {id(p) for p in agent.encoder.parameters()} | {id(p) for p in agent.backbone.encoder.parameters()}
        self.assertEqual(policy & value, encoder)
        self.assertEqual(policy | value, {id(p) for p in agent.parameters()})
        self.assertEqual(agent.parameter_counts()["shared"], sum(p.numel() for p in agent.parameters() if id(p) in encoder))

    def test_actor_history_gradient_value_isolation_and_burnin_graph(self):
        agent = Agent(self.config.agent)
        features, starts, valid, positions = _sequence_inputs(self.config, 2, 15, torch.device("cpu"))
        features.requires_grad_()
        logits, values = agent._sequence_core(features, starts, valid, positions)
        # Only a final loss token; earlier burn-in features must remain connected.
        loss = logits[0, -1].square().sum()
        policy_gradients = torch.autograd.grad(loss, list(agent.parameters()), allow_unused=True, retain_graph=True)
        for (name, _), gradient in zip(agent.named_parameters(), policy_gradients):
            if name.startswith("encoder.") or name.startswith("backbone.encoder."):
                self.assertIsNotNone(gradient, name)
                self.assertGreater(gradient.norm().item(), 0, name)
        input_gradient, = torch.autograd.grad(loss, features, retain_graph=True)
        self.assertGreater(input_gradient[0, :-1].norm().item(), 0)
        value_gradients = torch.autograd.grad(values[0, -1], list(agent.parameters()), allow_unused=True)
        for (name, _), gradient in zip(agent.named_parameters(), value_gradients):
            if name.startswith(("decoder_projection.", "backbone.decoder.", "actor.")):
                self.assertIsNone(gradient, name)

    def test_step_sequence_policy_snapshot_and_all_cache_tensors_match(self):
        agent = Agent(self.config.agent)
        snapshot = PolicySnapshot(agent)
        self.assertFalse(any(name.startswith(("critic.", "value_")) for name in snapshot.state_dict()))
        features, starts, valid, positions = _sequence_inputs(self.config, 3, 29, torch.device("cpu"))
        valid[:] = True
        expected = agent._sequence_core(features, starts, valid, positions)
        cache, policy_cache = agent.create_cache(3), snapshot.backbone.create_cache(3)
        logits, values = [], []
        with torch.no_grad():
            for t in range(29):
                actual, value = agent._step_core(features[:, t], cache, starts[:, t])
                torch.testing.assert_close(snapshot.step(features[:, t], policy_cache, starts[:, t]), actual)
                logits.append(actual)
                values.append(value)
        for actual, reference in zip((torch.stack(logits, 1), torch.stack(values, 1)), expected):
            torch.testing.assert_close(actual, reference, rtol=2e-5, atol=3e-6)
        for name, tensor in cache_tensors(cache).items():
            torch.testing.assert_close(cache_tensors(policy_cache)[name], tensor)
        with torch.no_grad():
            agent.decoder_projection[0].weight.add_(0.1)
        snapshot.sync(agent)
        torch.testing.assert_close(snapshot.decoder_projection[0].weight, agent.decoder_projection[0].weight)

    def test_two_ppo_updates_rebuild_and_readonly_value_peek(self):
        envs = fixture_vector((100, 100))
        agent = Agent(self.config.agent)
        collector = Collector(envs, agent, torch.device("cpu"), self.config.seed,
                              sensor_config=self.config.sensors, total_timesteps=self.config.effective_total_timesteps)
        trainer = PPOTrainer(agent, self.config)
        try:
            with tempfile.TemporaryDirectory() as directory:
                for update in range(1, 3):
                    buffer = collector.collect(self.config.num_steps)
                    old_probabilities = buffer.old_log_probs.clone()
                    report = check_consistency(agent, buffer, self.config)
                    self.assertLess(report["consistency_max_logprob_error"], 1e-4)
                    before = {name: value.clone() for name, value in cache_tensors(collector.cache).items()}
                    with patch.object(agent.actor, "forward", side_effect=AssertionError("actor used by value peek")), \
                         patch.object(agent.backbone.decoder[0], "forward", side_effect=AssertionError("decoder used by value peek")):
                        peek = collector.peek_values(collector.observations, collector.previous_actions, [0, 1])
                    self.assertTrue(peek.isfinite().all())
                    for name, value in cache_tensors(collector.cache).items():
                        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
                    snapshot = Path(directory) / f"update{update}.pt"
                    save_snapshot(snapshot, agent, buffer, self.config, update=update,
                                  global_step=collector.global_step, report=report, environment={}, severity="test")
                    replay = replay_snapshot(snapshot, precision="fp64")
                    self.assertEqual(replay["position_mismatches"], 0)
                    self.assertLess(replay["parallel_vs_serial"]["max_abs_error"], 1e-10)
                    metrics = trainer.update(buffer)
                    self.assertGreater(metrics["optimizer_steps"], 0)
                    self.assertGreater(trainer.last_diagnostics["cross_attention_policy_grad_norm"], 0)
                    self.assertGreater(trainer.last_diagnostics["encoder_policy_grad_norm"], 0)
                    self.assertGreater(trainer.last_diagnostics["encoder_value_grad_norm"], 0)
                    torch.testing.assert_close(buffer.old_log_probs, old_probabilities, rtol=0, atol=0)
                    collector.invalidate_cache()
        finally:
            envs.close()

    def test_async_resets_truncation_and_bootstrap_match_sequence_values(self):
        envs = fixture_vector((3, 5), (True, False))
        agent = Agent(self.config.agent)
        collector = Collector(envs, agent, torch.device("cpu"), self.config.seed,
                              sensor_config=self.config.sensors, total_timesteps=self.config.effective_total_timesteps)
        try:
            buffer = collector.collect(self.config.num_steps)
            self.assertTrue(buffer.terminated[:, 0].any())
            self.assertTrue(buffer.truncated[:, 1].any())
            self.assertEqual(buffer.next_values[buffer.terminated].count_nonzero().item(), 0)
            self.assertTrue((buffer.next_values[buffer.truncated].abs() > 0).all())
            check_consistency(agent, buffer, self.config)
            for batch in buffer.batches(4, 3, shuffle=False):
                _, full = agent.forward_sequence(batch["observations"], batch["previous_actions"],
                                                  episode_starts=batch["episode_starts"], valid_mask=batch["valid_mask"],
                                                  position_ids=batch["position_ids"])
                value = agent.values_sequence(batch["observations"], batch["previous_actions"],
                                               episode_starts=batch["episode_starts"], valid_mask=batch["valid_mask"],
                                               position_ids=batch["position_ids"])
                torch.testing.assert_close(value, full, rtol=0, atol=0)
        finally:
            envs.close()

    def test_v2_checkpoint_roundtrip_and_legacy_missing_decoder_field(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("shared", "separate", "encoder_decoder"):
                config = replace(self.config, agent=replace(self.config.agent, backbone_mode=mode))
                agent = Agent(config.agent)
                path = Path(directory) / f"{mode}.pt"
                save_checkpoint(path, agent, config, 26)
                if mode != "encoder_decoder":
                    saved = torch.load(path, weights_only=True)
                    saved["agent_config"].pop("decoder_layers")
                    torch.save(saved, path)
                loaded, saved = load_checkpoint(path, torch.device("cpu"))
                self.assertEqual(saved["format_version"], 2)
                for name, value in agent.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)

    def test_experiment_contract_and_smoke_override(self):
        path = PROJECT / "encoder_decoder_experiments.json"
        plan = build_plan(path)
        self.assertEqual(plan["groups"], ["ED_PO01", "ED_PO03"])
        self.assertEqual(sum(task["kind"] == "train" for task in plan["tasks"]), 6)
        self.assertEqual(sum(task["kind"] == "evaluate" for task in plan["tasks"]), 36)
        fo04 = build_plan(PROJECT / "experiments.json", ["FO04"])["tasks"][0]["parameters"]
        for task in plan["tasks"]:
            if task["kind"] != "train":
                continue
            config = deepcopy(task["parameters"])
            self.assertEqual(config["agent"]["backbone_mode"], "encoder_decoder")
            config["agent"]["backbone_mode"] = "separate"
            config["sensors"], config["seed"] = fo04["sensors"], fo04["seed"]
            self.assertEqual(config, fo04)
        for task in build_plan(path, smoke=True)["tasks"]:
            if task["kind"] == "train":
                config = task["parameters"]
                self.assertEqual(config["agent"]["decoder_layers"], 2)
                self.assertEqual(config["total_timesteps"], 64)
                self.assertEqual(config["max_episode_steps"], 19)

    def test_dependency_lock_accepts_line_endings_and_rejects_content_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            linux, windows = Path(directory) / "linux.lock", Path(directory) / "windows.lock"
            linux.write_bytes(b"torch==2.9.1\ngymnasium==1.2.2\n")
            windows.write_bytes(b"torch==2.9.1\r\ngymnasium==1.2.2\r\n")
            self.assertEqual(dependency_lock_hash(linux), dependency_lock_hash(windows))
            windows.write_bytes(b"torch==2.9.0\r\ngymnasium==1.2.2\r\n")
            self.assertNotEqual(dependency_lock_hash(linux), dependency_lock_hash(windows))

    def test_real_box2d_cpu_training_and_independent_evaluation(self):
        for mode, schedule in (("partial", "fixed"), ("curriculum", "performance")):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                config = config_for_test(total_timesteps=64, num_steps=16, sequence_length=6,
                                         max_episode_steps=19, eval_interval=1, eval_episodes=2, eval_num_envs=2,
                                         sensors=SensorConfig(mode=mode, schedule=schedule))
                run = run_training(config, directory, experiment_id="ED_SMOKE")
                metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["status"], "completed")
                self.assertEqual(metadata["completed_updates"], 2)
                self.assertGreater(metadata["model"]["rollout_cache_bytes"], 0)
                self.assertEqual(metadata["p1_training_steps"], 64 if mode == "partial" else 0)
                self.assertTrue((run / "diagnostics.csv").is_file())
                result = run_evaluation(run / "checkpoints/latest.pt", episodes=3, num_envs=2,
                                        seed=20000, device="cpu", policy="sample", sensor_failure_probability=0.5,
                                        output_dir=directory, experiment_id="ED_SMOKE")
                evaluation = json.loads((result / "evaluation.json").read_text(encoding="utf-8"))
                self.assertEqual(len(evaluation["episodes"]), 3)
                self.assertEqual(evaluation["checkpoint_global_step"], 64)


if __name__ == "__main__":
    unittest.main()
