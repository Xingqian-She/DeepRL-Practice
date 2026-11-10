"""Focused diagnostics, KL rejection, evaluation isolation and curriculum tests."""

import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from curriculum import Curriculum
from diagnose_consistency import replay_snapshot, save_snapshot
from evaluate import evaluate_agent
from fixtures import FixtureEnv, fixture_vector, small_config
from model import Agent
from ppo import ConsistencyError, PPOTrainer, check_consistency
from rollout import Collector
from runtime import configure_runtime, load_checkpoint, save_checkpoint
from sensors import FAILURE_INDICES, SensorConfig
from train import make_parser, run_training


class StabilityTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(3, "cpu")

    def collector(self, config=None):
        config = config or small_config(sensors=SensorConfig(mode="full"), eval_interval=0)
        env = fixture_vector()
        self.addCleanup(env.close)
        agent = Agent(config.agent)
        collector = Collector(env, agent, torch.device("cpu"), config.seed,
                              sensor_config=config.sensors, total_timesteps=config.effective_total_timesteps)
        return config, collector

    def test_defaults_validation_and_legacy_checkpoint(self):
        args = make_parser().parse_args([])
        self.assertEqual((args.target_kl, args.consistency_warning_tolerance, args.consistency_tolerance),
                         (0.02, 1e-4, 5e-4))
        self.assertEqual((args.curriculum_schedule, args.eval_interval, args.eval_policy), ("fixed", 100, "sample"))
        config = small_config()
        for changes in ({"target_kl": -1}, {"eval_interval": -1}, {"eval_policy": "bad"},
                        {"consistency_warning_tolerance": 1}, {"curriculum_step": 0},
                        {"curriculum_patience": 0}, {"curriculum_success_rate": 2},
                        {"sensors": SensorConfig(schedule="performance"), "eval_interval": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-v2.pt"
            agent = Agent(config.agent)
            save_checkpoint(path, agent, config, 52)
            saved = torch.load(path, weights_only=True)
            saved.pop("training_state")
            saved["train_config"]["sensors"].pop("schedule")
            for field in ("target_kl", "consistency_warning_tolerance", "eval_interval"):
                saved["train_config"].pop(field)
            torch.save(saved, path)
            loaded, _ = load_checkpoint(path, torch.device("cpu"))
            for name, value in agent.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)

    def test_dual_threshold_boundaries_and_nonfinite_inputs(self):
        config, collector = self.collector()
        buffer = collector.collect(config.num_steps)
        # Binary-exact bounds avoid testing decimal addition rounding.
        config = replace(config, consistency_warning_tolerance=2**-13, consistency_tolerance=2**-11)
        def outputs(delta):
            return lambda b: (b["old_log_probs"].double() + delta,
                              torch.zeros_like(b["old_log_probs"]), torch.zeros_like(b["old_log_probs"]))
        for delta, warning in ((0, 0), (2**-13, 0), (2**-12, 1), (2**-11, 1)):
            with patch.object(collector.agent, "evaluate_actions", side_effect=outputs(delta)):
                self.assertEqual(check_consistency(collector.agent, buffer, config)["consistency_warning"], warning)
        with patch.object(collector.agent, "evaluate_actions", side_effect=outputs(2**-10)):
            with self.assertRaises(ConsistencyError):
                check_consistency(collector.agent, buffer, config)
        with patch.object(collector.agent, "evaluate_actions", side_effect=outputs(0.0)):
            buffer.old_log_probs[0, 0] = float("nan")
            with self.assertRaises(ConsistencyError) as caught:
                check_consistency(collector.agent, buffer, config)
            self.assertEqual(caught.exception.report["consistency_finite"], 0)
        with patch.object(collector.agent, "evaluate_actions", side_effect=ValueError("invalid logits NaN")):
            with self.assertRaises(ConsistencyError):
                check_consistency(collector.agent, buffer, config)

    def test_ratio_guard_can_reject_below_custom_logprob_limit(self):
        config, collector = self.collector()
        buffer = collector.collect(config.num_steps)
        config = replace(config, consistency_tolerance=1.0)
        original = collector.agent.evaluate_actions
        def shifted(batch):
            logp, entropy, value = original(batch)
            return logp + 0.3, entropy, value
        with patch.object(collector.agent, "evaluate_actions", side_effect=shifted):
            with self.assertRaises(ConsistencyError) as caught:
                check_consistency(collector.agent, buffer, config)
            self.assertGreater(caught.exception.report["consistency_clip_fraction"], 0)

    def test_nonfinite_weights_can_be_replayed_as_evidence(self):
        config, collector = self.collector()
        buffer = collector.collect(config.num_steps)
        with torch.no_grad():
            collector.agent.actor.weight.fill_(float("nan"))
        with self.assertRaises(ConsistencyError) as caught:
            check_consistency(collector.agent, buffer, config)
        with tempfile.TemporaryDirectory() as directory:
            path = save_snapshot(Path(directory) / "nonfinite.pt", collector.agent, buffer, config,
                                 update=1, global_step=26, report=caught.exception.report,
                                 environment={}, severity="failure")
            result = replay_snapshot(path)
            self.assertFalse(result["parallel_vs_saved"]["finite"])
            self.assertIsNone(result["parallel_vs_saved"]["max_abs_error"])
            json.dumps(result, allow_nan=False)

    def test_warning_failure_snapshots_precede_updates_and_replay(self):
        config = small_config(total_timesteps=104, eval_interval=0)
        original_collect = Collector.collect
        calls = 0
        def corrupt(collector, steps):
            nonlocal calls
            buffer = original_collect(collector, steps)
            calls += 1
            buffer.old_log_probs[0, 0] += 0.0002 if calls < 3 else 0.01
            return buffer
        with tempfile.TemporaryDirectory() as directory:
            with patch("train.make_vector_env", side_effect=lambda *a: fixture_vector()), \
                    patch.object(Collector, "collect", new=corrupt), self.assertRaises(ConsistencyError):
                run_training(config, directory)
            run = next(Path(directory).iterdir())
            metadata = json.loads((run / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "failed")
            self.assertEqual(metadata["last_completed_step"], 52)
            self.assertEqual(metadata["failed_sampled_step"], 78)
            self.assertEqual(metadata["consistency_warning_count"], 2)
            snapshots = list((run / "diagnostics").glob("*.pt"))
            self.assertEqual(len(snapshots), 2)
            failure = torch.load(run / metadata["failure_snapshot"], weights_only=True)
            latest = torch.load(run / "checkpoints/latest.pt", weights_only=True)
            self.assertEqual(failure["parameter_version"], 2)
            self.assertEqual(failure["buffer"]["step"], config.num_steps)
            for key, value in latest["state_dict"].items():
                torch.testing.assert_close(value, failure["state_dict"][key], rtol=0, atol=0)
            result = replay_snapshot(run / metadata["warning_snapshot"], precision="fp64")
            self.assertEqual(result["position_mismatches"], 0)
            self.assertGreater(result["sampling_logits_vs_saved"]["max_abs_error"], 0.00019)
            self.assertLess(result["parallel_vs_serial"]["max_abs_error"], 1e-10)
            warning = torch.load(run / metadata["warning_snapshot"], weights_only=True)
            configure_runtime(config.seed, "cpu")
            initial = Agent(config.agent)
            for key, value in initial.state_dict().items():
                torch.testing.assert_close(value, warning["state_dict"][key], rtol=0, atol=0)

    def test_kl_rejected_minibatch_does_not_update_parameters(self):
        for target_kl in (0.02, 0.0):
            config, collector = self.collector(small_config(sensors=SensorConfig(mode="full"), target_kl=target_kl))
            buffer = collector.collect(config.num_steps)
            trainer = PPOTrainer(collector.agent, config)
            original = collector.agent.evaluate_actions
            calls, rejected_state = 0, None
            def outputs(batch):
                nonlocal calls, rejected_state
                calls += 1
                result = original(batch)
                if calls == 2:
                    rejected_state = copy.deepcopy(collector.agent.state_dict())
                return (result[0] + (1.0 if calls >= 2 else 0.0), result[1], result[2])
            with patch.object(collector.agent, "evaluate_actions", side_effect=outputs), \
                    patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as step:
                with patch("ppo.check_consistency", return_value={"consistency_warning": 0}):
                    metrics = trainer.update(buffer)
            self.assertEqual(metrics["optimizer_steps"], step.call_count)
            if target_kl:
                self.assertEqual(metrics["optimizer_steps"], 1)
                self.assertEqual(metrics["ppo_early_stop"], 1)
                for key, value in collector.agent.state_dict().items():
                    torch.testing.assert_close(value, rejected_state[key], rtol=0, atol=0)
            else:
                self.assertEqual(metrics["ppo_early_stop"], 0)
                self.assertGreater(step.call_count, 1)

    def test_behavior_metrics_and_evaluation_rng_isolation(self):
        config, collector = self.collector()
        buffer = collector.collect(config.num_steps)
        metrics = buffer.behavior_metrics()
        self.assertAlmostEqual(sum(metrics[f"action_fraction_{i}"] for i in range(4)), 1.0, places=6)
        expected = torch.distributions.Categorical(logits=buffer.sampling_logits)
        self.assertAlmostEqual(metrics["main_engine_probability"], expected.probs[..., 2].mean().item())
        collector.agent.train()
        state = copy.deepcopy(collector.agent.state_dict())
        cache = {key: getattr(collector.cache, key).clone() for key in ("keys", "values", "next_positions")}
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
        first = evaluate_agent(collector.agent, episodes=2, seed=10000, sensor_failure_probability=0.5,
                               env_factory=lambda *a: FixtureEnv(limit=5))
        second = evaluate_agent(collector.agent, episodes=2, seed=10000, sensor_failure_probability=0.5,
                                env_factory=lambda *a: FixtureEnv(limit=5))
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
        torch.testing.assert_close(torch.get_rng_state(), rng[2], rtol=0, atol=0)
        self.assertTrue(collector.agent.training)
        for key, value in collector.agent.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        for key, value in cache.items():
            torch.testing.assert_close(getattr(collector.cache, key), value, rtol=0, atol=0)
        self.assertEqual(first["score_200_rate"], 0)
        self.assertAlmostEqual(sum(first[f"action_fraction_{i}"] for i in range(4)), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_evaluation_rng_isolation(self):
        configure_runtime(9, "cuda")
        agent = Agent(small_config().agent).cuda().train()
        # Periodic evaluation must not use the training compiled entry point.
        agent._compiled_step = lambda *a: (_ for _ in ()).throw(AssertionError("training step used"))
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        result = evaluate_agent(agent, episodes=2, env_factory=lambda *a: FixtureEnv(limit=5))
        torch.testing.assert_close(cpu_rng, torch.get_rng_state(), rtol=0, atol=0)
        for original, current in zip(cuda_rng, torch.cuda.get_rng_state_all()):
            torch.testing.assert_close(original, current, rtol=0, atol=0)
        self.assertEqual(result["mean_length"], 5)
        self.assertTrue(agent.training)

class CurriculumTests(unittest.TestCase):
    def performance_config(self, **kwargs):
        return small_config(sensors=SensorConfig(schedule="performance"), eval_interval=1, **kwargs)

    def test_patience_failure_reset_nondividing_stages_and_final_goal(self):
        curriculum = Curriculum(self.performance_config(curriculum_step=0.6))
        self.assertEqual(curriculum.levels, [0, 0.6, 1])
        def result(mean=220, rate=0.9):
            return {"sensor_failure_probability": curriculum.failure_probability(0, 52),
                    "mean_return": mean, "score_200_rate": rate}
        self.assertIsNone(curriculum.observe(result(), 10))
        self.assertIsNone(curriculum.observe(result(rate=0.7), 20))
        self.assertEqual(curriculum.streak, 0)
        self.assertIsNone(curriculum.observe(result(), 30))
        self.assertEqual(curriculum.observe(result(), 40)["to_probability"], 0.6)
        curriculum.record_steps(26, 66)
        self.assertIsNone(curriculum.observe(result(), 70))
        self.assertEqual(curriculum.observe(result(), 80)["to_probability"], 1)
        self.assertTrue(curriculum.reached_final)
        self.assertFalse(curriculum.final_task_solved)
        self.assertIsNone(curriculum.observe(result(), 90))
        curriculum.observe(result(), 100)
        self.assertTrue(curriculum.final_task_solved)
        self.assertEqual(curriculum.stage_steps, [0, 26, 0])

    def test_transition_refreshes_only_pending_observation(self):
        config = self.performance_config(curriculum_step=1, curriculum_patience=1)
        configure_runtime(3, "cpu")
        env = fixture_vector((100, 100))
        try:
            schedule = Curriculum(config)
            collector = Collector(env, Agent(config.agent), torch.device("cpu"), 3,
                                  sensor_config=schedule, total_timesteps=52)
            buffer = collector.collect(13)
            history = collector.history.snapshot()
            old = buffer.observations.clone()
            previous = collector.previous_actions.clone()
            schedule.observe({"sensor_failure_probability": 0, "mean_return": 220, "score_200_rate": 1}, 26)
            collector.refresh_pending_observation()
            self.assertTrue((collector.observations[:, 8 + np.array(FAILURE_INDICES)] == 0).all())
            torch.testing.assert_close(buffer.observations, old, rtol=0, atol=0)
            torch.testing.assert_close(collector.previous_actions, previous, rtol=0, atol=0)
            for before, after in zip(history, collector.history.tokens):
                torch.testing.assert_close(before.observations, after.observations, rtol=0, atol=0)
            check_consistency(collector.agent, collector.collect(13), config)
        finally:
            env.close()

    def test_training_stage_checkpoints_and_budget_without_qualification(self):
        for qualified in (False, True):
            config = self.performance_config(total_timesteps=130, curriculum_step=1, eval_episodes=2)
            seen = []
            def evaluation(agent, **kwargs):
                probability = kwargs["sensor_failure_probability"]
                seen.append(probability)
                return {"sensor_failure_probability": probability, "policy": "sample", "episodes": [],
                        "mean_return": 220 if qualified else -110, "score_200_rate": 1 if qualified else 0,
                        "std_return": 1, "mean_length": 69}
            with tempfile.TemporaryDirectory() as directory:
                with patch("train.make_vector_env", side_effect=lambda *a: fixture_vector()), \
                        patch("train.evaluate_agent", side_effect=evaluation):
                    run = run_training(config, directory)
                metadata = json.loads((run / "metadata.json").read_text())
                self.assertTrue(metadata["training_budget_completed"])
                self.assertEqual(metadata["final_task_solved"], qualified)
                self.assertEqual(metadata["entered_final"], qualified)
                self.assertEqual(seen, [0, 0, 1, 1, 1] if qualified else [0] * 5)
                checkpoints = [p.name for p in (run / "checkpoints").glob("*.pt")]
                self.assertIn("best_stage_00_p0.00.pt", checkpoints)
                self.assertEqual("transition_stage_00.pt" in checkpoints, qualified)
                self.assertEqual("best_stage_01_p1.00.pt" in checkpoints, qualified)
                with (run / "evaluation.csv").open() as file:
                    self.assertEqual(len(list(csv.DictReader(file))), 5)
                saved = torch.load(run / "checkpoints/latest.pt", weights_only=True)
                self.assertEqual(sum(saved["training_state"]["curriculum"]["stage_steps"]), 130)


if __name__ == "__main__":
    unittest.main()
