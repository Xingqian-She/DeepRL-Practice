"""Component math, schedule boundaries, diagnostic isolation and output safety."""

import copy
from dataclasses import asdict, replace
import tempfile
import unittest
from unittest.mock import patch

import torch

from buffer import RolloutBuffer
from fixtures import fixture_vector, small_config
from model import Agent
from ppo import PPOTrainer, ppo_losses
from rollout import Collector
from runtime import Run, configure_runtime, fingerprint


class ComponentTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(3, "cpu")

    def test_schedule_endpoints_single_update_and_validation(self):
        config = small_config(total_timesteps=78, lr_schedule="linear", ent_coef_schedule="linear")
        trainer = PPOTrainer(Agent(config.agent), config)
        for update, fraction in ((1, 1), (2, 0.5), (3, 0)):
            lr, entropy = trainer.schedule(update)
            self.assertAlmostEqual(lr, fraction * config.learning_rate)
            self.assertAlmostEqual(entropy, fraction * config.ent_coef)
            self.assertEqual(lr, trainer.optimizer.param_groups[0]["lr"])
        single = replace(config, total_timesteps=26)
        self.assertEqual(PPOTrainer(Agent(single.agent), single).schedule(1), (single.learning_rate, single.ent_coef))
        constant = replace(config, lr_schedule="constant", ent_coef_schedule="constant")
        self.assertEqual(PPOTrainer(Agent(constant.agent), constant).schedule(3),
                         (constant.learning_rate, constant.ent_coef))
        for changes in ({"learning_rate_final": -1}, {"ent_coef_final": float("nan")},
                        {"value_clip_coef": -0.1}, {"normalize_advantage": 1}, {"diagnostic_interval": -1},
                        {"eval_num_envs": 0}, {"lr_schedule": "bad"}, {"ent_coef_schedule": "bad"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(config, **changes)

    def test_raw_gae_and_normalization_do_not_change_value_targets(self):
        buffer = RolloutBuffer(2, 1, torch.device("cpu"), [], 1)
        buffer.step = 2
        buffer.values[:, 0] = torch.tensor([1., 2.])
        buffer.rewards[:, 0] = torch.tensor([3., 5.])
        buffer.next_values[:, 0] = torch.tensor([2., 0.])
        buffer.terminated[1, 0] = True
        buffer.compute_returns_and_advantages(0.9, 0.5, False)
        torch.testing.assert_close(buffer.raw_advantages[:, 0], torch.tensor([5.15, 3.]))
        torch.testing.assert_close(buffer.advantages, buffer.raw_advantages)
        returns = buffer.returns.clone()
        buffer.compute_returns_and_advantages(0.9, 0.5, True)
        torch.testing.assert_close(buffer.returns, returns, rtol=0, atol=0)
        self.assertAlmostEqual(buffer.training_statistics()["raw_advantage_mean"], 4.075, places=5)
        batch = next(buffer.batches(1, 2, shuffle=False))
        torch.testing.assert_close(batch["old_values"][batch["loss_mask"]], torch.tensor([1., 2.]))
        self.assertEqual(set(batch["sample_ids"][batch["loss_mask"]].tolist()), {0, 1})

    def test_value_clipping_loss_gradient_and_frozen_old_values(self):
        values = torch.tensor([1.5, -0.5], requires_grad=True)
        old = torch.tensor([1., 0.], requires_grad=True)
        _, loss, _ = ppo_losses(torch.ones(2), torch.zeros(2), values, torch.tensor([2., -1.]),
                                torch.zeros(2), clip_coef=.2, vf_coef=.5, ent_coef=.01,
                                old_values=old, value_clip_coef=.2)
        self.assertAlmostEqual(loss.item(), .32, places=6)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.zeros(2))
        self.assertIsNone(old.grad)
        worse = torch.tensor([3.], requires_grad=True)
        _, loss, _ = ppo_losses(torch.ones(1), torch.zeros(1), worse, torch.ones(1), torch.zeros(1),
                                clip_coef=.2, vf_coef=.5, ent_coef=.01,
                                old_values=torch.zeros(1), value_clip_coef=.2)
        loss.backward()
        self.assertEqual(loss.item(), 2)
        self.assertEqual(worse.grad.item(), 2)

    def test_diagnostics_do_not_change_updates_gradients_optimizer_or_rng(self):
        for mode in ("shared", "separate"):
            config = small_config(diagnostic_interval=0)
            config = replace(config, agent=replace(config.agent, backbone_mode=mode))
            env = fixture_vector()
            try:
                agent = Agent(config.agent)
                collector = Collector(env, agent, torch.device("cpu"), config.seed,
                                      sensor_config=config.sensors, total_timesteps=config.effective_total_timesteps)
                buffer = collector.collect(config.num_steps)
                other = copy.deepcopy(agent)
                base = PPOTrainer(agent, config)
                probed = PPOTrainer(other, replace(config, diagnostic_interval=1))
                rng = torch.get_rng_state()
                plain_metrics = base.update(copy.deepcopy(buffer))
                plain_rng = torch.get_rng_state()
                torch.set_rng_state(rng)
                probe_metrics = probed.update(copy.deepcopy(buffer))
                torch.testing.assert_close(torch.get_rng_state(), plain_rng, rtol=0, atol=0)
                for left, right in zip(agent.parameters(), other.parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                    torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
                    for key in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(base.optimizer.state[left][key], probed.optimizer.state[right][key], rtol=0, atol=0)
                for key in plain_metrics:
                    if key != "diagnostic_seconds":
                        self.assertEqual(plain_metrics[key], probe_metrics[key], key)
                self.assertEqual(probe_metrics["unique_sample_coverage"], 1)
                self.assertEqual(probe_metrics["sample_reuse_count"], config.update_epochs)
                self.assertGreater(probed.last_diagnostics["total_parameter_delta_norm"], 0)
                self.assertEqual(probed.last_diagnostics["shared_gradient_cosine"] is None, mode == "separate")
            finally:
                env.close()

    def test_rejected_first_minibatch_reports_zero_coverage(self):
        config = small_config(diagnostic_interval=0)
        env = fixture_vector()
        try:
            agent = Agent(config.agent)
            buffer = Collector(env, agent, torch.device("cpu"), config.seed, sensor_config=config.sensors,
                               total_timesteps=config.effective_total_timesteps).collect(config.num_steps)
            original = agent.evaluate_actions
            def shifted(batch):
                logp, entropy, values = original(batch)
                return logp + 1, entropy, values
            with patch("ppo.check_consistency", return_value={"consistency_warning": 0}), \
                    patch.object(agent, "evaluate_actions", side_effect=shifted):
                metrics = PPOTrainer(agent, config).update(buffer)
            self.assertEqual(metrics["optimizer_steps"], 0)
            self.assertEqual(metrics["unique_sample_coverage"], 0)
            self.assertEqual(metrics["accepted_sample_visits"], 0)
            self.assertEqual(metrics["planned_sample_visits"], config.update_epochs * 26)
        finally:
            env.close()

    def test_exact_directory_provenance_and_csv_field_safety(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Run("train", asdict(small_config()), torch.device("cpu"), run_dir=directory + "/exact", experiment_id="FO01")
            try:
                self.assertEqual(run.path.name, "exact")
                self.assertEqual(run.metadata["experiment_id"], "FO01")
                self.assertEqual(run.metadata["config_fingerprint"], fingerprint(asdict(small_config())))
                self.assertEqual(len(run.metadata["source_fingerprint"]), 64)
                run.append_csv("fixed.csv", {"a": 1})
                with self.assertRaises(ValueError):
                    run.append_csv("fixed.csv", {"b": 1})
                with self.assertRaises(FileExistsError):
                    Run("train", asdict(small_config()), torch.device("cpu"), run_dir=run.path)
                with self.assertRaises(ValueError):
                    Run("train", asdict(small_config()), torch.device("cpu"), directory, run_dir=run.path)
            finally:
                run.close()
