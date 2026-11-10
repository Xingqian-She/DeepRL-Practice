"""真实策略的采样、历史、缓存、边界与概率一致性。"""

import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from model import Agent
from ppo import PPOTrainer, check_consistency
from rollout import Collector
from runtime import configure_runtime
from sensors import PROTECTED_INDICES, SensorConfig, SensorProcessor
from fixtures import fixture_vector, small_config


def make_collector(envs, config):
    device = configure_runtime(config.seed, config.device)
    agent = Agent(config.agent).to(device)
    return Collector(envs, agent, device, config.seed, sensor_config=config.sensors,
                     total_timesteps=config.effective_total_timesteps)


class RolloutTests(unittest.TestCase):
    def test_async_boundaries_alignment_final_observation_and_sequence_coverage(self):
        config = small_config()
        envs = fixture_vector()
        try:
            collector = make_collector(envs, config)
            with patch.object(collector, "peek_values", wraps=collector.peek_values) as peek:
                buffer = collector.collect(config.num_steps)
            self.assertEqual(collector.global_step, 26)
            self.assertEqual(buffer.observations.shape, (13, 2, 16))
            for t in range(config.num_steps):
                for env in range(config.num_envs):
                    if buffer.episode_starts[t, env]:
                        self.assertEqual(buffer.previous_actions[t, env].item(), -1)
                        self.assertEqual(buffer.position_ids[t, env].item(), 0)
                    elif t:
                        self.assertEqual(buffer.previous_actions[t, env], buffer.actions[t - 1, env])
            self.assertTrue((buffer.next_values[buffer.terminated] == 0).all())
            self.assertEqual(len(buffer.episodes), 6)
            timeout_calls = [call for call in peek.call_args_list if call.args[0][:, 1].eq(5).any()]
            self.assertEqual(len(timeout_calls), 2)
            for t, call in zip((4, 9), timeout_calls):
                final, actions, indices = call.args
                self.assertEqual(final.shape, (1, 16))
                torch.testing.assert_close(final[:, 8 + np.array(PROTECTED_INDICES)], torch.ones(1, 4))
                torch.testing.assert_close(actions, buffer.actions[t, 1:2])
                self.assertEqual(indices.item(), 1)
                self.assertEqual(buffer.observations[t + 1, 1, 1].item(), 0)
                self.assertGreater(abs(buffer.next_values[t, 1].item() - buffer.values[t + 1, 1].item()), 1e-6)
            for t in range(config.num_steps - 1):
                continuing = ~(buffer.terminated[t] | buffer.truncated[t])
                torch.testing.assert_close(buffer.next_values[t, continuing], buffer.values[t + 1, continuing])
            report = check_consistency(collector.agent, buffer, config)
            self.assertLess(report["consistency_max_logprob_error"], 1e-4)
            self.assertEqual(report["consistency_clip_fraction"], 0)
            self.assertEqual(sum(int(b["loss_mask"].sum()) for b in buffer.batches(4, 3, shuffle=False)), 26)
            availability = buffer.observations[..., 8:]
            self.assertTrue((buffer.observations[..., :8][availability == 0] == 0).all())
        finally:
            envs.close()

    def test_frozen_masks_bootstrap_and_cross_update_history(self):
        config = small_config()
        envs = fixture_vector((100, 100))
        try:
            collector = make_collector(envs, config)
            first = collector.collect(13)
            pending_obs = collector.observations.clone()
            pending_action = collector.previous_actions.clone()
            old_observations = first.observations.clone()
            snapshot = collector.history.snapshot()
            rng = copy.deepcopy(collector.sensor_processor.rng.bit_generator.state)
            with patch.object(collector.sensor_processor, "process", wraps=collector.sensor_processor.process) as process:
                PPOTrainer(collector.agent, config).update(first)
                collector.history.rebuild(collector.agent, 2)
                self.assertEqual(process.call_count, 0)
            self.assertEqual(rng, collector.sensor_processor.rng.bit_generator.state)
            for old, rebuilt in zip(snapshot, collector.history.tokens):
                torch.testing.assert_close(old.observations, rebuilt.observations, rtol=0, atol=0)
            collector.invalidate_cache()
            second = collector.collect(13)
            torch.testing.assert_close(second.observations[0], pending_obs, rtol=0, atol=0)
            torch.testing.assert_close(second.previous_actions[0], pending_action)
            torch.testing.assert_close(first.observations, old_observations, rtol=0, atol=0)
            self.assertEqual(len(second.prefix), config.agent.warmup_length)
            self.assertEqual(second.sensor_probability_end, 1)
            check_consistency(collector.agent, second, config)
        finally:
            envs.close()

    def test_successor_peek_preserves_cache_and_sensor_rng(self):
        config = small_config()
        envs = fixture_vector((100, 100))
        try:
            collector = make_collector(envs, config)
            collector.collect(8)
            before = {name: getattr(collector.cache, name).clone()
                      for name in ("keys", "values", "lengths", "next_positions")}
            rng = copy.deepcopy(collector.sensor_processor.rng.bit_generator.state)
            collector.peek_values(collector.observations, collector.previous_actions, [0, 1])
            for name, expected in before.items():
                torch.testing.assert_close(getattr(collector.cache, name), expected, rtol=0, atol=0)
            self.assertEqual(rng, collector.sensor_processor.rng.bit_generator.state)
        finally:
            envs.close()

    def test_both_done_flags_prioritize_termination(self):
        envs = fixture_vector((2, 2), (True, True), both=True)
        try:
            collector = make_collector(envs, small_config())
            with patch.object(collector, "peek_values", wraps=collector.peek_values) as peek:
                buffer = collector.collect(4)
            self.assertTrue((buffer.next_values[buffer.terminated] == 0).all())
            self.assertEqual(peek.call_count, 0)
        finally:
            envs.close()

    def test_missing_final_observation_is_rejected(self):
        envs = fixture_vector((2, 2), (False, False))
        try:
            original_step = envs.step

            def without_final(actions):
                obs, rewards, term, trunc, infos = original_step(actions)
                infos.pop("final_obs", None)
                return obs, rewards, term, trunc, infos

            collector = make_collector(envs, small_config())
            with patch.object(envs, "step", side_effect=without_final), self.assertRaisesRegex(RuntimeError, "final_obs"):
                collector.collect(4)
        finally:
            envs.close()

    def test_future_episode_padding_isolation_and_history_gradients(self):
        configure_runtime(3, "cpu")
        agent = Agent(small_config().agent)
        raw = np.random.default_rng(3).normal(size=(2, 12, 8))
        obs = torch.tensor(SensorProcessor(3).process(raw, 0.5), requires_grad=True)
        prev = torch.zeros(2, 12, dtype=torch.long)
        starts = torch.zeros(2, 12, dtype=torch.bool)
        starts[:, 0], starts[0, 7] = True, True
        valid = torch.ones(2, 12, dtype=torch.bool)
        valid[1, 10:] = False
        positions = torch.arange(12).expand(2, 12)
        kwargs = dict(episode_starts=starts, valid_mask=valid, position_ids=positions)
        original, values = agent.forward_sequence(obs, prev, **kwargs)
        altered = obs.detach().clone()
        altered[0, :7] += 100
        altered[1, 10:] += 100
        changed, _ = agent.forward_sequence(altered, prev, **kwargs)
        torch.testing.assert_close(original[0, 7:], changed[0, 7:])
        torch.testing.assert_close(original[1, :10], changed[1, :10])
        future = obs.detach().clone()
        future[:, 9:] += 100
        changed_future, _ = agent.forward_sequence(future, prev, **kwargs)
        torch.testing.assert_close(original[:, :9], changed_future[:, :9])
        values[1, 8].backward()
        self.assertGreater(obs.grad[1, 5:8].abs().sum().item(), 0)
        self.assertEqual(obs.grad[1, 9:].abs().sum().item(), 0)

    def test_preupdate_check_rejects_wrong_old_probability(self):
        config = small_config()
        envs = fixture_vector()
        try:
            collector = make_collector(envs, config)
            buffer = collector.collect(config.num_steps)
            buffer.old_log_probs[0, 0] += 0.01
            before = buffer.old_log_probs.clone()
            with self.assertRaisesRegex(RuntimeError, "概率不一致"):
                check_consistency(collector.agent, buffer, config)
            torch.testing.assert_close(before, buffer.old_log_probs, rtol=0, atol=0)
        finally:
            envs.close()

    @unittest.skipUnless(torch.cuda.is_available(), "需要 GPU 容器")
    def test_cuda_stream_parallel_update_and_rebuild_both_action_modes(self):
        for enabled in (True, False):
            config = small_config(device="cuda")
            config = replace(config, agent=replace(config.agent, use_previous_action=enabled))
            envs = fixture_vector((100, 100))
            try:
                collector = make_collector(envs, config)
                trainer = PPOTrainer(collector.agent, config)
                for _ in range(2):
                    buffer = collector.collect(config.num_steps)
                    metrics = trainer.update(buffer)
                    self.assertLess(metrics["consistency_max_logprob_error"], 1e-4)
                    collector.invalidate_cache()
            finally:
                envs.close()
