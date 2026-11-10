"""独立骨干的梯度隔离、成对初始化、双缓存和旧 checkpoint 兼容。"""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import torch

from acceleration import _sequence_inputs
from diagnose_consistency import replay_snapshot, save_snapshot
from model import Agent, SeparateCache, cache_tensors
from ppo import PPOTrainer, check_consistency
from rollout import Collector
from runtime import configure_runtime, load_checkpoint, save_checkpoint
from train import make_parser
from fixtures import fixture_vector, small_config


class SeparateBackboneTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(3, "cpu")
        config = small_config(compile_mode="off")
        self.config = replace(config, agent=replace(config.agent, backbone_mode="separate"))

    def inputs(self):
        return _sequence_inputs(self.config, 2, 12, torch.device("cpu"))

    def test_paired_initialization_rng_and_disjoint_parameters(self):
        shared = Agent(replace(self.config.agent, backbone_mode="shared"))
        shared_rng = torch.get_rng_state().clone()
        configure_runtime(3, "cpu")
        separate = Agent(self.config.agent)
        torch.testing.assert_close(torch.get_rng_state(), shared_rng, rtol=0, atol=0)
        for name, value in shared.state_dict().items():
            torch.testing.assert_close(separate.state_dict()[name], value, rtol=0, atol=0)
        args = self.inputs()
        for left, right in zip(shared._sequence_core(*args), separate._sequence_core(*args)):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        policy, value = list(separate.policy_parameters()), list(separate.value_parameters())
        self.assertFalse({id(p) for p in policy} & {id(p) for p in value})
        self.assertFalse({p.data_ptr() for p in policy} & {p.data_ptr() for p in value})
        self.assertEqual(separate.parameter_counts()["shared"], 0)
        self.assertGreater(shared.parameter_counts()["shared"], 0)
        self.assertEqual(separate.value_backbone.config, separate.backbone.config)
        with self.assertRaisesRegex(ValueError, "value_hidden"):
            separate.heads(torch.zeros(1, self.config.agent.hidden_dim))
        self.assertEqual(make_parser().parse_args([]).backbone_mode, "shared")
        self.assertEqual(make_parser().parse_args(["--backbone-mode", "separate"]).backbone_mode, "separate")
        with self.assertRaises(ValueError):
            replace(self.config.agent, backbone_mode="invalid")

    def test_losses_and_value_only_step_cannot_change_policy(self):
        agent = Agent(self.config.agent)
        args = self.inputs()
        logits, values = agent._sequence_core(*args)
        initial_logits = logits.detach().clone()
        policy_state = {name: tensor.clone() for name, tensor in agent.state_dict().items()
                        if name.startswith(("encoder.", "backbone.", "actor."))}
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        (values - 100).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in agent.policy_parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in agent.value_parameters()), 0)
        optimizer.step()
        for name, tensor in policy_state.items():
            torch.testing.assert_close(agent.state_dict()[name], tensor, rtol=0, atol=0)
        changed_logits, changed_values = agent._sequence_core(*args)
        torch.testing.assert_close(changed_logits, initial_logits, rtol=0, atol=0)
        self.assertGreater((changed_values.detach() - values.detach()).abs().max().item(), 0)
        optimizer.zero_grad(set_to_none=True)
        changed_logits[..., 2].mean().backward()
        self.assertTrue(all(p.grad is None for p in agent.value_parameters()))
        self.assertGreater(sum(p.grad.abs().sum().item() for p in agent.policy_parameters()), 0)

    def test_dual_cache_matches_sequence_with_wraparound_and_async_resets(self):
        agent = Agent(self.config.agent)
        # 令两分支有不同特征，避免同值初始条件掩盖误用策略缓存/表示的问题。
        with torch.no_grad():
            agent.value_encoder[0].weight.add_(0.1)
        features, starts, valid, positions = self.inputs()
        starts.zero_()
        starts[:, 0] = True
        starts[0, 5] = starts[1, 8] = True
        valid.fill_(True)
        times = torch.arange(12).expand(2, 12)
        positions = times - torch.where(starts, times, 0).cummax(-1).values
        parallel = agent._sequence_core(features, starts, valid, positions)
        cache = agent.create_cache(2)
        self.assertIsInstance(cache, SeparateCache)
        self.assertNotEqual(cache.policy.keys.data_ptr(), cache.value.keys.data_ptr())
        steps = []
        with torch.no_grad():
            for t in range(12):
                steps.append(agent._step_core(features[:, t], cache, starts[:, t]))
                torch.testing.assert_close(cache.policy.next_positions, cache.value.next_positions)
                torch.testing.assert_close(cache.policy.lengths, cache.value.lengths)
        for index, reference in enumerate(parallel):
            torch.testing.assert_close(torch.stack([s[index] for s in steps], 1), reference, rtol=1e-4, atol=1e-5)
        # 两个输出都必须遵守回合、未来与 padding 隔离。
        valid[1, 10:] = False
        original = agent._sequence_core(features, starts, valid, positions)
        altered = features.clone()
        altered[0, :5] += 100
        altered[1, 10:] += 100
        for actual, expected in zip(agent._sequence_core(altered, starts, valid, positions), original):
            torch.testing.assert_close(actual[0, 5:], expected[0, 5:])
            torch.testing.assert_close(actual[1, :10], expected[1, :10])
        future = features.clone()
        future[:, 9:] += 100
        for actual, expected in zip(agent._sequence_core(future, starts, valid, positions), original):
            torch.testing.assert_close(actual[:, :9], expected[:, :9])

    def test_updates_rebuild_both_caches_peek_and_diagnostic_replay(self):
        # 一个环境在 rollout 内结束，另一个跨更新继续，覆盖双缓存的两种边界。
        envs = fixture_vector((3, 100), (False, False))
        try:
            agent = Agent(self.config.agent)
            collector = Collector(envs, agent, torch.device("cpu"), self.config.seed,
                                  sensor_config=self.config.sensors,
                                  total_timesteps=self.config.effective_total_timesteps)
            trainer = PPOTrainer(agent, self.config)
            for update in range(2):
                buffer = collector.collect(self.config.num_steps)
                before = {name: tensor.clone() for name, tensor in cache_tensors(collector.cache).items()}
                collector.peek_values(collector.observations, collector.previous_actions, [0, 1])
                for name, tensor in cache_tensors(collector.cache).items():
                    torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
                report = check_consistency(agent, buffer, self.config)
                self.assertLess(report["consistency_max_logprob_error"], 1e-4)
                if update:
                    self.assertEqual(len(buffer.prefix), self.config.agent.warmup_length)
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / "snapshot.pt"
                        save_snapshot(path, agent, buffer, self.config, update=2, global_step=52,
                                      report=report, environment={}, severity="test")
                        replay = replay_snapshot(path, device="cpu", precision="fp64")
                        self.assertEqual(replay["position_mismatches"], 0)
                        self.assertLess(replay["parallel_vs_serial"]["max_abs_error"], 1e-10)
                trainer.update(buffer)
                collector.invalidate_cache()
        finally:
            envs.close()

    def test_checkpoint_separate_and_legacy_shared(self):
        inputs = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            for mode in ("shared", "separate"):
                config = replace(self.config, agent=replace(self.config.agent, backbone_mode=mode))
                agent = Agent(config.agent)
                save_checkpoint(path, agent, config, 52)
                if mode == "shared":
                    old = torch.load(path, weights_only=True)
                    old["agent_config"].pop("backbone_mode")
                    old["train_config"]["agent"].pop("backbone_mode")
                    torch.save(old, path)
                loaded, _ = load_checkpoint(path, torch.device("cpu"))
                self.assertEqual(loaded.config.backbone_mode, mode)
                for actual, expected in zip(loaded._sequence_core(*inputs), agent._sequence_core(*inputs)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                disabled, _ = load_checkpoint(path, torch.device("cpu"), use_previous_action=False)
                self.assertFalse(disabled.config.use_previous_action)
                self.assertEqual(disabled.config.backbone_mode, mode)
