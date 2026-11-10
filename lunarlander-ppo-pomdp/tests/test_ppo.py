"""真实模型、GAE、PPO 数学和 checkpoint 协议。"""

import tempfile
from pathlib import Path
import unittest

import torch

from buffer import RolloutBuffer
from model import Agent, make_model_input
from ppo import ppo_losses
from runtime import configure_runtime, load_checkpoint, save_checkpoint
from sensors import input_spec
from fixtures import small_config


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(3, "cpu")

    def test_previous_actions_and_start_sentinel(self):
        observations = torch.randn(5, 16)
        actions = torch.tensor([-1, 0, 1, 2, 3])
        enabled = make_model_input(observations, actions)
        self.assertEqual(enabled.shape, (5, 20))
        torch.testing.assert_close(enabled[:, :16], observations)
        torch.testing.assert_close(enabled[:, 16:], torch.cat((torch.zeros(1, 4), torch.eye(4))))
        torch.testing.assert_close(make_model_input(observations, actions, False)[:, 16:], torch.zeros(5, 4))
        for wrong in (torch.tensor([-2, 0, 1, 2, 3]), torch.tensor([4, 0, 1, 2, 3])):
            with self.assertRaises(ValueError):
                make_model_input(observations, wrong)
        with self.assertRaises(TypeError):
            make_model_input(observations, actions.float())
        with self.assertRaises(ValueError):
            make_model_input(torch.zeros(5, 2), actions)

    def test_encoder_heads_sampling_and_gradients(self):
        agent = Agent(small_config().agent)
        observations = torch.randn(2, 3, 16, requires_grad=True)
        encoded = agent.encode(observations, torch.zeros(2, 3, dtype=torch.long))
        logits, values = agent.heads(encoded)
        self.assertEqual(logits.shape, (2, 3, 4))
        self.assertEqual(values.shape, (2, 3))
        values.sum().backward()
        self.assertGreater(observations.grad.abs().sum().item(), 0)
        self.assertGreater(agent.encoder[0].weight.grad.abs().sum().item(), 0)
        self.assertEqual(agent.heads(torch.zeros(1, 32))[1].shape, (1,))
        actions, log_probs = agent.sample(torch.zeros(1024, 4, requires_grad=True))
        self.assertEqual(actions.dtype, torch.long)
        torch.testing.assert_close(log_probs, torch.full((1024,), -torch.log(torch.tensor(4.)).item()))
        self.assertTrue(log_probs.requires_grad)
        frequencies = torch.bincount(actions, minlength=4).float() / 1024
        self.assertTrue(((frequencies > 0.15) & (frequencies < 0.35)).all())
        table = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()
        actions, log_probs = agent.sample(table.expand(256, 4))
        torch.testing.assert_close(log_probs, table[actions])

    def test_gae_termination_timeout_and_unscaled_returns(self):
        buffer = RolloutBuffer(3, 2, torch.device("cpu"), [], 7)
        buffer.step = 3
        buffer.rewards.copy_(torch.tensor([[1., 2.], [3., 4.], [5., 6.]]))
        buffer.values.copy_(torch.tensor([[0.5, 1.], [1.5, 2.], [2.5, 3.]]))
        buffer.next_values.copy_(torch.tensor([[99., 2.], [2.5, 10.], [4., 5.]]))
        buffer.terminated[0, 0] = buffer.terminated[2, 1] = True
        buffer.truncated[1, 1] = True
        buffer.compute_returns_and_advantages(0.9, 0.5)
        torch.testing.assert_close(buffer.returns, torch.tensor([[1., 8.75], [7.995, 13.], [8.6, 6.]]))
        expected = torch.tensor([[0.5, 7.75], [6.495, 11.], [6.1, 3.]])
        torch.testing.assert_close(buffer.advantages, (expected - expected.mean()) / expected.std(unbiased=False))
        single = RolloutBuffer(1, 1, torch.device("cpu"), [], 7)
        with self.assertRaises(RuntimeError):
            single.compute_returns_and_advantages(0.9, 0.5)
        single.step = 1
        single.compute_returns_and_advantages(0.9, 0.5)
        torch.testing.assert_close(single.advantages, torch.zeros(1, 1))

    def test_clipped_policy_value_entropy_losses_and_gradient_direction(self):
        values = torch.tensor([1., 3.], requires_grad=True)
        entropy = torch.tensor([0.2, 0.4], requires_grad=True)
        policy, value, total = ppo_losses(
            torch.tensor([1.5, 0.5], requires_grad=True), torch.tensor([1., -1.]),
            values, torch.tensor([2., 1.]), entropy, clip_coef=0.2, vf_coef=0.5, ent_coef=0.01,
        )
        torch.testing.assert_close(policy, torch.tensor(-0.2))
        torch.testing.assert_close(value, torch.tensor(1.25))
        torch.testing.assert_close(total, torch.tensor(0.422))
        total.backward()
        self.assertGreater(values.grad.abs().sum().item(), 0)
        self.assertTrue((entropy.grad < 0).all())
        ratio = torch.tensor([1.1, 0.9], requires_grad=True)
        policy, _, _ = ppo_losses(ratio, torch.tensor([1., -1.]), values, torch.zeros(2), entropy,
                                  clip_coef=0.2, vf_coef=0.5, ent_coef=0.01)
        policy.backward()
        torch.testing.assert_close(ratio.grad, torch.tensor([-0.5, 0.5]))

    def test_checkpoint_v2_roundtrip_action_override_and_legacy_rejection(self):
        config = small_config()
        agent = Agent(config.agent)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latest.pt"
            save_checkpoint(path, agent, config, 52)
            loaded, saved = load_checkpoint(path, torch.device("cpu"))
            self.assertEqual(saved["format_version"], 2)
            self.assertEqual(saved["input_spec"], input_spec())
            self.assertEqual(saved["train_config"]["sensors"]["mode"], "curriculum")
            self.assertEqual(loaded.encoder[0].in_features, 20)
            disabled, _ = load_checkpoint(path, torch.device("cpu"), False)
            self.assertFalse(disabled.config.use_previous_action)
            for name, expected in agent.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], expected)
                torch.testing.assert_close(disabled.state_dict()[name], expected)
            torch.save({"format_version": 1}, path)
            with self.assertRaisesRegex(ValueError, "6 维.*20 维"):
                load_checkpoint(path, torch.device("cpu"))
            saved["input_spec"]["sensor_names"].reverse()
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "输入布局"):
                load_checkpoint(path, torch.device("cpu"))
