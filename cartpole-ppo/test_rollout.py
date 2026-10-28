"""检查 GAE 边界和 SAME_STEP 采样；无需运行完整训练。"""

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from model import PPOSeparateActorCritic
from ppo import PPOTrainer, RolloutBuffer
import train as training


class RolloutTests(unittest.TestCase):
    def test_episode_boundaries_use_different_masks(self):
        # 四列分别是真正终止、仅截断、同时终止和截断、正常继续。
        buffer = RolloutBuffer(2, 4, 1, "cpu")
        buffer.add(
            torch.zeros(4, 1), torch.zeros(4), torch.zeros(4),
            torch.ones(4), torch.ones(4),
            terminated=torch.tensor([True, False, True, False]),
            truncated=torch.tensor([False, True, True, False]),
            next_value=torch.tensor([10.0, 10.0, 10.0, 2.0]),
        )
        # 大奖励用于检查 reset 后的新回合是否错误地影响前一回合。
        buffer.add(
            torch.zeros(4, 1), torch.zeros(4), torch.zeros(4),
            torch.full((4,), 100.0), torch.full((4,), 2.0),
            terminated=torch.zeros(4, dtype=torch.bool),
            truncated=torch.zeros(4, dtype=torch.bool),
            next_value=torch.full((4,), 4.0),
        )
        buffer.compute_returns_and_advantages(gamma=0.9, gae_lambda=0.5)

        # 末步 A = 100 + 0.9 * 4 - 2 = 101.6。
        # 首步仅正常列接上 0.9 * 0.5 * 101.6；仅截断列保留 bootstrap。
        expected = torch.tensor([
            [0.0, 9.0, 0.0, 47.52],
            [101.6, 101.6, 101.6, 101.6],
        ])
        torch.testing.assert_close(buffer.advantages, expected)
        torch.testing.assert_close(buffer.returns, expected + buffer.values)
        torch.testing.assert_close(buffer.rewards[0], torch.ones(4))

    def test_single_env_rollout_tail_bootstraps(self):
        buffer = RolloutBuffer(1, 1, 1, "cpu")
        buffer.add(
            torch.zeros(1, 1), torch.zeros(1), torch.zeros(1),
            torch.tensor([3.0]), torch.tensor([2.0]),
            terminated=torch.tensor([False]),
            truncated=torch.tensor([False]),
            next_value=torch.tensor([5.0]),
        )
        buffer.compute_returns_and_advantages(gamma=0.8)
        # rollout 收满不是终止：return = 3 + 0.8 * 5 = 7。
        torch.testing.assert_close(buffer.returns, torch.tensor([[7.0]]))
        self.assertEqual(buffer.get_batch_data()[3].shape, (1,))

    def test_same_step_uses_final_obs_without_changing_reset_obs(self):
        agent = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            agent.weight.fill_(1.0)

        class SumCritic:
            def get_value(self, obs):
                return agent(obs).squeeze(-1)

        next_obs = torch.tensor([[1.0, 2.0], [100.0, 100.0],
                                 [200.0, 200.0], [300.0, 300.0]])
        original = next_obs.clone()
        next_values = training.get_transition_next_values(
            SumCritic(), next_obs,
            np.array([False, True, False, True]),
            np.array([False, False, True, True]),
            {"final_obs": [None, None, np.array([3.0, 4.0]), None]},
        )
        torch.testing.assert_close(next_values, torch.tensor([3.0, 0.0, 7.0, 0.0]))
        torch.testing.assert_close(next_obs, original)
        self.assertFalse(next_values.requires_grad)

        # 没有回合结束时，infos 中可以没有 final_obs。
        normal_values = training.get_transition_next_values(
            SumCritic(), next_obs[:1], np.array([False]), np.array([False]), {}
        )
        torch.testing.assert_close(normal_values, torch.tensor([3.0]))

    def test_critic_keeps_single_env_batch_dimension(self):
        agent = PPOSeparateActorCritic(4, 2)
        self.assertEqual(agent.get_value(torch.zeros(1, 4)).shape, (1,))

    def test_short_training_with_repeated_timeouts(self):
        # 两次 rollout，每两步截断一次，覆盖自动 reset、优势计算和参数更新。
        trainers = []
        losses = []

        class CheckedTrainer(PPOTrainer):
            def update(self, buffer):
                torch.testing.assert_close(buffer.rewards, torch.ones_like(buffer.rewards))
                if not torch.isfinite(buffer.returns).all():
                    raise AssertionError("Non-finite returns")
                result = super().update(buffer)
                losses.append(result)
                return result

        def make_trainer(agent):
            trainer = CheckedTrainer(
                agent, gamma=0.8, gae_lambda=0.6, update_epochs=1, minibatch_size=4
            )
            trainers.append(trainer)
            return trainer

        def short_env(env_id):
            return lambda: gym.make(env_id, max_episode_steps=2)

        with (
            TemporaryDirectory() as temp_dir,
            patch.multiple(training, NUM_ENVS=2, NUM_STEPS=4, TOTAL_TIMESTEPS=16,
                           DEVICE=torch.device("cpu"), RUN_DIR=Path(temp_dir)),
            patch.object(training, "make_env", side_effect=short_env),
            patch.object(training, "PPOTrainer", side_effect=make_trainer),
            patch("builtins.print"),
        ):
            training.train()
            output = Path(temp_dir)
            with (output / "training.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([int(row["global_step"]) for row in rows], [8, 16])
            self.assertEqual([int(row["update"]) for row in rows], [1, 2])
            self.assertTrue(np.isfinite(training.plt.imread(output / "training.png")).all())
            weights = torch.load(output / "checkpoints" / "ppo_model.pth", weights_only=True)
            for key, value in trainers[0].agent.state_dict().items():
                torch.testing.assert_close(weights[key], value)

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(np.isfinite(list(result.values())).all() for result in losses))
        self.assertTrue(trainers[0].optimizer.state)

    def test_log_files_are_overwritten_and_figures_are_closed(self):
        row = {"update": 1, "global_step": 8, "policy_loss": 0.1,
               "value_loss": 0.2, "entropy": 0.5, "clip_fraction": 0.0}
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "runs"
            training.save_training_logs([row, {**row, "update": 2, "global_step": 16}], output)
            # 模拟重新训练，只写一条日志；上次的第二条不能残留。
            training.save_training_logs([{**row, "policy_loss": 0.9}], output)
            with (output / "training.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["policy_loss"]), 0.9)
            self.assertEqual((output / "training.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(training.plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
