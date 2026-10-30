"""运行：python -B -m unittest discover -s mountaincar-ppo-rnd -p test_ppo_rnd.py -v"""

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from model import PPOActorCritic
from ppo_rnd import RewardForwardFilter, RolloutBuffer, combine_advantages
from rnd import RunningMeanStd, scale_intrinsic_rewards
from diagnostics import evaluate, rollout_metrics
import train


class PPORNDTests(unittest.TestCase):
    def test_evaluation_reaches_goal_without_changing_training_rng(self):
        class SwingPolicy(torch.nn.Module):
            def forward(self, obs):
                logits = torch.full((len(obs), 3), -50.0)
                actions = torch.where(obs[:, 1] >= 0, 2, 0)
                logits[torch.arange(len(obs)), actions] = 50.0
                return logits, torch.zeros(len(obs)), torch.zeros(len(obs))

        before = torch.random.get_rng_state().clone()
        result = evaluate(SwingPolicy(), torch.device("cpu"), episodes=8,
                          deterministic=False)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertGreater(result["return_mean"], -200)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_scaled_model_checkpoint_preserves_input_transform(self):
        for architecture in ["residual", "mlp"]:
            model = PPOActorCritic(2, 3, architecture=architecture, normalize_input=True)
            restored = PPOActorCritic(2, 3, architecture=architecture, normalize_input=True)
            restored.load_state_dict(model.state_dict())
            obs = torch.tensor([[-1.2, -0.07], [0.6, 0.07]])
            for actual, expected in zip(restored(obs), model(obs)):
                torch.testing.assert_close(actual, expected)
            torch.testing.assert_close((obs - model.obs_center) / model.obs_scale,
                                       torch.tensor([[-1., -1.], [1., 1.]]))

    def test_reward_scaling_uses_filtered_std_but_single_step_numerator(self):
        buffer = RolloutBuffer(2, 2, 2, "cpu")
        raw_rewards = torch.tensor([[1., 10.], [2., 0.]])
        buffer.rewards_int.copy_(raw_rewards)
        reward_filter = RewardForwardFilter(gamma=0.5, num_envs=2)
        reward_rms = RunningMeanStd()
        expected_rms = RunningMeanStd()
        expected_rms.update(np.array([1., 10., 2.5, 5.]))

        logs = scale_intrinsic_rewards(buffer, reward_filter, reward_rms)

        torch.testing.assert_close(
            buffer.rewards_int, raw_rewards / (float(expected_rms.std) + 1e-8)
        )
        np.testing.assert_allclose(reward_filter.discounted_rewards, [2.5, 5.])
        self.assertAlmostEqual(logs["reward_int_raw_mean"], raw_rewards.mean().item())
        self.assertAlmostEqual(logs["int_return_std"], float(expected_rms.std))

    def test_intrinsic_gae_continues_across_episode_boundary(self):
        buffer = RolloutBuffer(2, 1, 2, "cpu")
        buffer.step = 2
        buffer.rewards_int[:, 0] = torch.tensor([1., 2.])
        buffer.values_int[:, 0] = torch.tensor([0.5, 0.7])
        buffer.next_values_int[:, 0] = torch.tensor([0.7, 3.])
        buffer.terminated[0, 0] = True
        buffer.truncated[0, 0] = True

        buffer.compute_int_returns_and_advantages(gamma=0.9, lmbda=0.8)

        # delta = [1.13, 4]；A_0 = 1.13 + 0.9 * 0.8 * 4。
        torch.testing.assert_close(buffer.adv_int[:, 0], torch.tensor([4.01, 4.]))
        torch.testing.assert_close(buffer.returns_int[:, 0], torch.tensor([4.51, 4.7]))

    def test_forward_filter_keeps_each_env_and_rollout_independent(self):
        reward_filter = RewardForwardFilter(gamma=0.5, num_envs=2)
        first = reward_filter.update([1, 10])
        second = reward_filter.update([2, 0])
        third = reward_filter.update([3, 2])  # 下一 rollout 继续调用，不清零
        np.testing.assert_allclose(first, [1, 10])
        np.testing.assert_allclose(second, [2.5, 5])
        np.testing.assert_allclose(third, [4.25, 4.5])

    def test_combination_preserves_relative_scale(self):
        ext = torch.tensor([1.0, 2.0, 4.0])
        intrinsic = torch.tensor([3.0, -1.0, 2.0])
        original = combine_advantages(ext, intrinsic, 2.0, 1.0)
        faded = combine_advantages(ext, intrinsic * 0.01, 2.0, 1.0)
        self.assertFalse(torch.allclose(original, faded))
        torch.testing.assert_close(original.mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
        self.assertTrue(torch.isfinite(combine_advantages(ext[:1], intrinsic[:1], 2, 1)).all())

    def test_episode_tracking_handles_async_and_simultaneous_endings(self):
        tracker = train.EpisodeTracker(2)
        self.assertTrue(np.isnan(tracker.metrics()["success_rate"]))
        tracker.update(np.array([-1., -1.]), np.array([False, False]), np.array([True, False]), 2)
        tracker.update(np.array([-1., -1.]), np.array([True, True]), np.array([True, False]), 4)
        self.assertEqual([e["length"] for e in tracker.episodes], [1, 1, 2])
        self.assertEqual([e["return_ext"] for e in tracker.episodes], [-1, -1, -2])
        self.assertAlmostEqual(tracker.metrics()["success_rate"], 2 / 3)
        np.testing.assert_array_equal(tracker.lengths, [0, 0])

    def test_real_same_step_final_obs_and_bootstrap(self):
        envs = gym.vector.SyncVectorEnv(
            [lambda: gym.make("MountainCar-v0")],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        try:
            agent = PPOActorCritic(2, 3)
            envs.reset(seed=5)
            for _ in range(200):
                obs, _, terminated, truncated, infos = envs.step(np.array([1]))
            self.assertTrue(truncated[0])
            true_obs, next_ext, next_int = train.get_transition_next_info(
                agent, torch.tensor(obs), terminated, truncated, infos
            )
            torch.testing.assert_close(true_obs, torch.tensor(np.stack(infos["final_obs"])))
            with torch.no_grad():
                torch.testing.assert_close(next_ext, agent(true_obs)[1])
                torch.testing.assert_close(next_int, agent(torch.tensor(obs))[2])

            # 强制到终点附近，验证真正终止时外在 bootstrap 为 0。
            envs.envs[0].unwrapped.state = np.array([0.499, 0.02])
            obs, _, terminated, truncated, infos = envs.step(np.array([2]))
            self.assertTrue(terminated[0])
            _, next_ext, next_int = train.get_transition_next_info(
                agent, torch.tensor(obs), terminated, truncated, infos
            )
            torch.testing.assert_close(next_ext, torch.zeros_like(next_ext))
            with torch.no_grad():
                torch.testing.assert_close(next_int, agent(torch.tensor(obs))[2])
        finally:
            envs.close()

    def test_extrinsic_trace_stops_at_truncation(self):
        buffer = RolloutBuffer(2, 1, 2, "cpu")
        buffer.step = 2
        buffer.rewards_ext[:, 0] = torch.tensor([1., 100.])
        buffer.next_values_ext[:, 0] = torch.tensor([5., 0.])
        buffer.truncated[0, 0] = True
        buffer.compute_ext_returns_and_advantages(gamma=0.9, lmbda=0.95)
        torch.testing.assert_close(buffer.adv_ext[:, 0], torch.tensor([5.5, 100.]))

    def test_short_training_uses_same_rnd_samples_and_statistics(self):
        torch.manual_seed(7)
        np.random.seed(7)
        original_update = train.PPOTrainer.update
        original_reward = train.compute_intrinsic_reward
        collected = []
        checks = []

        def record_reward(rnd_model, obs, rms):
            collected.append((obs.clone(), rms.mean.copy(), rms.var.copy()))
            return original_reward(rnd_model, obs, rms)

        def check_update(trainer, buffer, obs_rms):
            torch.testing.assert_close(buffer.rnd_obs, torch.stack([x[0] for x in collected]))
            for _, mean, var in collected:
                np.testing.assert_array_equal(obs_rms.mean, mean)
                np.testing.assert_array_equal(obs_rms.var, var)
            target_before = [p.detach().clone() for p in trainer.rnd_model.target.parameters()]
            predictor_before = [p.detach().clone() for p in trainer.rnd_model.predictor.parameters()]
            # 拦截 predictor 的输入，确认更新使用奖励计算时相同的归一化后继状态。
            seen = []
            handle = trainer.rnd_model.register_forward_pre_hook(
                lambda model, args: seen.append(args[0].detach().clone())
            )
            expected = train.normalize_rnd_obs(buffer.rnd_obs.reshape(-1, 2), obs_rms)
            try:
                logs = original_update(trainer, buffer, obs_rms)
            finally:
                handle.remove()
            # 更新时索引会打乱；按行排序后比较完整样本集合。
            actual = torch.cat(seen[:8]).numpy()  # 首个 epoch: 512 / 64 个 minibatch
            expected_np = expected.numpy()
            np.testing.assert_allclose(
                actual[np.lexsort(actual.T[::-1])],
                expected_np[np.lexsort(expected_np.T[::-1])], atol=1e-6,
            )
            self.assertTrue(all(np.isfinite(v) for v in logs.values()))
            self.assertTrue(all(torch.equal(p, q) for p, q in zip(target_before, trainer.rnd_model.target.parameters())))
            self.assertTrue(any(not torch.equal(p, q) for p, q in zip(predictor_before, trainer.rnd_model.predictor.parameters())))
            checks.append(True)
            collected.clear()
            return logs

        with tempfile.TemporaryDirectory(prefix="ppo-rnd-test-") as temp_dir:
            with (
                patch.object(train, "RUN_DIR", Path(temp_dir)),
                patch.object(train, "TOTAL_TIMESTEPS", 1024),
                patch.object(train, "WARMUP_STEPS", 8),
                patch.object(train, "LOG_INTERVAL", 2),
                patch.object(train, "DEVICE", torch.device("cpu")),
                patch("torch.compile", side_effect=lambda model, **kw: model),
                patch.object(train, "compute_intrinsic_reward", side_effect=record_reward),
                patch.object(train.PPOTrainer, "update", new=check_update),
            ):
                train.train()
            self.assertEqual(len(checks), 2)
            with (Path(temp_dir) / "training.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows[-1]["episodes_completed"]), 4)
            for key in ["episode_return_mean", "episode_length_mean", "success_rate", "reward_int_raw_mean", "reward_int_scaled_mean", "int_return_std"]:
                self.assertTrue(np.isfinite(float(rows[-1][key])), key)
            with (Path(temp_dir) / "episodes.csv").open(encoding="utf-8") as file:
                episodes = list(csv.DictReader(file))
            self.assertEqual(len(episodes), 4)
            self.assertTrue((Path(temp_dir) / "training.png").is_file())
            self.assertTrue((Path(temp_dir) / "checkpoints/ppo_rnd_model.pth").is_file())
            checkpoint = torch.load(
                Path(temp_dir) / "checkpoints/ppo_rnd_model.pth",
                map_location="cpu", weights_only=True,
            )
            self.assertEqual(checkpoint["int_reward_filter"].shape, (4,))
            self.assertEqual(checkpoint["obs_rms"]["mean"].shape, (2,))
            restored = PPOActorCritic(**checkpoint["model_config"])
            restored.load_state_dict(checkpoint["agent"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
