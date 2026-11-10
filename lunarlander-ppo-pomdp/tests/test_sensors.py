"""失灵语义、独立性、随机流和课程边界。"""

import unittest

import numpy as np
import torch

from sensors import FAILURE_INDICES, PROTECTED_INDICES, SensorConfig, SensorProcessor
from fixtures import small_config


class SensorTests(unittest.TestCase):
    def test_full_partial_and_protected_channels(self):
        raw = np.array([0.2, 0.3, 1, 2, 3, 4, 0, 1], dtype=np.float32)
        unchanged = raw.copy()
        full = SensorProcessor(7).process(raw, 0)
        np.testing.assert_array_equal(full[:8], raw)
        np.testing.assert_array_equal(full[8:], np.ones(8))
        partial = SensorProcessor(7).process(raw, 1)
        np.testing.assert_array_equal(partial[list(FAILURE_INDICES)], np.zeros(4))
        np.testing.assert_array_equal(partial[8 + np.array(FAILURE_INDICES)], np.zeros(4))
        np.testing.assert_array_equal(partial[list(PROTECTED_INDICES)], raw[list(PROTECTED_INDICES)])
        np.testing.assert_array_equal(partial[8 + np.array(PROTECTED_INDICES)], np.ones(4))
        np.testing.assert_array_equal(raw, unchanged)
        self.assertEqual(partial.dtype, np.float32)

    def test_zero_and_missing_are_distinct_and_hidden_values_do_not_leak(self):
        raw = np.zeros((2, 8), dtype=np.float32)
        normal, failed = SensorProcessor(1).process(raw, 0), SensorProcessor(1).process(raw, 1)
        np.testing.assert_array_equal(normal[:, :8], failed[:, :8])
        self.assertFalse(np.array_equal(normal, failed))
        raw[:, FAILURE_INDICES] = 999
        np.testing.assert_array_equal(SensorProcessor(1).process(raw, 1), failed)

    def test_independent_sensor_environment_and_time_draws(self):
        data = SensorProcessor(42).process(np.ones((4000, 8, 8), dtype=np.float32), 0.5)
        mask = data[..., 8 + np.array(FAILURE_INDICES)]
        np.testing.assert_allclose(mask.mean(axis=(0, 1)), 0.5, atol=0.015)
        # 联合概率为 .25；排除所有传感器、环境或时间共用一个掩码的实现。
        self.assertAlmostEqual(float((mask[..., 0] * mask[..., 1]).mean()), 0.25, delta=0.015)
        self.assertAlmostEqual(float((mask[:, 0, 0] * mask[:, 1, 0]).mean()), 0.25, delta=0.025)
        self.assertAlmostEqual(float((mask[:-1, 0, 0] * mask[1:, 0, 0]).mean()), 0.25, delta=0.025)
        self.assertGreater(np.unique(mask.reshape(-1, 4), axis=0).shape[0], 10)

    def test_reproducibility_and_isolated_random_stream(self):
        raw = np.ones((32, 8), dtype=np.float32)
        first, second = SensorProcessor(3), SensorProcessor(3)
        np.random.seed(11)
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        for probability in (0, 0.2, 0.8, 1):
            np.testing.assert_array_equal(first.process(raw, probability), second.process(raw, probability))
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_state[0], numpy_after[0])
        np.testing.assert_array_equal(numpy_state[1], numpy_after[1])
        self.assertEqual(numpy_state[2:], numpy_after[2:])
        torch.testing.assert_close(torch_state, torch.get_rng_state(), rtol=0, atol=0)
        self.assertFalse(np.array_equal(SensorProcessor(3).process(raw, 0.5),
                                       SensorProcessor(4).process(raw, 0.5)))

    def test_schedule_boundaries_controls_and_actual_budget(self):
        schedule = SensorConfig()
        for step, expected in ((0, 0), (10, 0), (45, 0.5), (80, 1), (100, 1), (110, 1)):
            self.assertAlmostEqual(schedule.failure_probability(step, 100), expected)
        for mode, expected in (("full", 0), ("partial", 1)):
            for step in (0, 45, 100):
                self.assertEqual(SensorConfig(mode=mode).failure_probability(step, 100), expected)
        config = small_config(total_timesteps=55)
        self.assertEqual(config.effective_total_timesteps, 52)
        self.assertEqual(config.sensors.failure_probability(42, config.effective_total_timesteps), 1)
        self.assertAlmostEqual(SensorConfig(start_fraction=0, end_fraction=1).failure_probability(50, 100), 0.5)

    def test_invalid_settings_fail_early(self):
        for kwargs in ({"mode": "unknown"}, {"start_fraction": -0.1}, {"end_fraction": 1.1},
                       {"start_fraction": 0.8, "end_fraction": 0.8}, {"end_fraction": float("nan")},
                       {"start_fraction": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SensorConfig(**kwargs)
        processor = SensorProcessor(1)
        for probability in (-0.1, 1.1, float("nan"), float("inf"), True):
            with self.subTest(probability=probability), self.assertRaises(ValueError):
                processor.process(np.ones(8), probability)
        for raw in (np.ones(2), np.ones(9), np.full(8, np.nan), np.array(1)):
            with self.assertRaises(ValueError):
                processor.process(raw, 0.5)
