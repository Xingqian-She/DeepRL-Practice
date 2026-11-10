"""真实 Box2D 短程 PPO、产物、评估和视频；不检验长期收敛。"""

import csv
from dataclasses import replace
import json
import math
import tempfile
import unittest

import torch

from envs import make_env
from evaluate import run_evaluation
from model import Agent
from runtime import configure_runtime
from sensors import SensorConfig
from train import run_training
from fixtures import small_config


class TrainingTests(unittest.TestCase):
    def _closed_loop(self, mode, device="cpu", previous_action=True, video=False, backbone_mode="shared"):
        config = small_config(total_timesteps=64, num_steps=16, sequence_length=6,
                              seed=9, device=device, sensors=SensorConfig(mode=mode))
        config = replace(config, agent=replace(config.agent, use_previous_action=previous_action,
                                               backbone_mode=backbone_mode))
        configure_runtime(config.seed, device)
        initial = {name: value.clone().cpu() for name, value in Agent(config.agent).state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            run = run_training(config, directory)
            checkpoint = run / "checkpoints/latest.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            groups = ["encoder.", "backbone.", "actor.", "critic."]
            if backbone_mode == "separate":
                groups.extend(("value_encoder.", "value_backbone."))
            for group in groups:
                self.assertTrue(any(not torch.equal(initial[name], value) for name, value in saved["state_dict"].items()
                                    if name.startswith(group)), f"{group} 参数没有更新")
            metadata = json.loads((run / "metadata.json").read_text())
            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["global_step"], 64)
            self.assertEqual(metadata["effective_total_timesteps"], 64)
            self.assertEqual(metadata["input_spec"]["input_dim"], 20)
            self.assertEqual(metadata["model"]["backbone_mode"], backbone_mode)
            self.assertEqual(metadata["model"]["parameters"]["shared"] == 0, backbone_mode == "separate")
            self.assertEqual(metadata["acceleration"]["enabled"], device == "cuda")
            self.assertEqual(saved["acceleration"], metadata["acceleration"])
            self.assertGreater(metadata["elapsed_seconds"], 0)
            self.assertTrue(list((run / "tensorboard").glob("events.*")))
            self.assertTrue((run / "episodes.csv").is_file())
            with (run / "training.csv").open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            for row in rows:
                self.assertTrue(all(math.isfinite(float(value)) for value in row.values() if value))
                for key in ("collect_seconds", "ppo_seconds", "steps_per_second"):
                    self.assertGreater(float(row[key]), 0)
            self.assertEqual(float(rows[0]["sensor_failure_probability_start"]), 1 if mode == "partial" else 0)
            self.assertEqual(float(rows[-1]["sensor_failure_probability_end"]), 0 if mode == "full" else 1)
            if mode != "curriculum":
                for row in rows:
                    for key, value in row.items():
                        if key.startswith("sensor_missing_rate_"):
                            self.assertEqual(float(value), 0 if mode == "full" else 1)
            for probability, override in ((0, None), (1, False)):
                evaluated = run_evaluation(checkpoint, episodes=2, device=device, seed=10000,
                                           sensor_failure_probability=probability,
                                           use_previous_action=override, output_dir=directory)
                result = json.loads((evaluated / "evaluation.json").read_text())
                self.assertGreaterEqual(result["mean_length"], 1)
                self.assertLessEqual(result["mean_length"], 5)
                self.assertEqual(result["parameters"]["sensor_failure_probability"], probability)
                self.assertEqual(result["parameters"]["training_sensors"]["mode"], mode)
                self.assertEqual(result["parameters"]["effective_use_previous_action"], previous_action if override is None else override)
                self.assertEqual([e["seed"] for e in result["episodes"]], [10000, 10001])
            if mode == "curriculum":
                # 随机动作和中间失灵概率也必须在相同种子下可复现。
                results = []
                for _ in range(2):
                    evaluated = run_evaluation(checkpoint, episodes=2, device=device, policy="sample",
                                               sensor_failure_probability=0.5, output_dir=directory)
                    results.append(json.loads((evaluated / "evaluation.json").read_text())["episodes"])
                self.assertEqual(results[0], results[1])
            if video:
                import imageio.v2 as imageio
                evaluated = run_evaluation(checkpoint, episodes=1, device=device, mode="video", output_dir=directory)
                videos = list((evaluated / "videos").glob("*.mp4"))
                self.assertEqual(len(videos), 1)
                with imageio.get_reader(videos[0]) as reader:
                    self.assertEqual(reader.get_data(0).shape[-1], 3)

    def test_cpu_all_sensor_modes(self):
        for mode in ("curriculum", "full", "partial"):
            with self.subTest(mode=mode):
                self._closed_loop(mode, video=mode == "curriculum")

    def test_cpu_previous_action_disabled(self):
        self._closed_loop("curriculum", previous_action=False)

    def test_cpu_separate_backbones(self):
        for mode in ("full", "partial"):
            with self.subTest(mode=mode):
                self._closed_loop(mode, backbone_mode="separate", previous_action=mode == "full")

    @unittest.skipUnless(torch.cuda.is_available(), "需要 GPU 容器")
    def test_cuda_real_ppo(self):
        self._closed_loop("curriculum", device="cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "需要 GPU 容器")
    def test_cuda_separate_backbones_real_ppo(self):
        self._closed_loop("curriculum", device="cuda", backbone_mode="separate")

    def test_raw_environment_has_all_eight_channels(self):
        env = make_env(max_episode_steps=5)
        try:
            obs, _ = env.reset(seed=4)
            self.assertEqual(obs.shape, (8,))
            self.assertEqual(env.observation_space.shape, (8,))
        finally:
            env.close()
