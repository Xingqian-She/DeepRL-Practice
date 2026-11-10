"""编译预热的状态隔离、真实 Inductor 数值对照和连续 PPO。"""

import copy
from dataclasses import replace
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import acceleration
from acceleration import configure_acceleration
from model import Agent, cache_tensors
from ppo import PPOTrainer
from rollout import Collector
from runtime import configure_runtime, load_checkpoint, save_checkpoint
from train import make_parser
from fixtures import fixture_vector, small_config


class AccelerationTests(unittest.TestCase):
    def test_modes_and_cpu_default(self):
        device = configure_runtime(3, "cpu")
        config = small_config()
        self.assertEqual(make_parser().parse_args([]).compile_mode, "auto")
        self.assertEqual(make_parser().parse_args(["--compile", "off"]).compile_mode, "off")
        with self.assertRaises(ValueError):
            replace(config, compile_mode="invalid")
        agent = Agent(config.agent)
        with patch.object(torch, "compile", side_effect=AssertionError("CPU must stay eager")):
            status = configure_acceleration(agent, config, device)
            self.assertFalse(status["enabled"])
            self.assertIn("Linux/CUDA", status["reason"])
            self.assertFalse(configure_acceleration(agent, replace(config, compile_mode="off"), device)["enabled"])
            with self.assertRaisesRegex(RuntimeError, "Linux/CUDA"):
                configure_acceleration(agent, replace(config, compile_mode="on"), device)
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)

    def test_warmup_failure_restores_rng_parameters_and_clears_functions(self):
        device = configure_runtime(3, "cpu")
        agent = Agent(small_config().agent).eval()
        original = copy.deepcopy(agent.state_dict())
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()

        def failing_warmup(*args):
            random.random()
            np.random.random()
            torch.rand(4)
            next(agent.parameters()).grad = torch.ones_like(next(agent.parameters()))
            agent.train()
            raise RuntimeError("injected compiler failure")

        with patch.object(acceleration, "_unsupported_reason", return_value=None), \
                patch.object(torch, "compile", side_effect=lambda fn, **kwargs: fn), \
                patch.object(acceleration, "_warmup", side_effect=failing_warmup):
            for mode in ("auto", "on"):
                with self.subTest(mode=mode):
                    if mode == "on":
                        with self.assertRaisesRegex(RuntimeError, "injected compiler failure"):
                            configure_acceleration(agent, small_config(compile_mode=mode), device)
                    else:
                        status = configure_acceleration(agent, small_config(compile_mode=mode), device)
                        self.assertFalse(status["enabled"])
                        self.assertIn("injected compiler failure", status["reason"])
                    self.assertIsNone(agent._compiled_step)
                    self.assertIsNone(agent._compiled_sequence)
                    self.assertFalse(agent.training)
                    self.assertTrue(all(p.grad is None for p in agent.parameters()))
                    self.assertEqual(python_rng, random.getstate())
                    self.assertEqual(numpy_rng[0], np.random.get_state()[0])
                    np.testing.assert_array_equal(numpy_rng[1], np.random.get_state()[1])
                    self.assertEqual(numpy_rng[2:], np.random.get_state()[2:])
                    torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)
                    for name, value in agent.state_dict().items():
                        torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "需要 GPU 容器")
    def test_real_inductor_outputs_gradients_cache_and_ppo(self):
        for backbone_mode, previous_action in (("shared", True), ("shared", False),
                                               ("separate", True), ("separate", False)):
            with self.subTest(backbone_mode=backbone_mode, previous_action=previous_action):
                device = configure_runtime(3, "cuda")
                config = small_config(device="cuda", compile_mode="on")
                config = replace(config, agent=replace(config.agent, use_previous_action=previous_action,
                                                       backbone_mode=backbone_mode))
                eager = Agent(config.agent).to(device)
                compiled = copy.deepcopy(eager)
                expected_state = copy.deepcopy(compiled.state_dict())
                rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                status = configure_acceleration(compiled, config, device)
                self.assertTrue(status["enabled"])
                self.assertFalse(status["cuda_graphs"])
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                torch.testing.assert_close(cuda_rng, torch.cuda.get_rng_state(), rtol=0, atol=0)
                for name, value in compiled.state_dict().items():
                    torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in compiled.parameters()))

                # 多次回绕、单环境重置和非零历史门控，比较整个 KV 状态。
                left, right = eager.create_cache(2), compiled.create_cache(2)
                for step in range(12):
                    obs = torch.randn(2, 16, device=device)
                    previous = torch.tensor([-1 if step % 3 == 0 else 0, 2], device=device)
                    starts = torch.tensor([step % 3 == 0, step == 0], device=device)
                    for actual, reference in zip(compiled.step_outputs(obs, previous, right, starts),
                                                 eager.step_outputs(obs, previous, left, starts)):
                        torch.testing.assert_close(actual, reference, rtol=5e-4, atol=5e-5)
                    for name, tensor in cache_tensors(right).items():
                        torch.testing.assert_close(tensor, cache_tensors(left)[name], rtol=5e-4, atol=5e-5)

                # 已有 fixture 包含异步终止/截断；跨更新重建缓存并更换序列形状。
                envs = fixture_vector()
                try:
                    collector = Collector(envs, compiled, device, config.seed, sensor_config=config.sensors,
                                          total_timesteps=config.effective_total_timesteps)
                    trainer = PPOTrainer(compiled, config)
                    for update in range(2):
                        buffer = collector.collect(config.num_steps)
                        batch = next(buffer.batches(4 if update == 0 else 3, 3, shuffle=False))
                        eager.load_state_dict(compiled.state_dict())
                        outputs, gradients = [], []
                        for agent in (eager, compiled):
                            agent.zero_grad(set_to_none=True)
                            out = agent.evaluate_actions(batch)
                            # 只选 loss_mask；预热、padding 和历史梯度实际参与图构建。
                            sum(value[batch["loss_mask"]].square().mean() for value in out).backward()
                            outputs.append(tuple(value.detach() for value in out))
                            gradients.append([p.grad.clone() for p in agent.parameters()])
                        for actual, reference in zip(outputs[1] + tuple(gradients[1]),
                                                     outputs[0] + tuple(gradients[0])):
                            torch.testing.assert_close(actual, reference, rtol=5e-4, atol=5e-5)
                        metrics = trainer.update(buffer)
                        self.assertLess(metrics["consistency_max_logprob_error"], 1e-4)
                        collector.invalidate_cache()
                    with tempfile.TemporaryDirectory() as directory:
                        path = directory + "/latest.pt"
                        save_checkpoint(path, compiled, config, collector.global_step)
                        loaded, saved = load_checkpoint(path, device)
                        self.assertEqual(saved["format_version"], 2)
                        self.assertTrue(saved["acceleration"]["enabled"])
                        self.assertIsNone(loaded._compiled_step)
                        self.assertIsNone(loaded._compiled_sequence)
                        self.assertEqual(set(loaded.state_dict()), set(compiled.state_dict()))
                finally:
                    envs.close()

                # 正式开始后不捕获编译异常；特别不能重试会修改在线 cache 的调用。
                compiled._compiled_step = lambda *args: (_ for _ in ()).throw(RuntimeError("late failure"))
                with self.assertRaisesRegex(RuntimeError, "late failure"):
                    compiled.step_outputs(obs, previous, right, starts)
