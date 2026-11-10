"""Exact episode scheduling, policy-only equivalence, seeded sampling and resource cleanup."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from evaluate import evaluate_agent
from evaluation_runner import EvaluationRunner, PolicySnapshot, episode_outcome
from fixtures import small_config
from model import Agent, cache_tensors, make_model_input
from runtime import configure_runtime


class VariableEnv(gym.Env):
    def __init__(self, fail=False):
        self.closed, self.fail = False, fail

    def observation(self):
        return np.array([self.seed % 7, self.length, self.action_sum, -.25, .4, -.3, 0, 1], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.seed, self.length, self.action_sum = seed, 0, 0
        self.limit = 2 + seed % 3
        return self.observation(), {}

    def step(self, action):
        if self.fail:
            raise RuntimeError("injected environment failure")
        self.length += 1
        self.action_sum += action
        done = self.length == self.limit
        reward = [200, 199, 201][self.seed % 3] if done else 0
        return self.observation(), reward, done and self.seed % 2 == 0, done and self.seed % 2 == 1, {}

    def close(self):
        self.closed = True


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(9, "cpu")

    def test_policy_only_matches_logits_and_cache_with_async_resets(self):
        for mode in ("shared", "separate"):
            agent = Agent(replace(small_config().agent, backbone_mode=mode))
            policy = PolicySnapshot(agent)
            full_cache, policy_cache = agent.create_cache(3), policy.backbone.create_cache(3)
            for step in range(11):
                observations = torch.randn(3, 16)
                previous = torch.tensor([step % 4, (step + 1) % 4, -1])
                starts = torch.tensor([step % 3 == 0, step % 5 == 0, step == 0])
                logits, _ = agent.step_outputs(observations, previous, full_cache, starts)
                actual = policy.step(make_model_input(observations, previous), policy_cache, starts)
                torch.testing.assert_close(actual, logits, rtol=0, atol=0)
                expected_cache = full_cache.policy if mode == "separate" else full_cache
                for name, value in cache_tensors(policy_cache).items():
                    torch.testing.assert_close(value, cache_tensors(expected_cache)[name], rtol=0, atol=0)
            self.assertFalse(hasattr(policy, "critic"))
            self.assertFalse(hasattr(policy, "value_backbone"))

    def test_parallelism_does_not_change_episode_seeds_actions_or_counts(self):
        for mode in ("shared", "separate"):
            agent = Agent(replace(small_config().agent, backbone_mode=mode))
            with torch.no_grad():
                agent.actor.weight.zero_()
                agent.actor.bias.copy_(torch.tensor([.1, .2, .3, .4]).log())
            for policy in ("sample", "argmax"):
                for probability in (0, .5, 1):
                    results = [evaluate_agent(agent, episodes=7, seed=10000, num_envs=count,
                                              sensor_failure_probability=probability, policy=policy,
                                              env_factory=lambda *a: VariableEnv()) for count in (1, 3, 8)]
                    self.assertEqual(results[0]["episodes"], results[1]["episodes"])
                    self.assertEqual(results[0]["episodes"], results[2]["episodes"])
                    for result in results:
                        self.assertEqual([row["seed"] for row in result["episodes"]], list(range(10000, 10007)))
                        self.assertAlmostEqual(sum(result[f"action_fraction_{i}"] for i in range(4)), 1)
                        self.assertEqual(sum(row["length"] for row in result["episodes"]),
                                         sum(sum(row[f"action_count_{i}"] for i in range(4)) for row in result["episodes"]))

    def test_score_boundary_and_terminal_classification(self):
        agent = Agent(small_config().agent)
        result = evaluate_agent(agent, episodes=3, seed=0, num_envs=8, env_factory=lambda *a: VariableEnv())
        self.assertEqual(result["mean_return"], 200)
        self.assertEqual(result["score_200_rate"], 2 / 3)
        self.assertEqual(result["timeout_rate"], 1 / 3)
        self.assertEqual(episode_outcome(True, False, -100), "terminal_penalty")
        self.assertEqual(episode_outcome(True, False, 100), "positive_terminal")
        self.assertEqual(episode_outcome(False, True, -100), "timeout")

    def test_reuse_syncs_policy_without_recreating_environments(self):
        agent = Agent(small_config().agent).train()
        created = []
        def factory(*args):
            env = VariableEnv()
            created.append(env)
            return env
        runner = EvaluationRunner(agent, num_envs=3, env_factory=factory)
        try:
            runner.evaluate(agent, episodes=7, policy="argmax")
            with torch.no_grad():
                agent.actor.weight.zero_()
                agent.actor.bias.copy_(torch.tensor([100., 0, 0, 0]))
            result = runner.evaluate(agent, episodes=7, policy="argmax")
            self.assertEqual(len(created), 3)
            self.assertEqual(result["action_fraction_0"], 1)
            self.assertTrue(agent.training)
            self.assertGreater(runner.last_performance["evaluation_seconds"], 0)
            self.assertEqual(runner.last_performance["evaluation_environment_steps"], sum(row["length"] for row in result["episodes"]))
        finally:
            runner.close()
        self.assertTrue(all(env.closed for env in created))

    def test_all_environments_close_on_step_creation_and_callback_errors(self):
        agent = Agent(small_config().agent)
        for failure in ("step", "creation", "callback"):
            created = []
            def factory(*args):
                if failure == "creation" and created:
                    raise RuntimeError("injected creation failure")
                env = VariableEnv(fail=failure == "step")
                created.append(env)
                return env
            def callback(row):
                raise RuntimeError("injected callback failure")
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                evaluate_agent(agent, episodes=7, num_envs=3, env_factory=factory,
                               on_episode=callback if failure == "callback" else None)
            self.assertTrue(created)
            self.assertTrue(all(env.closed for env in created))

    def test_human_mode_uses_one_environment_and_value_branch_is_unused(self):
        agent = Agent(replace(small_config().agent, backbone_mode="separate"))
        with patch.object(agent.critic, "forward", side_effect=AssertionError("critic used")), \
                patch.object(agent.value_backbone, "step", side_effect=AssertionError("value used")):
            runner = EvaluationRunner(agent, num_envs=8, mode="human", env_factory=lambda *a: VariableEnv())
            try:
                runner.evaluate(agent, episodes=2)
                self.assertEqual(runner.last_performance["evaluation_num_envs"], 1)
            finally:
                runner.close()
