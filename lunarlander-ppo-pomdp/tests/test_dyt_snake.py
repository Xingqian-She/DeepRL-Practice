"""DyT/Snake project contracts, FP32 policy and historical reference integrity."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from acceleration import configure_acceleration
from config import AgentConfig
from dyt_snake_comparison import normalized_recipe, write_comparison
from evaluation_runner import PolicySnapshot
from fixtures import small_config
from model import Agent, cache_tensors
import run_experiments as runner
from runtime import configure_runtime, EVALUATION_PROTOCOL, PRECISION
from summarize_experiments import training_arguments
from test_experiments import EvidenceExecutor
import test_encoder_decoder as ed_tests
from train import make_parser


class DyTSnakeIntegrationTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(3, "cpu")
        config = small_config(compile_mode="off")
        self.config = replace(config, agent=replace(config.agent, backbone_mode="encoder_decoder",
            decoder_layers=2, norm_type="dyt", ffn_type="swi_snake_glu"))

    # Reuse behavioral integration cases against the new components.
    test_history_gradient_and_value_isolation = ed_tests.EncoderDecoderAgentTests.test_actor_history_gradient_value_isolation_and_burnin_graph
    test_stream_snapshot_and_cache = ed_tests.EncoderDecoderAgentTests.test_step_sequence_policy_snapshot_and_all_cache_tensors_match
    test_ppo_rebuild_and_value_peek = ed_tests.EncoderDecoderAgentTests.test_two_ppo_updates_rebuild_and_readonly_value_peek
    test_async_reset_and_bootstrap = ed_tests.EncoderDecoderAgentTests.test_async_resets_truncation_and_bootstrap_match_sequence_values
    test_checkpoint_roundtrip = ed_tests.EncoderDecoderAgentTests.test_v2_checkpoint_roundtrip_and_legacy_missing_decoder_field

    def test_recipe_cli_roundtrip_and_legacy_defaults(self):
        plan = runner.build_plan(runner.PROJECT / "dyt_snake_experiments.json")
        self.assertEqual(len(plan["tasks"]), 42)
        original = runner.build_plan(runner.PROJECT / "encoder_decoder_experiments.json")
        self.assertEqual(normalized_recipe(plan["tasks"][0]["parameters"]),
                         normalized_recipe(original["tasks"][0]["parameters"]))
        arguments = training_arguments(plan["tasks"][0]["parameters"])
        parsed = vars(make_parser().parse_args(arguments))
        for name in ("norm_type", "ffn_type", "qk_norm", "final_norm", "ffn_dim", "snake_eps"):
            self.assertEqual(parsed[name], plan["tasks"][0]["parameters"]["agent"][name])
        self.assertEqual(AgentConfig(**{"hidden_dim": 32}).norm_type, "rmsnorm")
        self.assertEqual(AgentConfig(**{"hidden_dim": 32}).ffn_type, "swiglu")

    def test_fp32_survives_ambient_autocast_and_updates_new_parameters(self):
        agent = Agent(self.config.agent)
        snapshot = PolicySnapshot(agent)
        cache = agent.create_cache(2)
        observations = torch.randn(2, 16)
        previous = torch.zeros(2, dtype=torch.long)
        starts = torch.ones(2, dtype=torch.bool)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            logits, values = agent.step_outputs(observations, previous, cache, starts)
            from model import make_model_input
            copied = snapshot.step(make_model_input(observations, previous), snapshot.backbone.create_cache(2), starts)
            sequence = agent.forward_sequence(observations[:, None], previous[:, None],
                episode_starts=starts[:, None], valid_mask=starts[:, None],
                position_ids=torch.zeros(2, 1, dtype=torch.long))
        for value in (logits, values, copied, *sequence):
            self.assertEqual(value.dtype, torch.float32)
        torch.testing.assert_close(logits, copied)
        self.assertEqual(cache.dtype, torch.float32)
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
        self.assertFalse(torch.backends.cudnn.allow_tf32)
        new = {name: p for name, p in agent.named_parameters() if "log_alpha" in name or name.endswith("_norm.alpha")}
        before = {name: p.detach().clone() for name, p in new.items()}
        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        # More than one key is needed for Q/K slopes to affect softmax.
        sequence_starts = torch.zeros(2, 4, dtype=torch.bool)
        sequence_starts[:, 0] = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            training_outputs = agent.forward_sequence(torch.randn(2, 4, 16), torch.zeros(2, 4, dtype=torch.long),
                episode_starts=sequence_starts, valid_mask=torch.ones_like(sequence_starts),
                position_ids=torch.arange(4).expand(2, 4))
        sum(value.square().sum() for value in training_outputs).backward()
        optimizer.step()
        self.assertTrue(all(torch.isfinite(p.grad).all() and p.grad.dtype == torch.float32 for p in new.values()))
        self.assertTrue(all(not torch.equal(p, before[name]) for name, p in new.items()))


class HistoricalComparisonTests(unittest.TestCase):
    def test_registered_reference_missing_incomplete_and_corrupt_artifacts(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = [base / "historical", base / "new"]
            states = []
            environment = dict(precision=PRECISION, device="cpu", evaluation_protocol=EVALUATION_PROTOCOL,
                               dependencies={}, hardware={})
            for root, filename in zip(roots, ("encoder_decoder_experiments.json", "dyt_snake_experiments.json")):
                plan = runner.build_plan(runner.PROJECT / filename, smoke=True)
                # Controlled small evidence, explicitly marked non-smoke to exercise reference validation.
                plan["contract"]["smoke"] = False
                with redirect_stdout(io.StringIO()):
                    runner.run_batch(root, plan, runner.Cancellation(), executor=EvidenceExecutor(root),
                                     environment=environment, source=root.name)
                states.append(runner.load_state(root))
            state = states[1]
            missing = write_comparison(roots[1], state, base / "missing")
            self.assertFalse(missing["complete"])
            self.assertTrue(missing["issues"])
            result = write_comparison(roots[1], state, roots[0])
            self.assertTrue(result["complete"], result["issues"])
            self.assertEqual(len(result["rows"]), 72)
            self.assertEqual(len(result["differences"]), 36)
            self.assertEqual(len(result["training"]), 12)
            self.assertEqual(result["source_fingerprints"], {"new": "new", "historical": "historical"})
            partial = deepcopy(state)
            next(t for t in partial["tasks"].values() if t["spec"]["kind"] == "evaluate")["status"] = "pending"
            self.assertFalse(write_comparison(roots[1], partial, roots[0])["complete"])
            target = next(roots[0].glob("tasks/eval--*/attempt-*/run/evaluation.json"), None)
            if target is None:
                task = next(t for t in states[0]["tasks"].values() if t["spec"]["kind"] == "evaluate")
                target = runner.attempt_path(roots[0], runner.current_attempt(task)) / "run/evaluation.json"
            target.write_text("{}", encoding="utf-8")
            rejected = write_comparison(roots[1], state, roots[0])
            self.assertFalse(rejected["complete"])
            self.assertEqual(rejected["rows"], [])
            self.assertTrue(rejected["issues"])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class DyTSnakeCUDATests(unittest.TestCase):
    def test_compiled_fp32_gradients_and_streaming_cache(self):
        device = configure_runtime(3, "cuda")
        config = small_config(device="cuda", compile_mode="on")
        config = replace(config, agent=replace(config.agent, backbone_mode="encoder_decoder",
            decoder_layers=2, norm_type="dyt", ffn_type="swi_snake_glu"))
        eager = Agent(config.agent).to(device)
        compiled = deepcopy(eager)
        status = configure_acceleration(compiled, config, device)
        self.assertTrue(status["enabled"])
        left, right = eager.create_cache(2), compiled.create_cache(2)
        for step in range(10):
            observations = torch.randn(2, 16, device=device)
            previous = torch.zeros(2, dtype=torch.long, device=device)
            starts = torch.tensor([step % 3 == 0, step == 0], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                actual = compiled.step_outputs(observations, previous, right, starts)
            expected = eager.step_outputs(observations, previous, left, starts)
            for a, b in zip(actual, expected):
                self.assertEqual(a.dtype, torch.float32)
                torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)
            for name, value in cache_tensors(right).items():
                torch.testing.assert_close(value, cache_tensors(left)[name], rtol=5e-4, atol=5e-5)
        from acceleration import _sequence_inputs
        inputs = _sequence_inputs(config, 2, 13, device)
        expected = eager._sequence_core(*inputs)
        actual = compiled._compiled_sequence(*inputs)
        sum(v.square().mean() for v in expected).backward()
        sum(v.square().mean() for v in actual).backward()
        for a, b in zip((*actual, *(p.grad for p in compiled.parameters())),
                        (*expected, *(p.grad for p in eager.parameters()))):
            torch.testing.assert_close(a, b, rtol=5e-4, atol=5e-5)


if __name__ == "__main__":
    unittest.main()
