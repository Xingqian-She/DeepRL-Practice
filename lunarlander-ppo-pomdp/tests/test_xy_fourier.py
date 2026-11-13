"""XY integration: state reconstruction, cached inference, checkpoints and task contracts."""

from copy import deepcopy
from dataclasses import replace
import math
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from acceleration import _sequence_inputs
from config import AgentConfig
from evaluation_runner import EvaluationRunner, PolicySnapshot
from model import Agent, cache_tensors, make_model_input
from ppo import check_consistency
from rollout import Collector
from run_experiments import PROJECT, build_plan, command_for
from runtime import configure_runtime, load_checkpoint, save_checkpoint
from sensors import SensorConfig, input_spec
from train import make_parser
from xy_fourier import XYPhaseDiagnostics
from xy_fourier_comparison import write_comparison
from fixtures import fixture_vector, small_config
from test_evaluation import VariableEnv


class XYFourierTests(unittest.TestCase):
    def setUp(self):
        configure_runtime(9, "cpu")
        base = small_config(sensors=SensorConfig(mode="partial"))
        self.config = replace(base, agent=replace(base.agent, backbone_mode="separate", xy_fourier=True))

    def test_preserved_prefix_order_rng_and_coordinate_gradients(self):
        obs = torch.zeros(2, 3, 16, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            obs[..., 0] = .125
            obs[..., 1] = .25
        previous = torch.tensor([[-1, 0, 1], [2, 3, -1]])
        rng = torch.get_rng_state().clone()
        encoded = make_model_input(obs, previous, xy_fourier=True)
        self.assertEqual(encoded.shape, (2, 3, 32))
        torch.testing.assert_close(encoded[..., :20], make_model_input(obs, previous), rtol=0, atol=0)
        expected = [f(w * u) for u, ws in ((.125, (1, 2, 4, 8)), (.25, (1, 2)))
                    for w in ws for f in (lambda z: math.sin(math.pi * z), lambda z: math.cos(math.pi * z))]
        torch.testing.assert_close(encoded[0, 0, 20:], torch.tensor(expected, dtype=torch.float64))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        # Local sensitivity of the appended coordinates is independent of position.
        derivatives = [torch.autograd.grad(encoded[0, 0, i], obs, retain_graph=True)[0][0, 0, :2]
                       for i in range(20, 32)]
        norms = torch.stack(derivatives).square().sum(0).sqrt()
        torch.testing.assert_close(norms, torch.tensor([math.pi * math.sqrt(85), math.pi * math.sqrt(5)],
                                                       dtype=torch.float64))
        disabled_action = make_model_input(obs, previous, False, xy_fourier=True)
        self.assertEqual(disabled_action[..., 16:20].count_nonzero(), 0)
        torch.testing.assert_close(disabled_action[..., 20:], encoded[..., 20:], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            AgentConfig(xy_fourier=1)

    def test_phase_thresholds_terminal_substitution_and_missing_finals(self):
        phase = XYPhaseDiagnostics()
        phase.update(np.zeros((2, 8)), [[.0625, .25], [.125, .5]])
        before = np.zeros((2, 8))
        after = np.ones((2, 8)) * 1000  # resets must never be counted as displacements
        final = np.zeros(8)
        phase.vector_step(before, after, np.array([True, True]),
                          {"final_obs": np.array([final, None], dtype=object), "_final_obs": [True, False]})
        metrics = phase.metrics()
        self.assertEqual(metrics["xy_phase_transitions"], 3)
        self.assertEqual(metrics["xy_phase_skipped_terminal_transitions"], 1)
        for axis in ("x", "y"):
            self.assertEqual(metrics[f"xy_{axis}_phase_step_max_over_pi"], 1)
            self.assertEqual(metrics[f"xy_{axis}_phase_ge_half_pi_count"], 2)
            self.assertEqual(metrics[f"xy_{axis}_phase_ge_pi_count"], 1)

    @torch.no_grad()
    def test_step_sequence_policy_snapshot_and_independent_value_branch(self):
        for mode in ("shared", "separate", "encoder_decoder"):
            with self.subTest(mode=mode):
                config = replace(self.config, agent=replace(self.config.agent, backbone_mode=mode, decoder_layers=2))
                agent = Agent(config.agent)
                features, starts, valid, positions = _sequence_inputs(config, 3, 29, torch.device("cpu"))
                expected_logits, expected_values = agent._sequence_core(features, starts, valid, positions)
                cache = agent.create_cache(3)
                policy = PolicySnapshot(agent)
                policy_cache = policy.backbone.create_cache(3)
                for t in range(features.shape[1]):
                    with sdpa_kernel(SDPBackend.MATH):
                        logits, values = agent._step_core(features[:, t], cache, starts[:, t])
                    # Padding appears only at the last token and cannot affect earlier valid tokens.
                    selected = valid[:, t]
                    torch.testing.assert_close(logits[selected], expected_logits[:, t][selected], rtol=1e-4, atol=1e-6)
                    torch.testing.assert_close(values[selected], expected_values[:, t][selected], rtol=1e-4, atol=1e-6)
                    torch.testing.assert_close(policy.step(features[:, t], policy_cache, starts[:, t]), logits, rtol=0, atol=0)
                expected_cache = cache.policy if mode == "separate" else cache
                for name, tensor in cache_tensors(policy_cache).items():
                    torch.testing.assert_close(tensor, cache_tensors(expected_cache)[name], rtol=0, atol=0)
                if mode == "separate":
                    self.assertFalse({id(p) for p in agent.policy_parameters()} & {id(p) for p in agent.value_parameters()})
                    with torch.no_grad():
                        agent.value_encoder[0].bias.add_(1)
                    actual_logits, actual_values = agent._sequence_core(features, starts, valid, positions)
                    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
                    self.assertFalse(torch.equal(actual_values, expected_values))
        base = Agent(AgentConfig(backbone_mode="separate"))
        expanded = Agent(AgentConfig(backbone_mode="separate", xy_fourier=True))
        self.assertEqual(expanded.parameter_counts()["total"] - base.parameter_counts()["total"], 6144)

    def test_rollout_cross_boundary_final_states_bootstrap_and_evaluation(self):
        agent = Agent(self.config.agent)
        envs = fixture_vector()
        collector = Collector(envs, agent, torch.device("cpu"), self.config.seed,
                              sensor_config=self.config.sensors, total_timesteps=self.config.effective_total_timesteps)
        try:
            for _ in range(2):
                buffer = collector.collect(13)
                self.assertEqual(buffer.observations.shape, (13, 2, 16))
                self.assertEqual(buffer.xy_phase_metrics["xy_phase_transitions"], 26)
                self.assertEqual(buffer.xy_phase_metrics["xy_phase_skipped_terminal_transitions"], 0)
                self.assertEqual(buffer.xy_phase_metrics["xy_x_phase_step_max_over_pi"], 0)
                self.assertEqual(buffer.xy_phase_metrics["xy_y_phase_step_max_over_pi"], 2)
                self.assertEqual(buffer.next_values[buffer.terminated].count_nonzero(), 0)
                self.assertTrue((buffer.next_values[buffer.truncated].abs() > 0).all())
                report = check_consistency(agent, buffer, self.config)
                self.assertLess(report["consistency_max_logprob_error"], 1e-4)
                for batch in buffer.batches(4, 3, shuffle=False):
                    kwargs = {k: batch[k] for k in ("episode_starts", "valid_mask", "position_ids")}
                    _, full = agent.forward_sequence(batch["observations"], batch["previous_actions"], **kwargs)
                    values = agent.values_sequence(batch["observations"], batch["previous_actions"], **kwargs)
                    torch.testing.assert_close(values, full, rtol=0, atol=0)
                collector.invalidate_cache()
        finally:
            envs.close()
        runner = EvaluationRunner(agent, num_envs=3, env_factory=lambda *a: VariableEnv())
        try:
            for _ in range(2):
                summary = runner.evaluate(agent, episodes=7)
                self.assertEqual(summary["xy_phase_transitions"], sum(e["length"] for e in summary["episodes"]))
                self.assertEqual(summary["xy_phase_skipped_terminal_transitions"], 0)
        finally:
            runner.close()

    def test_checkpoint_legacy_defaults_new_roundtrip_and_spec_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for enabled in (False, True):
                config = replace(self.config, agent=replace(self.config.agent, xy_fourier=enabled))
                agent = Agent(config.agent)
                save_checkpoint(path, agent, config, 52)
                saved = torch.load(path, weights_only=True)
                if not enabled:
                    saved["agent_config"].pop("xy_fourier")
                    saved["train_config"]["agent"].pop("xy_fourier")
                    torch.save(saved, path)
                loaded, _ = load_checkpoint(path, torch.device("cpu"))
                self.assertEqual(loaded.config.xy_fourier, enabled)
                for name, value in agent.state_dict().items():
                    torch.testing.assert_close(loaded.state_dict()[name], value, rtol=0, atol=0)
                if enabled:
                    saved["input_spec"]["xy_fourier"]["angular_frequencies"]["x"][0] = 99
                    torch.save(saved, path)
                    with self.assertRaisesRegex(ValueError, "输入布局"):
                        load_checkpoint(path, torch.device("cpu"))
        self.assertEqual(input_spec()["input_dim"], 20)
        self.assertNotIn("xy_fourier", input_spec())
        self.assertEqual(input_spec(xy_fourier=True)["input_dim"], 32)

    def test_experiment_contract_cli_and_small_overrides(self):
        plan = build_plan(PROJECT / "xy_fourier_experiments.json")
        self.assertEqual(plan["groups"], ["XY_PO01", "XY_PO03"])
        self.assertEqual(plan["contract"]["seeds"], [42, 43, 44])
        self.assertEqual(sum(t["kind"] == "train" for t in plan["tasks"]), 6)
        self.assertEqual(sum(t["kind"] == "evaluate" for t in plan["tasks"]), 36)
        fo04 = build_plan(PROJECT / "experiments.json", ["FO04"])["tasks"][0]["parameters"]
        for task in plan["tasks"]:
            if task["kind"] != "train":
                continue
            config = deepcopy(task["parameters"])
            self.assertTrue(config["agent"]["xy_fourier"])
            config["agent"]["xy_fourier"] = False
            config["sensors"], config["seed"] = fo04["sensors"], fo04["seed"]
            self.assertEqual(config, fo04)
            command = command_for(PROJECT, {}, task, Path("/tmp/xy-run"))
            parsed = make_parser().parse_args(command[3:])
            self.assertTrue(parsed.xy_fourier)
            self.assertEqual(parsed.backbone_mode, "separate")
            self.assertEqual(parsed.sensor_mode, "partial" if task["group"] == "XY_PO01" else "curriculum")
        smoke = build_plan(PROJECT / "xy_fourier_experiments.json", smoke=True)
        self.assertEqual(smoke["contract"]["training_defaults"]["total_timesteps"], 64)
        self.assertTrue(smoke["contract"]["training_defaults"]["agent"]["xy_fourier"])
        self.assertFalse(make_parser().parse_args(["--no-xy-fourier"]).xy_fourier)

    def test_historical_comparison_separates_sources_and_rejects_corruption(self):
        from run_experiments import digest, file_hash

        plan = build_plan(PROJECT / "xy_fourier_experiments.json")
        with tempfile.TemporaryDirectory() as directory:
            old_root, new_root = Path(directory) / "historical", Path(directory) / "new"
            states = []
            for root, historical in ((old_root, True), (new_root, False)):
                state = {"contract": plan["contract"], "source_fingerprint": "old-source" if historical else "new-source",
                         "tasks": {}}
                for spec in deepcopy(plan["tasks"]):
                    if historical:
                        spec["key"] = spec["key"].replace("XY_PO", "PO")
                        spec["group"] = spec["group"].replace("XY_PO", "PO")
                        if spec["kind"] == "train":
                            spec["parameters"]["agent"].pop("xy_fourier")
                            spec["parameters"]["agent"].pop("decoder_layers")
                    run = root / spec["key"] / "run"
                    run.mkdir(parents=True)
                    metadata = {"parameters": spec["parameters"], "status": "completed", "run_id": spec["key"],
                                "experiment_id": spec["group"], "source_fingerprint": state["source_fingerprint"],
                                "config_fingerprint": digest(spec["parameters"]),
                                **{key: "same" for key in ("precision", "device", "dependencies", "hardware", "evaluation_protocol")}}
                    (run / "metadata.json").write_text(json.dumps(metadata))
                    hashes = {"metadata.json": file_hash(run / "metadata.json")}
                    if spec["kind"] == "evaluate":
                        (run / "evaluation.json").write_text(json.dumps({"mean_return": spec["seed"], "score_200_rate": .5}))
                        hashes["evaluation.json"] = file_hash(run / "evaluation.json")
                    state["tasks"][spec["key"]] = {"spec": spec, "status": "completed", "current_attempt": 1,
                        "attempts": [{"number": 1, "path": spec["key"], "run_id": spec["key"], "artifact_hashes": hashes}]}
                (root / "state.json").write_text(json.dumps(state))
                states.append(state)
            result = write_comparison(new_root, states[1], old_root)
            self.assertTrue(result["complete"], result["issues"])
            self.assertEqual(len(result["rows"]), 72)
            self.assertEqual(result["source_fingerprints"], {"new": "new-source", "historical": "old-source"})
            target = next(old_root.glob("eval--*/run/evaluation.json"))
            target.write_text("{}")
            rejected = write_comparison(new_root, states[1], old_root)
            self.assertFalse(rejected["complete"])
            self.assertEqual(rejected["rows"], [])
            self.assertIn("hash mismatch", rejected["issues"][0])


if __name__ == "__main__":
    unittest.main()
