"""Evidence grouping, seed completeness, deterministic selection and held-out separation."""

import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from fixtures import small_config
from runtime import EVALUATION_PROTOCOL, fingerprint
from sensors import SensorConfig
from summarize_experiments import summarize, training_arguments
from train import make_parser


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class SummaryTests(unittest.TestCase):
    def fake_train(self, root, condition, seed, score=220, rate=.9, **changes):
        config = replace(small_config(), sensors=SensorConfig(mode="full"), seed=seed, **changes)
        parameters = asdict(config)
        path = root / condition / f"seed{seed}" / "train"
        path.mkdir(parents=True)
        (path / "checkpoints").mkdir()
        (path / "checkpoints/latest.pt").touch()
        metadata = {"run_id": f"{condition}-{seed}", "experiment_id": condition, "kind": "train",
                    "status": "completed", "parameters": parameters,
                    "config_fingerprint": fingerprint(parameters), "source_fingerprint": "same-source",
                    "evaluation_protocol": EVALUATION_PROTOCOL, "effective_total_timesteps": config.effective_total_timesteps,
                    "training_budget_completed": True}
        (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        write_csv(path / "evaluation.csv", [{"update": update, "mean_return": score, "score_200_rate": rate,
                                             "timeout_rate": .1, "evaluation_protocol": EVALUATION_PROTOCOL}
                                            for update in (1, 2)])
        write_csv(path / "training.csv", [{"optimizer_steps": 4, "unique_sample_coverage": 1}])
        write_csv(path / "episodes.csv", [{"return": 100}])
        return path, metadata

    def test_selection_uses_training_seeds_gate_late_scores_and_frozen_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition, scores in (("FO01", [220, 210, -50]), ("FO02", [240, 230, -20])):
                for seed, score in zip((42, 43, 44), scores):
                    self.fake_train(root, condition, seed, score=score, rate=.9 if score >= 200 else 0)
            result = summarize([root], root / "summary", conditions=["FO01", "FO02"])
            self.assertTrue(result["selection"]["eligible"])
            self.assertEqual(result["selection"]["experiment_id"], "FO02")
            self.assertEqual(result["groups"][0]["qualifying_seeds"], 2)
            self.assertAlmostEqual(result["groups"][0]["mean_return_mean"], (220 + 210 - 50) / 3)
            arguments = result["selection"]["arguments"]
            parsed = make_parser().parse_args(arguments)
            self.assertEqual(parsed.diagnostic_interval, small_config().diagnostic_interval)
            self.assertEqual(parsed.alpha_init, small_config().agent.alpha_init)
            self.assertNotIn("--total-timesteps", arguments)
            self.assertNotIn("--seed", arguments)
            self.assertTrue(all((root / "summary" / name).is_file()
                                for name in ("summary.json", "selection.json", "summary.md", "runs.csv", "groups.csv")))

    def test_missing_duplicate_and_mixed_configuration_are_not_pooled(self):
        for problem in ("missing", "duplicate", "mixed"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for seed in (42, 43) if problem == "missing" else (42, 43, 44):
                    self.fake_train(root, "FO01", seed, value_clip_coef=.2 if problem == "mixed" and seed == 44 else 0)
                if problem == "duplicate":
                    shutil.copytree(root / "FO01/seed42/train", root / "FO01/seed42/duplicate")
                result = summarize([root], root / "summary")
                self.assertFalse(result["groups"][0]["valid"])
                self.assertIsNone(result["groups"][0]["mean_return_mean"])
                self.assertFalse(result["selection"]["eligible"])
                self.assertTrue(any(problem in issue or "inconsistent" in issue for issue in result["groups"][0]["issues"]))

    def test_missing_condition_failed_run_and_unreadable_metadata_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43, 44):
                self.fake_train(root, "FO01", seed)
            result = summarize([root], root / "summary", conditions=["FO01", "FO02"])
            self.assertEqual(result["missing_conditions"], ["FO02"])
            self.assertFalse(result["selection"]["eligible"])
            path = root / "FO01/seed42/train/metadata.json"
            metadata = json.loads(path.read_text())
            metadata["status"] = "failed"
            path.write_text(json.dumps(metadata))
            result = summarize([root], root / "summary")
            self.assertIn("status=failed", result["groups"][0]["issues"])
            path.write_text("{")
            result = summarize([root], root / "summary")
            self.assertEqual(len(result["unreadable"]), 1)
            self.assertFalse(result["selection"]["eligible"])

    def test_heldout_episodes_are_not_training_seeds_and_never_select_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43, 44):
                train, training = self.fake_train(root / "training", "FO01", seed)
                path = root / "heldout" / f"seed{seed}"
                path.mkdir(parents=True)
                parameters = {"seed": 20000, "training_seed": seed, "training_parameters": training["parameters"],
                              "sensor_failure_probability": 1, "policy": "sample", "episodes": 100,
                              "checkpoint": str(train / "checkpoints/latest.pt"), "checkpoint_name": "latest.pt"}
                provenance = {"source_fingerprint": "same-source", "run_id": training["run_id"]}
                metadata = {"kind": "evaluate", "status": "completed", "experiment_id": "FO01",
                            "parameters": parameters, "config_fingerprint": fingerprint(parameters),
                            "source_fingerprint": "same-source", "evaluation_protocol": EVALUATION_PROTOCOL,
                            "training_provenance": provenance, "checkpoint_global_step": 52}
                (path / "metadata.json").write_text(json.dumps(metadata))
                (path / "evaluation.json").write_text(json.dumps({"mean_return": seed, "score_200_rate": .5,
                    "timeout_rate": .1, "training_provenance": provenance, "evaluation_protocol": EVALUATION_PROTOCOL}))
            result = summarize([root / "heldout"], root / "summary")
            self.assertTrue(result["groups"][0]["valid"])
            self.assertEqual(result["groups"][0]["seeds"], [42, 43, 44])
            self.assertEqual(result["groups"][0]["mean_return_mean"], 43)
            self.assertFalse(result["selection"]["eligible"])

    def test_source_or_budget_drift_blocks_cross_condition_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43, 44):
                self.fake_train(root, "FO01", seed)
                self.fake_train(root, "FO02", seed, total_timesteps=104)
            result = summarize([root], root / "summary")
            self.assertTrue(all(group["valid"] for group in result["groups"]))
            self.assertFalse(result["selection"]["eligible"])

    def test_heldout_label_cannot_replace_missing_requested_training_condition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (42, 43, 44):
                self.fake_train(root, "FO01", seed)
            path = root / "leftover-evaluation"
            path.mkdir()
            parameters = {"seed": 20000, "training_seed": 42, "training_parameters": {},
                          "sensor_failure_probability": 0, "policy": "sample"}
            metadata = {"kind": "evaluate", "status": "completed", "experiment_id": "FO02",
                        "parameters": parameters, "config_fingerprint": fingerprint(parameters),
                        "source_fingerprint": "same-source", "evaluation_protocol": EVALUATION_PROTOCOL}
            (path / "metadata.json").write_text(json.dumps(metadata))
            result = summarize([root], root / "summary", conditions=["FO01", "FO02"])
            self.assertEqual(result["missing_conditions"], ["FO02"])
            self.assertFalse(result["selection"]["eligible"])
