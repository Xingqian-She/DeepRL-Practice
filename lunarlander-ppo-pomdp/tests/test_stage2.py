"""Stage-two freezing, stage-one reuse, managed retries and fair held-out summaries."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import run_experiments as managed
from runtime import EVALUATION_PROTOCOL, PRECISION, dependency_version
from stage2 import runner as stage2
from summarize_experiments import write_csv
from test_experiments import EvidenceExecutor


class Stage2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.baseline = self.directory / "runs/experiments/baseline"
        self.root = self.directory / "runs/experiments/course"
        # Small real checkpoints exercise provenance/retry paths without a 10M-step fixture.
        budget = patch.object(stage2, "TRAINING_BUDGET", 64)
        budget.start()
        self.addCleanup(budget.stop)
        self.environment = {"precision": PRECISION, "device": "cpu", "evaluation_protocol": EVALUATION_PROTOCOL,
            "dependencies": {n: dependency_version(n) for n in ("torch", "gymnasium", "numpy", "box2d-py")},
            "hardware": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device_name": "cpu"}}
        self.baseline_plan = managed.build_plan(managed.PROJECT / "experiments.json", stage2.FO_GROUPS, smoke=True)
        self.baseline_plan["contract"].update(smoke=False, evaluation=deepcopy(stage2.EVALUATION))
        # Selection needs all 21 training runs, reuse needs only FO04's 18 evaluations.
        self.baseline_plan["tasks"] = [t for t in self.baseline_plan["tasks"]
                                       if t["kind"] == "train" or t["group"] == "FO04"]
        for task in self.baseline_plan["tasks"]:
            if task["kind"] == "evaluate":
                task["parameters"].update(episodes=100, num_envs=8)
        baseline_executor = EvidenceExecutor(self.baseline)

        def baseline_execute(command, log, stop):
            condition = command[command.index("--experiment-id") + 1]
            baseline_executor.score = 300 if condition == "FO04" else 220
            return baseline_executor(command, log, stop)

        self.baseline_execute = baseline_execute
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(managed.run_batch(self.baseline, self.baseline_plan, managed.Cancellation(),
                executor=baseline_execute, environment=self.environment), 0)
        self.baseline_hash = managed.file_hash(self.baseline / "state.json")
        self.executor = EvidenceExecutor(self.root)
        self.executor.score = -50

    def execute(self, command, log, stop):
        code = self.executor(command, log, stop)
        path = log.parent / "run/metadata.json"
        if path.exists():
            metadata = managed.read_json(path)
            if metadata["kind"] == "train":
                performance = metadata["parameters"]["sensors"]["schedule"] == "performance"
                metadata.update(entered_final=not performance, final_task_solved=False,
                    curriculum={"probability": 0.0 if performance else 1.0,
                                "stage_steps": [64] + [0]*10 if performance else None, "transitions": []})
                managed.write_json(path, metadata)
        return code

    def run_stage(self, **kwargs):
        options = {"executor": self.execute, "environment": self.environment, **kwargs}
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return stage2.run_stage2(self.root, self.baseline, managed.Cancellation(), **options)

    def test_default_plan_has_nine_training_and_fifty_four_evaluations(self):
        recipe = stage2.inspect_baseline(self.baseline)
        recipe["parameters"]["total_timesteps"] = 10_000_000
        plan = stage2.build_plan(recipe)
        training = [t for t in plan["tasks"] if t["kind"] == "train"]
        evaluations = [t for t in plan["tasks"] if t["kind"] == "evaluate"]
        self.assertEqual((len(training), len(evaluations)), (9, 54))
        for task in training:
            expected = deepcopy(recipe["parameters"])
            expected.update(seed=task["seed"], sensors=stage2.COURSES[task["group"]])
            self.assertEqual(task["parameters"], expected)
            self.assertNotIn("checkpoint", task["parameters"])
        self.assertEqual({t["parameters"]["seed"] for t in evaluations}, {20000})
        self.assertTrue(all(t["parameters"]["episodes"] == 100 for t in evaluations))
        self.assertEqual(managed.file_hash(self.baseline / "state.json"), self.baseline_hash)

    def test_dry_run_has_no_batch_or_freeze_side_effects(self):
        self.assertEqual(self.run_stage(dry_run=True), 0)
        self.assertFalse(self.root.exists())
        self.assertEqual(managed.file_hash(self.baseline / "state.json"), self.baseline_hash)
        self.assertEqual(self.executor.calls, [])

    def test_missing_training_and_missing_reference_are_rejected_before_target_creation(self):
        state = managed.load_state(self.baseline)
        for key in ("train--FO07--seed44", "eval--FO04--seed42--p0--sample"):
            with self.subTest(key=key):
                damaged = deepcopy(state)
                damaged["tasks"].pop(key)
                managed.write_json(self.baseline / "state.json", damaged)
                with self.assertRaisesRegex(ValueError, "缺少已完成任务"):
                    self.run_stage()
                self.assertFalse(self.root.exists())
        managed.write_json(self.baseline / "state.json", state)

    def test_corrupt_unselected_checkpoint_still_blocks_selection(self):
        task = managed.load_state(self.baseline)["tasks"]["train--FO03--seed42"]
        path = managed.attempt_path(self.baseline, managed.current_attempt(task)) / "run/checkpoints/latest.pt"
        path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "产物校验失败"):
            self.run_stage()
        self.assertFalse(self.root.exists())

    def test_ineligible_or_different_winner_is_rejected(self):
        state = managed.load_state(self.baseline)
        for seed in stage2.SEEDS:
            task = state["tasks"][f"train--FO04--seed{seed}"]
            attempt = managed.current_attempt(task)
            path = managed.attempt_path(self.baseline, attempt) / "run/evaluation.csv"
            values = managed.csv_rows(path)
            for row in values:
                row.update(mean_return=-50, score_200_rate=0)
            write_csv(path, values)
            attempt["artifact_hashes"]["evaluation.csv"] = managed.file_hash(path)
        managed.write_json(self.baseline / "state.json", state)
        with self.assertRaisesRegex(ValueError, "未按既定规则选出 FO04"):
            self.run_stage()
        self.assertFalse(self.root.exists())

    def test_source_and_environment_mismatch_do_not_freeze(self):
        with patch("runtime.source_fingerprint", return_value="changed-source"):
            with self.assertRaisesRegex(ValueError, "训练源码指纹"):
                self.run_stage()
        changed = {**self.environment, "device": "cuda"}
        with self.assertRaisesRegex(ValueError, "环境与第一阶段"):
            self.run_stage(environment=changed)
        self.assertFalse(self.root.exists())

    def test_complete_rerun_skips_all_and_comparison_reuses_exactly_twenty_one_runs(self):
        self.assertEqual(self.run_stage(), 0)
        self.assertEqual(len(self.executor.calls), 63)
        result = managed.read_json(self.root / "summary/summary.json")
        self.assertEqual(len(result["runs"]), 84)
        self.assertEqual(sum(r["origin"] == "reused" for r in result["runs"]), 21)
        self.assertEqual(len(result["groups"]), 28)
        self.assertTrue(all(g["valid"] for g in result["groups"]))
        self.assertEqual(managed.file_hash(self.baseline / "state.json"), self.baseline_hash)
        self.executor.calls.clear()
        self.assertEqual(self.run_stage(), 0)
        self.assertEqual(self.executor.calls, [])

    def test_only_evaluation_failure_retries_without_training(self):
        self.executor.fail_key = "eval--PO01--seed42--p0--sample"
        self.assertEqual(self.run_stage(groups=["PO01"]), 1)
        self.assertEqual(len(self.executor.calls), 4)
        self.executor.fail_key = None
        self.executor.calls.clear()
        self.assertEqual(self.run_stage(groups=["PO01"]), 0)
        self.assertEqual(len(self.executor.calls), 18)
        self.assertTrue(all(key.startswith("eval--") for key in self.executor.calls))

    def test_interrupted_training_uses_new_attempt_and_preserves_history(self):
        def interrupted(command, log, stop):
            stop.request("controlled interruption")
            log.write_text("interrupted")
            return 130
        self.assertEqual(self.run_stage(groups=["PO01"], executor=interrupted), 130)
        self.assertEqual(self.run_stage(groups=["PO01"]), 0)
        state = managed.load_state(self.root)
        self.assertEqual(state["tasks"]["train--PO01--seed42"]["current_attempt"], 2)
        self.assertTrue((self.root / "tasks/train--PO01--seed42/attempt-0001/console.log").is_file())
        result = managed.read_json(self.root / "summary/summary.json")
        self.assertEqual(len(result["runs"]), 42)

    def test_corrupt_new_checkpoint_retrains_only_that_seed_and_its_six_evaluations(self):
        self.assertEqual(self.run_stage(groups=["PO01"]), 0)
        (self.root / "tasks/train--PO01--seed42/attempt-0001/run/checkpoints/latest.pt").write_bytes(b"broken")
        self.executor.calls.clear()
        self.assertEqual(self.run_stage(groups=["PO01"]), 0)
        self.assertEqual(len(self.executor.calls), 7)
        self.assertTrue(all("seed42" in key for key in self.executor.calls))

    def test_adding_course_preserves_frozen_recipe_and_completed_tasks(self):
        self.assertEqual(self.run_stage(groups=["PO01"]), 0)
        frozen = managed.file_hash(self.root / "frozen-recipe.json")
        self.executor.calls.clear()
        self.assertEqual(self.run_stage(groups=["PO01", "PO02"]), 0)
        self.assertEqual(len(self.executor.calls), 21)
        self.assertTrue(all("PO02" in key for key in self.executor.calls))
        self.assertEqual(managed.file_hash(self.root / "frozen-recipe.json"), frozen)

    def test_changed_reference_attempt_requires_new_batch(self):
        self.assertEqual(self.run_stage(groups=["PO01"]), 0)
        path = self.baseline / "tasks/eval--FO04--seed42--p0--sample/attempt-0001/run/evaluation.json"
        path.write_text("{}")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(managed.run_batch(self.baseline, self.baseline_plan, managed.Cancellation(),
                executor=self.baseline_execute, environment=self.environment), 0)
        self.executor.calls.clear()
        with self.assertRaisesRegex(ValueError, "已改变|已变化"):
            self.run_stage(groups=["PO01"])
        self.assertEqual(self.executor.calls, [])

    def test_smoke_is_separate_and_excludes_formal_reference_statistics(self):
        self.assertEqual(self.run_stage(smoke=True), 0)
        result = managed.read_json(self.root / "summary/summary.json")
        self.assertEqual(result["reference_run_count"], 0)
        self.assertEqual(len(result["runs"]), 63)
        self.assertTrue(all(r["experiment_id"].startswith("PO") for r in result["runs"]))
        self.assertTrue((self.root / "smoke-recipe.json").exists())
        self.assertFalse((self.root / "frozen-recipe.json").exists())
        with self.assertRaises(ValueError):
            self.run_stage()

    def test_unsolved_performance_course_is_completed_without_claiming_final_stage(self):
        self.assertEqual(self.run_stage(groups=["PO03"]), 0)
        result = managed.read_json(self.root / "summary/summary.json")
        self.assertEqual(len(result["curriculum"]), 3)
        self.assertTrue(all(r["status"] == "completed" and r["final_probability"] == 0
                            and not r["entered_final"] and not r["final_task_solved"]
                            and sum(r["stage_steps"]) == 64 for r in result["curriculum"]))
        self.assertTrue(all(g["valid"] for g in result["groups"]))

    def test_duplicate_stage2_and_busy_baseline_are_rejected(self):
        with stage2.coordinator_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "已有调度器"):
                self.run_stage()
        with managed.batch_lock(self.baseline):
            with self.assertRaisesRegex(RuntimeError, "第一阶段批次正在"):
                self.run_stage()
        self.assertEqual(self.executor.calls, [])

    def test_analyze_only_revalidates_without_execution_and_keeps_reference(self):
        self.assertEqual(self.run_stage(groups=["PO01"]), 0)
        self.executor.calls.clear()
        self.assertEqual(self.run_stage(groups=["PO01"], analyze_only=True), 0)
        self.assertEqual(self.executor.calls, [])
        result = managed.read_json(self.root / "summary/summary.json")
        self.assertEqual(result["reference_run_count"], 21)

    def test_status_reads_saved_state_without_needing_baseline(self):
        self.assertEqual(self.run_stage(groups=["PO01"]), 0)
        with patch.object(stage2, "PROJECT", self.directory), patch.object(stage2, "inspect_baseline", side_effect=AssertionError), redirect_stdout(io.StringIO()):
            self.assertEqual(stage2.main(["--batch", "course", "--status"]), 0)

    def test_wrong_destination_and_invalid_groups_do_not_run(self):
        with self.assertRaisesRegex(ValueError, "不能与第一阶段"):
            stage2.run_stage2(self.baseline, self.baseline, managed.Cancellation())
        for groups in (["FO04"], ["PO01", "PO01"]):
            with self.assertRaises(ValueError):
                self.run_stage(groups=groups)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
