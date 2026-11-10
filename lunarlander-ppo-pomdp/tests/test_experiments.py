"""Scheduler contracts: validate evidence, isolate retries, and reap interrupted children."""

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import unittest

import torch

from config import AgentConfig, TrainConfig
from model import Agent
from runtime import EVALUATION_PROTOCOL, PRECISION, dependency_version, save_checkpoint
from sensors import SensorConfig
import run_experiments as runner


def write_csv(path, rows):
    from summarize_experiments import write_csv as write
    write(path, rows)


class EvidenceExecutor:
    """Write real loadable model checkpoints and controlled evidence, without running PPO."""

    def __init__(self, root):
        self.root, self.calls, self.fail_key = root, [], None
        self.score = 220.0
        self.invalid_success = False

    def __call__(self, command, log, cancellation):
        state = runner.load_state(self.root)
        task = next(task for task in state["tasks"].values()
                    if (runner.attempt_path(self.root, runner.current_attempt(task)) if runner.current_attempt(task) else None) == log.parent)
        spec = task["spec"]
        self.calls.append(spec["key"])
        run = log.parent / "run"
        run.mkdir()
        log.write_text("controlled test executor\n")
        if spec["key"] == self.fail_key or self.invalid_success:
            return 1 if self.fail_key else 0
        metadata = {**{key: state["environment"][key] for key in (
            "precision", "device", "dependencies", "hardware", "evaluation_protocol")},
            "kind": spec["kind"], "experiment_id": spec["group"], "status": "completed",
            "source_fingerprint": state["source_fingerprint"], "elapsed_seconds": 1.0,
            "run_id": f"{spec['key']}-attempt{task['current_attempt']}"}
        if spec["kind"] == "train":
            parameters = spec["parameters"]
            size = parameters["num_envs"] * parameters["num_steps"]
            updates = parameters["total_timesteps"] // size
            steps = updates * size
            metadata.update(parameters=parameters, config_fingerprint=runner.digest(parameters), global_step=steps,
                            effective_total_timesteps=steps, completed_updates=updates, training_budget_completed=True)
            config = {key: value for key, value in parameters.items() if key not in ("agent", "sensors")}
            config = TrainConfig(agent=AgentConfig(**parameters["agent"]), sensors=SensorConfig(**parameters["sensors"]), **config)
            agent = Agent(config.agent)
            agent.provenance = {key: metadata[key] for key in (
                "run_id", "experiment_id", "config_fingerprint", "source_fingerprint", "evaluation_protocol")}
            save_checkpoint(run / "checkpoints/latest.pt", agent, config, steps)
            write_csv(run / "training.csv", [{"update": update, "global_step": update * size,
                "optimizer_steps": 4, "unique_sample_coverage": 1} for update in range(1, updates + 1)])
            write_csv(run / "evaluation.csv", [{"update": update, "global_step": update * size,
                "mean_return": self.score, "score_200_rate": float(self.score >= 200), "timeout_rate": .1,
                "evaluation_protocol": EVALUATION_PROTOCOL} for update in range(1, updates + 1)])
            write_csv(run / "episodes.csv", [{"return": self.score}])
        else:
            train = state["tasks"][spec["depends_on"]]
            train_run = runner.attempt_path(self.root, runner.current_attempt(train)) / "run"
            train_metadata = runner.read_json(train_run / "metadata.json")
            provenance = {key: train_metadata[key] for key in (
                "run_id", "experiment_id", "config_fingerprint", "source_fingerprint", "evaluation_protocol")}
            parameters = {key: value for key, value in spec["parameters"].items() if key != "device"}
            parameters.update(checkpoint=str(train_run / "checkpoints/latest.pt"), checkpoint_name="latest.pt", mode="none",
                              effective_use_previous_action=spec["training_parameters"]["agent"]["use_previous_action"],
                              training_seed=spec["seed"], training_parameters=spec["training_parameters"])
            episodes = [{"episode": i, "seed": parameters["seed"] + i, "return": self.score, "length": 2}
                        for i in range(parameters["episodes"])]
            metadata.update(parameters=parameters, config_fingerprint=runner.digest(parameters),
                            training_provenance=provenance, completed_episodes=len(episodes),
                            checkpoint_global_step=train_metadata["global_step"], global_step=2 * len(episodes))
            runner.write_json(run / "evaluation.json", {"parameters": parameters, "episodes": episodes,
                "mean_return": self.score, "score_200_rate": float(self.score >= 200), "timeout_rate": .1,
                "evaluation_protocol": EVALUATION_PROTOCOL, "training_provenance": provenance,
                "checkpoint_global_step": train_metadata["global_step"]})
            write_csv(run / "episodes.csv", episodes)
        runner.write_json(run / "metadata.json", metadata)
        return 0


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "batch"
        self.config = runner.PROJECT / "experiments.json"
        self.plan = runner.build_plan(self.config, smoke=True)
        self.environment = {"precision": PRECISION, "device": "cpu", "evaluation_protocol": EVALUATION_PROTOCOL,
            "dependencies": {name: dependency_version(name) for name in ("torch", "gymnasium", "numpy", "box2d-py")},
            "hardware": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "device_name": "cpu"}}
        self.executor = EvidenceExecutor(self.root)

    def run_plan(self, plan=None):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.run_batch(self.root, plan or self.plan, runner.Cancellation(),
                                    executor=self.executor, environment=self.environment, source="controlled-source")

    def test_default_is_three_full_baselines_and_eighteen_heldout_tasks(self):
        plan = runner.build_plan(self.config)
        training = [task for task in plan["tasks"] if task["kind"] == "train"]
        heldout = [task for task in plan["tasks"] if task["kind"] == "evaluate"]
        self.assertEqual(plan["groups"], ["FO01"])
        self.assertEqual([task["seed"] for task in training], [42, 43, 44])
        self.assertTrue(all(task["parameters"]["total_timesteps"] == 10_000_000 for task in training))
        self.assertTrue(all(task["parameters"]["sensors"]["mode"] == "full" for task in training))
        self.assertEqual(len(heldout), 18)
        self.assertTrue(all(task["parameters"]["episodes"] == 100 for task in heldout))
        self.assertEqual({task["parameters"]["sensor_failure_probability"] for task in heldout}, {0, .5, 1})
        self.assertEqual({task["parameters"]["policy"] for task in heldout}, {"sample", "argmax"})

    def test_completed_rerun_does_no_work_and_unsolved_is_not_an_execution_failure(self):
        self.executor.score = -50
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(len(self.executor.calls), 21)
        self.executor.calls.clear()
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(self.executor.calls, [])
        summary = runner.read_json(self.root / "summary/summary.json")
        self.assertEqual(len(summary["runs"]), 21)
        self.assertTrue(all(group["valid"] for group in summary["groups"]))
        self.assertFalse(summary["selection"]["eligible"])

    def test_interrupted_training_retries_one_attempt_without_duplicate_history_in_summary(self):
        self.executor.fail_key = "train--FO01--seed42"
        self.assertEqual(self.run_plan(), 1)
        self.assertEqual(self.executor.calls, [self.executor.fail_key])
        self.executor.fail_key = None
        self.executor.calls.clear()
        self.assertEqual(self.run_plan(), 0)
        state = runner.load_state(self.root)
        self.assertEqual(state["tasks"]["train--FO01--seed42"]["current_attempt"], 2)
        self.assertTrue((self.root / "tasks/train--FO01--seed42/attempt-0001/console.log").is_file())
        summary = runner.read_json(self.root / "summary/summary.json")
        self.assertEqual(len(summary["runs"]), 21)
        self.assertTrue(all(group["valid"] for group in summary["groups"]))

    def test_evaluation_failure_does_not_retrain_completed_seeds(self):
        self.executor.fail_key = "eval--FO01--seed42--p0--sample"
        self.assertEqual(self.run_plan(), 1)
        self.assertEqual(len(self.executor.calls), 4)
        self.executor.fail_key = None
        self.executor.calls.clear()
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(len(self.executor.calls), 18)
        self.assertTrue(all(key.startswith("eval--") for key in self.executor.calls))

    def test_corrupt_weights_retrain_only_that_seed_and_invalidate_six_dependent_evaluations(self):
        self.assertEqual(self.run_plan(), 0)
        checkpoint = self.root / "tasks/train--FO01--seed42/attempt-0001/run/checkpoints/latest.pt"
        checkpoint.write_bytes(b"broken checkpoint")
        self.executor.calls.clear()
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(len(self.executor.calls), 7)
        self.assertTrue(all("seed42" in key for key in self.executor.calls))
        summary = runner.read_json(self.root / "summary/summary.json")
        self.assertEqual(len(summary["runs"]), 21)
        self.assertTrue(all(group["valid"] for group in summary["groups"]))

    def test_corrupt_result_is_invalid_in_analysis_and_only_that_evaluation_retries(self):
        self.assertEqual(self.run_plan(), 0)
        result = self.root / "tasks/eval--FO01--seed42--p0--sample/attempt-0001/run/evaluation.json"
        result.write_text("{")
        state = runner.load_state(self.root)
        runner.reconcile(self.root, state)
        with redirect_stdout(io.StringIO()):
            summary = runner.report(self.root, state)
        affected = next(group for group in summary["groups"] if group["kind"] == "evaluate"
                        and group["sensor_probability"] == 0 and group["policy"] == "sample")
        self.assertFalse(affected["valid"])
        self.assertIsNone(affected["mean_return_mean"])
        self.executor.calls.clear()
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(self.executor.calls, ["eval--FO01--seed42--p0--sample"])

    def test_complete_artifacts_survive_crash_before_state_commit(self):
        self.assertEqual(self.run_plan(), 0)
        state = runner.load_state(self.root)
        task = state["tasks"]["train--FO01--seed42"]
        task["status"] = "running"
        attempt = runner.current_attempt(task)
        attempt.pop("artifact_hashes")
        attempt.pop("run_id")
        runner.write_json(self.root / "state.json", state)
        self.executor.calls.clear()
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(self.executor.calls, [])

    def test_zero_exit_without_artifacts_is_failure(self):
        self.executor.invalid_success = True
        self.assertEqual(self.run_plan(), 1)
        self.assertEqual(len(self.executor.calls), 1)
        state = runner.load_state(self.root)
        self.assertEqual(state["tasks"]["train--FO01--seed42"]["status"], "failed")

    def test_crash_between_directory_allocation_and_state_commit_does_not_block_retry(self):
        orphan = self.root / "tasks/train--FO01--seed42/attempt-0001"
        orphan.mkdir(parents=True)
        self.assertEqual(self.run_plan(), 0)
        state = runner.load_state(self.root)
        self.assertEqual(state["tasks"]["train--FO01--seed42"]["current_attempt"], 2)
        self.assertTrue(orphan.is_dir())
        self.assertEqual(len(runner.read_json(self.root / "summary/summary.json")["runs"]), 21)

    def test_cancellation_reason_survives_reconciliation_and_can_resume(self):
        def interrupted(command, log, cancellation):
            cancellation.request("启动窗口心跳失联")
            log.write_text("interrupted before metadata\n")
            return 130

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = runner.run_batch(self.root, self.plan, runner.Cancellation(), executor=interrupted,
                                    environment=self.environment, source="controlled-source")
        self.assertEqual(code, 130)
        state = runner.load_state(self.root)
        task = state["tasks"]["train--FO01--seed42"]
        self.assertEqual(task["status"], "interrupted")
        self.assertIn("心跳失联", task["reason"])
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(runner.load_state(self.root)["tasks"]["train--FO01--seed42"]["current_attempt"], 2)

    def test_corrupt_checkpoint_without_committed_hashes_is_retried(self):
        self.assertEqual(self.run_plan(), 0)
        state = runner.load_state(self.root)
        task = state["tasks"]["train--FO01--seed42"]
        task["status"] = "running"
        attempt = runner.current_attempt(task)
        attempt.pop("artifact_hashes")
        checkpoint = runner.attempt_path(self.root, attempt) / "run/checkpoints/latest.pt"
        checkpoint.write_bytes(b"not a torch checkpoint")
        runner.write_json(self.root / "state.json", state)
        self.executor.calls.clear()
        self.assertEqual(self.run_plan(), 0)
        self.assertEqual(len(self.executor.calls), 7)

    def test_adding_groups_reuses_baseline_and_existing_task_changes_require_new_batch(self):
        self.assertEqual(self.run_plan(), 0)
        additional = runner.build_plan(self.config, ["FO01", "FO02"], smoke=True)
        self.executor.calls.clear()
        self.assertEqual(self.run_plan(additional), 0)
        self.assertEqual(len(self.executor.calls), 21)
        self.assertTrue(all("FO02" in key for key in self.executor.calls))
        changed = deepcopy(additional)
        changed["tasks"][0]["parameters"]["learning_rate"] = .123
        with self.assertRaisesRegex(ValueError, "参数已改变"):
            self.run_plan(changed)
        state = runner.load_state(self.root)
        with self.assertRaisesRegex(ValueError, "源码或运行环境"):
            runner.register_plan(state, additional, "new-source", self.environment)
        with self.assertRaisesRegex(ValueError, "源码或运行环境"):
            runner.register_plan(state, additional, "controlled-source", {**self.environment, "device": "cuda"})

    def test_batch_lock_prevents_duplicate_and_releases_after_failure(self):
        with runner.batch_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, "已有调度器"):
                with runner.batch_lock(self.root):
                    self.fail("acquired a duplicate lock")
        with runner.batch_lock(self.root):
            pass

    def test_path_traversal_and_bad_config_are_rejected(self):
        with self.assertRaises(ValueError):
            runner.attempt_path(self.root, {"path": "../../outside"})
        with self.assertRaises(ValueError):
            runner.identifier("../bad")
        config = runner.read_json(self.config)
        config["groups"]["FO01"]["training"] = {"totla_timesteps": 64}
        path = Path(self.temporary.name) / "bad.json"
        runner.write_json(path, config)
        with self.assertRaises(TypeError):
            runner.build_plan(path)

    def test_lost_heartbeat_interrupts_and_reaps_child(self):
        heartbeat = Path(self.temporary.name) / "heartbeat"
        heartbeat.write_text("one lease")
        log = Path(self.temporary.name) / "child.log"
        with redirect_stdout(io.StringIO()), runner.Cancellation(heartbeat, .25) as cancellation:
            code = runner.execute([os.sys.executable, "-u", "-c",
                "import os,time; print(os.getpid(), flush=True); time.sleep(60)"], log, cancellation)
        self.assertNotEqual(code, 0)
        self.assertIn("心跳失联", cancellation.reason)
        pid = int(log.read_text().splitlines()[0])
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_ctrl_c_and_sigterm_request_cancellation(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with runner.Cancellation() as cancellation:
                os.kill(os.getpid(), signum)
                self.assertTrue(cancellation.event.wait(.1))


if __name__ == "__main__":
    unittest.main()
