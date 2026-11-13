"""Fixed schedules and evaluation-gated stages; no implicit budget extension."""

import math


def curriculum_record(path, metadata):
    """Reusable phase/budget report; historical artifacts may lack entry metrics."""
    config, progress = metadata["parameters"], metadata.get("curriculum", {})
    phase_steps = progress.get("stage_steps")
    basis = "recorded performance stages"
    if config["sensors"]["schedule"] != "performance":
        from sensors import SensorConfig
        sensors = SensorConfig(**config["sensors"])
        completed = metadata.get("last_completed_step", metadata.get("global_step", 0))
        counts = {"p0": 0, "ramp": 0, "p1": 0}
        if sensors.mode in ("full", "partial"):
            counts["p0" if sensors.mode == "full" else "p1"] = completed
        else:
            budget = config["total_timesteps"] // (config["num_envs"] * config["num_steps"]) * config["num_envs"] * config["num_steps"]
            for step in range(0, completed, config["num_envs"]):
                probability = sensors.failure_probability(step, budget)
                counts["p0" if probability == 0 else "p1" if probability == 1 else "ramp"] += config["num_envs"]
        phase_steps, basis = counts, "derived from configured schedule and completed vector steps"
    return {"experiment_id": metadata["experiment_id"], "seed": config["seed"],
            "status": metadata["status"], "path": str(path), "global_step": metadata.get("global_step"),
            "sensor_mode": config["sensors"]["mode"], "schedule": config["sensors"]["schedule"],
            "final_probability": progress.get("probability"), "entered_final": metadata.get("entered_final"),
            "final_task_solved": metadata.get("final_task_solved"), "stage_steps": phase_steps,
            "stage_steps_basis": basis, "transitions": progress.get("transitions", []),
            "p1_full_rollout_training_steps": metadata.get("p1_training_steps"),
            "p1_entry_global_step": metadata.get("p1_entry_global_step"),
            "p1_entry_learning_rate": metadata.get("p1_entry_learning_rate"),
            "model_parameters": metadata.get("model", {}).get("parameters"),
            "rollout_cache_bytes": metadata.get("model", {}).get("rollout_cache_bytes"),
            "elapsed_seconds": metadata.get("elapsed_seconds")}


class Curriculum:
    def __init__(self, config):
        self.config = config
        self.performance = config.sensors.schedule == "performance"
        self.levels = [min(1.0, round(i * config.curriculum_step, 12))
                       for i in range(math.ceil(1 / config.curriculum_step) + 1)]
        self.stage = 0
        self.streak = 0
        self.last_eval_probability = None
        self.reached_final = config.sensors.mode == "partial"
        self.final_task_solved = False
        self.stage_steps = [0 for _ in self.levels]
        self.transitions = []

    def failure_probability(self, global_step, total_timesteps):
        if self.performance:
            return self.levels[self.stage]
        return self.config.sensors.failure_probability(global_step, total_timesteps)

    def record_steps(self, steps, global_step):
        if self.performance:
            self.stage_steps[self.stage] += steps
        if self.failure_probability(global_step, self.config.effective_total_timesteps) == 1:
            self.reached_final = True

    def observe(self, summary, global_step):
        probability = summary["sensor_failure_probability"]
        if probability != self.last_eval_probability:
            self.streak = 0
        self.last_eval_probability = probability
        qualified = (summary["mean_return"] >= self.config.curriculum_return_threshold
                     and summary["score_200_rate"] >= self.config.curriculum_success_rate)
        self.streak = self.streak + 1 if qualified else 0
        if probability == 1 and self.streak >= self.config.curriculum_patience:
            self.final_task_solved = True
        if not self.performance or probability == 1 or self.streak < self.config.curriculum_patience:
            return None
        old = self.levels[self.stage]
        self.stage += 1
        self.streak = 0
        new = self.levels[self.stage]
        self.reached_final |= new == 1
        transition = {"global_step": global_step, "from_probability": old,
                      "to_probability": new, "reason": "consecutive_evaluations_qualified"}
        self.transitions.append(transition)
        return transition

    def checkpoint_key(self, probability):
        # Continuously changing fixed curricula do not have comparable stage scores.
        if self.performance:
            return f"stage_{self.stage:02d}_p{probability:.2f}"
        if self.config.sensors.mode in ("full", "partial") or probability == 1:
            return f"p{probability:.2f}"
        return None

    def state(self, global_step):
        return {"schedule": self.config.sensors.schedule, "stage": self.stage,
                "probability": self.failure_probability(global_step, self.config.effective_total_timesteps),
                "qualifying_streak": self.streak, "reached_final": self.reached_final,
                "final_task_solved": self.final_task_solved,
                "stage_steps": list(self.stage_steps) if self.performance else None,
                "transitions": list(self.transitions)}
