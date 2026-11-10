"""Fixed schedules and evaluation-gated stages; no implicit budget extension."""

import math


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
