"""传感器退化课程与观测编码；这里只在新观测产生时抽样。"""

from dataclasses import dataclass
import math

import numpy as np


SENSOR_NAMES = ("x", "y", "vx", "vy", "angle", "angular_velocity", "left_contact", "right_contact")
FAILURE_INDICES = (2, 3, 4, 5)
PROTECTED_INDICES = (0, 1, 6, 7)
RAW_OBSERVATION_DIM = 8
OBSERVATION_DIM = 16
MODEL_INPUT_DIM = 20


def input_spec():
    return {
        "sensor_names": list(SENSOR_NAMES),
        "failure_indices": list(FAILURE_INDICES),
        "protected_indices": list(PROTECTED_INDICES),
        "observation_dim": OBSERVATION_DIM,
        "input_dim": MODEL_INPUT_DIM,
        "layout": "masked_readings_8,availability_8,previous_action_one_hot_4",
        "availability_encoding": "1=available; 0=missing; missing_reading=0",
        "action_encoding": "one_hot_4; no_action=-1",
    }


@dataclass(frozen=True)
class SensorConfig:
    mode: str = "curriculum"
    start_fraction: float = 0.1
    end_fraction: float = 0.8
    schedule: str = "fixed"

    def __post_init__(self):
        if self.mode not in ("curriculum", "full", "partial"):
            raise ValueError("sensor mode 只能是 curriculum、full 或 partial")
        if self.schedule not in ("fixed", "performance"):
            raise ValueError("curriculum schedule 只能是 fixed 或 performance")
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                   for v in (self.start_fraction, self.end_fraction)):
            raise ValueError("课程起止比例必须是有限数值")
        if not 0 <= self.start_fraction < self.end_fraction <= 1:
            raise ValueError("课程比例须满足 0 <= start_fraction < end_fraction <= 1")

    def failure_probability(self, global_step, total_timesteps):
        if global_step < 0 or total_timesteps <= 0:
            raise ValueError("global_step 须非负，实际总步数须为正")
        if self.schedule == "performance":
            raise ValueError("performance 课程须通过 Curriculum 状态控制器计算概率")
        if self.mode == "full":
            return 0.0
        if self.mode == "partial":
            return 1.0
        progress = global_step / total_timesteps
        return min(1.0, max(0.0, (progress - self.start_fraction) /
                            (self.end_fraction - self.start_fraction)))


def validate_probability(probability):
    if isinstance(probability, bool) or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("传感器失灵概率必须是 [0, 1] 内的有限数值")


class SensorProcessor:
    def __init__(self, seed):
        # 独立随机流，不改变 Gymnasium、NumPy 全局或 PyTorch 的随机数状态。
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53454E53]))

    def process(self, observations, failure_probability):
        """(..., 8) → (..., 16)，返回失灵置零的读数及可用性向量。"""
        validate_probability(failure_probability)
        readings = np.asarray(observations, dtype=np.float32)
        if readings.ndim < 1 or readings.shape[-1] != RAW_OBSERVATION_DIM or not np.isfinite(readings).all():
            raise ValueError("原始观测须为有限的 (..., 8) 数组")
        availability = np.ones_like(readings)
        draws = self.rng.random((*readings.shape[:-1], len(FAILURE_INDICES)))
        availability[..., FAILURE_INDICES] = draws >= failure_probability
        return np.concatenate((readings * availability, availability), axis=-1)
