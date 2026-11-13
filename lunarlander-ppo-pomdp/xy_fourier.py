"""Fixed XY Fourier features and read-only, episode-aware phase diagnostics."""

import math

import numpy as np
import torch


XY_FREQUENCY_MULTIPLIERS = ((1, 2, 4, 8), (1, 2))
XY_FOURIER_DIM = 2 * sum(map(len, XY_FREQUENCY_MULTIPLIERS))


def append_xy_fourier(features):
    """Keep the base input, then append x/y bands in ascending sin/cos pairs."""
    encoded = [features]
    for axis, multipliers in enumerate(XY_FREQUENCY_MULTIPLIERS):
        frequencies = features.new_tensor(multipliers) * math.pi
        phases = features[..., axis:axis + 1] * frequencies
        encoded.append(torch.stack((phases.sin(), phases.cos()), dim=-1).flatten(-2))
    return torch.cat(encoded, dim=-1)


class XYPhaseDiagnostics:
    """Measure highest-band phase increments; never alter features or random streams."""

    def __init__(self):
        self.transitions = 0
        self.skipped_terminal = 0
        self.maximum = np.zeros(2, dtype=np.float64)
        self.half_count = np.zeros(2, dtype=np.int64)
        self.pi_count = np.zeros(2, dtype=np.int64)

    def update(self, before, after):
        # Units are pi radians. Use coordinate differences, not wrapped atan2 phases.
        delta = np.abs(np.asarray(after, dtype=np.float64)[..., :2]
                       - np.asarray(before, dtype=np.float64)[..., :2]).reshape(-1, 2)
        phase = delta * np.asarray([max(v) for v in XY_FREQUENCY_MULTIPLIERS])
        if not np.isfinite(phase).all():
            raise ValueError("XY phase diagnostics require finite coordinate transitions")
        if len(phase):
            self.transitions += len(phase)
            self.maximum = np.maximum(self.maximum, phase.max(axis=0))
            self.half_count += (phase >= .5).sum(axis=0)
            self.pi_count += (phase >= 1).sum(axis=0)

    def vector_step(self, before, after, done, infos):
        """SAME_STEP returns reset states; substitute the actual terminal observations."""
        successor = np.array(after, copy=True)
        valid = np.ones(len(successor), dtype=bool)
        for index in np.flatnonzero(done):
            finals = infos.get("final_obs")
            available = infos.get("_final_obs")
            if finals is None or (available is not None and not available[index]) or finals[index] is None:
                valid[index] = False
                self.skipped_terminal += 1
            else:
                successor[index] = finals[index]
        self.update(np.asarray(before)[valid], successor[valid])

    def metrics(self):
        result = {"xy_phase_transitions": self.transitions,
                  "xy_phase_skipped_terminal_transitions": self.skipped_terminal}
        for axis, name in enumerate(("x", "y")):
            result.update({f"xy_{name}_phase_step_max_over_pi": float(self.maximum[axis]),
                           f"xy_{name}_phase_ge_half_pi_count": int(self.half_count[axis]),
                           f"xy_{name}_phase_ge_pi_count": int(self.pi_count[axis])})
        return result
