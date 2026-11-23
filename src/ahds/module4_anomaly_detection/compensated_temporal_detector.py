
from dataclasses import dataclass
import math
import torch


@dataclass
class CompensatedTemporalConfig:
    difference_threshold: float = 0.15
    min_changed_fraction: float = 0.005
    max_global_scale: float = 10.0
    epsilon: float = 1e-6

    def validate(self):
        values = (
            self.difference_threshold,
            self.min_changed_fraction,
            self.max_global_scale,
            self.epsilon
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Parameters must be finite.")

        if not 0 <= self.difference_threshold <= 1:
            raise ValueError("Invalid difference threshold.")

        if not 0 <= self.min_changed_fraction <= 1:
            raise ValueError("Invalid changed fraction.")

        if self.max_global_scale < 1:
            raise ValueError("Global scale limit must be >= 1.")

        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive.")


class CompensatedTemporalDetector:

    def __init__(self, config=None):
        self.config = config or CompensatedTemporalConfig()
        self.config.validate()
        self.previous_frame = None

    def reset(self):
        self.previous_frame = None

    def detect(self, sensor_output):

        if (
            sensor_output.ndim != 3
            or sensor_output.shape[0] != 3
        ):
            raise ValueError("Expected RGB tensor (3,H,W).")

        if not torch.isfinite(sensor_output).all():
            raise ValueError("Non-finite sensor values.")

        if (
            (sensor_output < 0).any()
            or (sensor_output > 1).any()
        ):
            raise ValueError("Expected normalized values in [0,1].")

        current = sensor_output.detach().clone()

        if self.previous_frame is None:
            self.previous_frame = current

            zero_map = torch.zeros_like(current[0])

            return {
                "detected": False,
                "changed_fraction": 0.0,
                "global_scale": 1.0,
                "difference_map": zero_map,
                "change_mask": zero_map.bool(),
                "initialized": True
            }

        previous = self.previous_frame

        if previous.shape != current.shape:
            raise ValueError("Frame dimensions changed. Call reset().")

        # Least-squares global brightness scaling.
        numerator = (previous * current).sum()
        denominator = (previous * previous).sum()

        scale = (
            numerator / denominator.clamp_min(self.config.epsilon)
        ).item()

        scale = max(
            1.0 / self.config.max_global_scale,
            min(scale, self.config.max_global_scale)
        )

        predicted = (previous * scale).clamp(0, 1)

        difference = (
            current - predicted
        ).abs().mean(dim=0)

        change_mask = (
            difference >= self.config.difference_threshold
        )

        changed_fraction = (
            change_mask.float().mean().item()
        )

        detected = (
            changed_fraction >= self.config.min_changed_fraction
            and bool(change_mask.any().item())
        )

        self.previous_frame = current

        return {
            "detected": bool(detected),
            "changed_fraction": changed_fraction,
            "global_scale": scale,
            "difference_map": difference,
            "change_mask": change_mask,
            "initialized": False
        }
