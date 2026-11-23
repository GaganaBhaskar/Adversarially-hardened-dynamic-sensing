
from dataclasses import dataclass
import math
import torch


@dataclass
class TemporalConfig:
    difference_threshold: float = 0.15
    min_changed_fraction: float = 0.005

    def validate(self):
        if (
            not math.isfinite(self.difference_threshold)
            or not 0 <= self.difference_threshold <= 1
        ):
            raise ValueError("Invalid difference threshold.")

        if (
            not math.isfinite(self.min_changed_fraction)
            or not 0 <= self.min_changed_fraction <= 1
        ):
            raise ValueError("Invalid changed fraction threshold.")


class TemporalAnomalyDetector:
    """
    Detect frame-to-frame sensor brightness changes.

    This detector identifies temporal inconsistencies,
    not confirmed adversarial attacks.
    """

    def __init__(self, config=None):
        self.config = config or TemporalConfig()
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
            raise ValueError("Sensor output must be finite.")

        if (
            (sensor_output < 0).any()
            or (sensor_output > 1).any()
        ):
            raise ValueError("Expected normalized sensor values in [0,1].")

        current = sensor_output.detach().clone()

        if self.previous_frame is None:
            self.previous_frame = current

            return {
                "detected": False,
                "changed_fraction": 0.0,
                "difference_map": torch.zeros_like(current[0]),
                "change_mask": torch.zeros_like(
                    current[0], dtype=torch.bool
                ),
                "initialized": True
            }

        if current.shape != self.previous_frame.shape:
            raise ValueError("Frame dimensions changed. Call reset().")

        difference = (
            current - self.previous_frame
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
            "difference_map": difference,
            "change_mask": change_mask,
            "initialized": False
        }
