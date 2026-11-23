
from dataclasses import dataclass
import math
import torch


@dataclass
class ControllerConfig:
    target_brightness: float = 0.35
    min_exposure_ms: float = 1.0
    max_exposure_ms: float = 30.0
    adjustment_rate: float = 0.4
    epsilon: float = 1e-6

    def validate(self):
        if not 0 < self.target_brightness < 1:
            raise ValueError("Target brightness must be between 0 and 1.")

        if self.min_exposure_ms <= 0:
            raise ValueError("Minimum exposure must be positive.")

        if self.max_exposure_ms < self.min_exposure_ms:
            raise ValueError("Invalid exposure range.")

        if not 0 < self.adjustment_rate <= 1:
            raise ValueError("Adjustment rate must be in (0, 1].")


class AdaptiveExposureController:
    """
    Conventional global auto-exposure baseline.

    No adversarial attack detection or spatial defense.
    """

    def __init__(
        self,
        config=None,
        initial_exposure_ms=10.0
    ):
        self.config = config or ControllerConfig()
        self.config.validate()

        self.exposure_ms = float(
            max(
                self.config.min_exposure_ms,
                min(
                    initial_exposure_ms,
                    self.config.max_exposure_ms
                )
            )
        )

    def update(self, sensor_output):
        """
        Update exposure using measured digital brightness.

        sensor_output: normalized RGB tensor (3,H,W)
        """

        if sensor_output.ndim != 3 or sensor_output.shape[0] != 3:
            raise ValueError("Expected normalized RGB sensor output.")

        if not torch.isfinite(sensor_output).all():
            raise ValueError("Sensor output must be finite.")

        if (sensor_output < 0).any() or (sensor_output > 1).any():
            raise ValueError("Sensor output must lie in [0,1].")

        measured_brightness = sensor_output.mean().item()

        desired_ratio = (
            self.config.target_brightness
            / max(measured_brightness, self.config.epsilon)
        )

        # Smooth multiplicative adjustment
        factor = desired_ratio ** self.config.adjustment_rate

        new_exposure = self.exposure_ms * factor

        self.exposure_ms = float(
            max(
                self.config.min_exposure_ms,
                min(
                    new_exposure,
                    self.config.max_exposure_ms
                )
            )
        )

        return {
            "measured_brightness": measured_brightness,
            "exposure_ms": self.exposure_ms
        }
