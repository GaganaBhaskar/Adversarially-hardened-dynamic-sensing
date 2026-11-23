
from dataclasses import dataclass
import math
import torch


@dataclass
class JointControllerConfig:
    target_brightness: float = 0.35

    min_exposure_ms: float = 1.0
    max_exposure_ms: float = 30.0

    min_gain: float = 1.0
    max_gain: float = 3.0

    adjustment_rate: float = 0.4
    max_adc_clip_ratio: float = 0.02
    epsilon: float = 1e-6

    def validate(self):
        values = [
            self.target_brightness,
            self.min_exposure_ms,
            self.max_exposure_ms,
            self.min_gain,
            self.max_gain,
            self.adjustment_rate,
            self.max_adc_clip_ratio,
            self.epsilon
        ]

        if not all(math.isfinite(v) for v in values):
            raise ValueError("All controller parameters must be finite.")

        if not 0 < self.target_brightness < 1:
            raise ValueError("Invalid brightness target.")

        if not 0 < self.min_exposure_ms <= self.max_exposure_ms:
            raise ValueError("Invalid exposure limits.")

        if not 0 < self.min_gain <= self.max_gain:
            raise ValueError("Invalid gain limits.")

        if not 0 < self.adjustment_rate <= 1:
            raise ValueError("Invalid adjustment rate.")

        if not 0 <= self.max_adc_clip_ratio <= 1:
            raise ValueError("Invalid clipping limit.")

        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive.")


class JointExposureGainController:

    def __init__(
        self,
        config=None,
        initial_exposure_ms=5.0,
        initial_gain=1.0
    ):
        self.config = config or JointControllerConfig()
        self.config.validate()

        self.exposure_ms = float(
            max(
                self.config.min_exposure_ms,
                min(initial_exposure_ms,
                    self.config.max_exposure_ms)
            )
        )

        self.gain = float(
            max(
                self.config.min_gain,
                min(initial_gain,
                    self.config.max_gain)
            )
        )

    def update(self, sensor_output, adc_clip_ratio=0.0):

        if sensor_output.ndim != 3 or sensor_output.shape[0] != 3:
            raise ValueError("Expected RGB sensor output.")

        if not torch.isfinite(sensor_output).all():
            raise ValueError("Sensor output must be finite.")

        if (sensor_output < 0).any() or (sensor_output > 1).any():
            raise ValueError("Sensor output must lie in [0,1].")

        if not math.isfinite(adc_clip_ratio) or not 0 <= adc_clip_ratio <= 1:
            raise ValueError("Invalid ADC clipping ratio.")

        brightness = sensor_output.mean().item()

        ratio = (
            self.config.target_brightness
            / max(brightness, self.config.epsilon)
        )

        factor = ratio ** self.config.adjustment_rate

        if adc_clip_ratio > self.config.max_adc_clip_ratio:
            # Reduce amplification when clipping is excessive.
            if self.gain > self.config.min_gain:
                self.gain = max(
                    self.config.min_gain,
                    self.gain / 1.25
                )
            else:
                self.exposure_ms = max(
                    self.config.min_exposure_ms,
                    self.exposure_ms / 1.25
                )

        elif factor >= 1:
            # Prefer increasing exposure first.
            proposed_exposure = self.exposure_ms * factor

            self.exposure_ms = min(
                proposed_exposure,
                self.config.max_exposure_ms
            )

            # Increase gain only when exposure is at its limit.
            if self.exposure_ms >= self.config.max_exposure_ms:
                self.gain = min(
                    self.gain * factor,
                    self.config.max_gain
                )

        else:
            # Reduce gain first, then exposure.
            proposed_gain = self.gain * factor

            if proposed_gain >= self.config.min_gain:
                self.gain = proposed_gain
            else:
                self.gain = self.config.min_gain
                self.exposure_ms = max(
                    self.config.min_exposure_ms,
                    self.exposure_ms * factor
                )

        return {
            "brightness": brightness,
            "exposure_ms": self.exposure_ms,
            "gain": self.gain,
            "adc_clip_ratio": adc_clip_ratio
        }
