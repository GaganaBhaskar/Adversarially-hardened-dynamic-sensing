
from dataclasses import dataclass
import math


@dataclass
class ExposureProtectionConfig:
    min_exposure_ms: float = 0.1
    max_exposure_ms: float = 30.0
    saturation_trigger: float = 0.005
    reduction_factor: float = 0.5
    recovery_factor: float = 1.1
    recovery_delay_frames: int = 3

    def validate(self):
        values = (
            self.min_exposure_ms,
            self.max_exposure_ms,
            self.saturation_trigger,
            self.reduction_factor,
            self.recovery_factor
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Configuration must be finite.")

        if not (
            0 < self.min_exposure_ms <= self.max_exposure_ms
        ):
            raise ValueError("Invalid exposure limits.")

        if not 0 <= self.saturation_trigger <= 1:
            raise ValueError("Invalid saturation trigger.")

        if not 0 < self.reduction_factor < 1:
            raise ValueError("Reduction factor must be between 0 and 1.")

        if self.recovery_factor <= 1:
            raise ValueError("Recovery factor must exceed 1.")

        if (
            not isinstance(self.recovery_delay_frames, int)
            or isinstance(self.recovery_delay_frames, bool)
            or self.recovery_delay_frames < 1
        ):
            raise ValueError("Invalid recovery delay.")


class ExposureProtectionController:

    def __init__(self, config=None):
        self.config = config or ExposureProtectionConfig()
        self.config.validate()
        self.clear_frames = 0

    def reset(self):
        self.clear_frames = 0

    def update(self, current_exposure_ms, assessment):
        if not math.isfinite(current_exposure_ms):
            raise ValueError("Exposure must be finite.")

        if not (
            self.config.min_exposure_ms
            <= current_exposure_ms
            <= self.config.max_exposure_ms
        ):
            raise ValueError("Exposure outside controller limits.")

        saturation = float(
            assessment["saturation_fraction"]
        )

        if not math.isfinite(saturation) or not 0 <= saturation <= 1:
            raise ValueError("Invalid saturation measurement.")

        if saturation >= self.config.saturation_trigger:
            self.clear_frames = 0
            next_exposure = (
                current_exposure_ms
                * self.config.reduction_factor
            )
            action = "PROTECT"

        else:
            self.clear_frames += 1

            if self.clear_frames >= self.config.recovery_delay_frames:
                next_exposure = (
                    current_exposure_ms
                    * self.config.recovery_factor
                )
                action = "RECOVER"
            else:
                next_exposure = current_exposure_ms
                action = "HOLD"

        next_exposure = max(
            self.config.min_exposure_ms,
            min(next_exposure, self.config.max_exposure_ms)
        )

        return {
            "exposure_ms": next_exposure,
            "action": action,
            "saturation_fraction": saturation,
            "clear_frames": self.clear_frames
        }
