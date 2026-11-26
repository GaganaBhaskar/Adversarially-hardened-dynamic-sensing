
from dataclasses import dataclass
import math


@dataclass
class StateAwareExposureConfig:
    min_exposure_ms: float = 0.1
    max_exposure_ms: float = 30.0
    saturation_trigger: float = 0.005
    reduction_factor: float = 0.5
    recovery_factor: float = 1.1

    def validate(self):
        values = (
            self.min_exposure_ms,
            self.max_exposure_ms,
            self.saturation_trigger,
            self.reduction_factor,
            self.recovery_factor
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Configuration values must be finite.")

        if not 0 < self.min_exposure_ms <= self.max_exposure_ms:
            raise ValueError("Invalid exposure limits.")

        if not 0 <= self.saturation_trigger <= 1:
            raise ValueError("Invalid saturation threshold.")

        if not 0 < self.reduction_factor < 1:
            raise ValueError("Invalid reduction factor.")

        if self.recovery_factor <= 1:
            raise ValueError("Recovery factor must exceed 1.")


class StateAwareExposureController:

    def __init__(self, config=None):
        self.config = config or StateAwareExposureConfig()
        self.config.validate()
        self.reset()

    def reset(self):
        self.protection_active = False
        self.reference_exposure = None

    def update(self, current_exposure_ms, assessment, state):

        exposure = float(current_exposure_ms)

        if (
            not math.isfinite(exposure)
            or not self.config.min_exposure_ms
            <= exposure <= self.config.max_exposure_ms
        ):
            raise ValueError("Invalid current exposure.")

        saturation = float(
            assessment["saturation_fraction"]
        )

        if not math.isfinite(saturation) or not 0 <= saturation <= 1:
            raise ValueError("Invalid saturation fraction.")

        if state not in (
            "NORMAL", "TRANSIENT", "ACTIVE", "RECOVERY"
        ):
            raise ValueError("Invalid disturbance state.")

        if saturation >= self.config.saturation_trigger:

            if not self.protection_active:
                self.reference_exposure = exposure

            self.protection_active = True

            next_exposure = max(
                self.config.min_exposure_ms,
                exposure * self.config.reduction_factor
            )
            action = "PROTECT"

        elif not self.protection_active:

            next_exposure = exposure
            action = "HOLD"

        elif state in ("ACTIVE", "RECOVERY", "TRANSIENT"):

            next_exposure = exposure
            action = "HOLD_PROTECTION"

        else:

            target = self.reference_exposure

            next_exposure = min(
                target,
                exposure * self.config.recovery_factor
            )

            if next_exposure >= target - 1e-9:
                self.protection_active = False
                self.reference_exposure = None
                action = "RESTORED"
            else:
                action = "RECOVER"

        next_exposure = max(
            self.config.min_exposure_ms,
            min(next_exposure, self.config.max_exposure_ms)
        )

        return {
            "exposure_ms": next_exposure,
            "action": action,
            "protection_active": self.protection_active,
            "reference_exposure": self.reference_exposure
        }
