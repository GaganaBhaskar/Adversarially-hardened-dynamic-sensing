
"""
AHDS — Module 5: Probe-Based Exposure Protection

Recovery ID: M5-REC-01C-R01

Reconstructed from the documented original controller
specifications. Behavioral equivalence with the archived
baseline must be established through reproduction testing.
"""

from dataclasses import dataclass
import math


@dataclass
class ProbedExposureConfig:

    min_exposure_ms: float = 0.1
    max_exposure_ms: float = 30.0

    saturation_trigger: float = 0.005
    reduction_factor: float = 0.5

    probe_factor: float = 1.05
    probe_interval_frames: int = 4
    failed_probe_cooldown: int = 8

    def validate(self):

        values = (
            self.min_exposure_ms,
            self.max_exposure_ms,
            self.saturation_trigger,
            self.reduction_factor,
            self.probe_factor,
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError(
                "Configuration values must be finite."
            )

        if not (
            0 < self.min_exposure_ms
            <= self.max_exposure_ms
        ):
            raise ValueError("Invalid exposure limits.")

        if not 0 < self.saturation_trigger <= 1:
            raise ValueError("Invalid saturation trigger.")

        if not 0 < self.reduction_factor < 1:
            raise ValueError("Invalid reduction factor.")

        if self.probe_factor <= 1:
            raise ValueError("Probe factor must exceed 1.")

        if (
            not isinstance(self.probe_interval_frames, int)
            or isinstance(self.probe_interval_frames, bool)
            or self.probe_interval_frames < 1
        ):
            raise ValueError("Invalid probe interval.")

        if (
            not isinstance(self.failed_probe_cooldown, int)
            or isinstance(self.failed_probe_cooldown, bool)
            or self.failed_probe_cooldown < 0
        ):
            raise ValueError("Invalid cooldown period.")


class ProbedExposureController:

    def __init__(self, config=None):

        self.config = config or ProbedExposureConfig()
        self.config.validate()

        self.reset()

    def reset(self):

        self.protection_active = False

        self.reference_exposure = None
        self.safe_exposure = None

        self.probe_pending = False

        self.clear_frames = 0
        self.cooldown_remaining = 0

    def _clip_exposure(self, exposure):

        return max(
            self.config.min_exposure_ms,
            min(
                exposure,
                self.config.max_exposure_ms
            )
        )

    def _result(self, exposure, action, saturation):

        return {
            "exposure_ms": self._clip_exposure(exposure),
            "action": action,
            "saturation_fraction": saturation,
            "protection_active": self.protection_active,
            "reference_exposure": self.reference_exposure,
            "safe_exposure": self.safe_exposure,
            "probe_pending": self.probe_pending,
            "clear_frames": self.clear_frames,
            "cooldown_remaining": self.cooldown_remaining,
        }

    def update(self, current_exposure_ms, assessment):

        exposure = float(current_exposure_ms)

        if not math.isfinite(exposure):
            raise ValueError("Exposure must be finite.")

        if not (
            self.config.min_exposure_ms
            <= exposure
            <= self.config.max_exposure_ms
        ):
            raise ValueError("Exposure outside limits.")

        saturation = float(
            assessment["saturation_fraction"]
        )

        if (
            not math.isfinite(saturation)
            or not 0 <= saturation <= 1
        ):
            raise ValueError(
                "Invalid saturation measurement."
            )

        excessive_saturation = (
            saturation >= self.config.saturation_trigger
        )

        # ------------------------------------------------
        # 1. Evaluate the previous exposure probe
        # ------------------------------------------------

        if self.probe_pending:

            self.probe_pending = False

            if excessive_saturation:

                next_exposure = self.safe_exposure

                self.clear_frames = 0

                self.cooldown_remaining = (
                    self.config.failed_probe_cooldown
                )

                return self._result(
                    next_exposure,
                    "PROBE_FAILED",
                    saturation
                )

            # Probe exposure produced acceptable saturation.
            self.safe_exposure = exposure

        # ------------------------------------------------
        # 2. Immediate protection
        # ------------------------------------------------

        if excessive_saturation:

            if not self.protection_active:

                self.reference_exposure = exposure

            self.protection_active = True

            self.clear_frames = 0

            next_exposure = self._clip_exposure(
                exposure * self.config.reduction_factor
            )

            # Reduced exposure is a candidate, not yet
            # confirmed by a subsequent sensor measurement.
            self.safe_exposure = None

            return self._result(
                next_exposure,
                "PROTECT",
                saturation
            )

        # ------------------------------------------------
        # 3. Normal operation
        # ------------------------------------------------

        if not self.protection_active:

            return self._result(
                exposure,
                "HOLD",
                saturation
            )

        # Current exposure has acceptable saturation.
        self.safe_exposure = exposure

        # ------------------------------------------------
        # 4. Cooldown following failed probe
        # ------------------------------------------------

        if self.cooldown_remaining > 0:

            self.cooldown_remaining -= 1

            return self._result(
                exposure,
                "HOLD_PROTECTION",
                saturation
            )

        # ------------------------------------------------
        # 5. Check whether reference exposure is restored
        # ------------------------------------------------

        if exposure >= self.reference_exposure - 1e-9:

            self.reset()

            return self._result(
                exposure,
                "RESTORED",
                saturation
            )

        # ------------------------------------------------
        # 6. Count acceptable measurement frames
        # ------------------------------------------------

        self.clear_frames += 1

        # ------------------------------------------------
        # 7. Schedule controlled recovery probe
        # ------------------------------------------------

        if (
            self.clear_frames
            >= self.config.probe_interval_frames
        ):

            next_exposure = min(
                self.reference_exposure,
                exposure * self.config.probe_factor
            )

            self.safe_exposure = exposure

            self.probe_pending = True
            self.clear_frames = 0

            return self._result(
                next_exposure,
                "PROBE",
                saturation
            )

        # ------------------------------------------------
        # 8. Maintain protected exposure
        # ------------------------------------------------

        return self._result(
            exposure,
            "HOLD_PROTECTION",
            saturation
        )
