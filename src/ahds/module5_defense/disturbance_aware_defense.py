
"""
AHDS Module 5
Disturbance-Aware Defensive Reconfiguration Controller.

Combines:
    - Physical clipping observations
    - Module 4 disturbance assessment
    - Exposure-aware optical reconstruction
    - Frozen-reference disturbance memory
    - Controlled exposure recovery

All exposure commands apply to the following frame.

No simulated attack labels or optical ground truth
are used by the controller.
"""

from dataclasses import dataclass
import math

from .exposure_aware_observation import (
    ExposureAwareObservationEstimator
)

from .disturbance_memory import (
    ExposureAwareDisturbanceMemory
)


@dataclass(frozen=True)
class DisturbanceAwareDefenseConfig:

    min_exposure_ms: float = 0.1

    max_exposure_ms: float = 30.0

    saturation_trigger: float = 0.005

    reduction_factor: float = 0.5

    persistent_probe_factor: float = 1.05

    clear_probe_factor: float = 1.10

    recovery_factor: float = 1.20

    persistent_probe_interval: int = 4

    clear_confirmation_frames: int = 3

    failed_probe_cooldown: int = 8

    def validate(self):

        for name in (
            "min_exposure_ms",
            "max_exposure_ms",
            "saturation_trigger",
            "reduction_factor",
            "persistent_probe_factor",
            "clear_probe_factor",
            "recovery_factor"
        ):

            value = getattr(self, name)

            if not math.isfinite(value) or value <= 0:

                raise ValueError(
                    f"{name} must be positive and finite."
                )

        if self.min_exposure_ms > self.max_exposure_ms:

            raise ValueError(
                "Invalid exposure bounds."
            )

        if self.saturation_trigger > 1:

            raise ValueError(
                "Invalid saturation threshold."
            )

        if not 0 < self.reduction_factor < 1:

            raise ValueError(
                "Invalid exposure reduction factor."
            )

        if not (
            1 < self.persistent_probe_factor
            <= self.clear_probe_factor
            <= self.recovery_factor
        ):

            raise ValueError(
                "Invalid recovery factor ordering."
            )

        for name in (
            "persistent_probe_interval",
            "clear_confirmation_frames",
            "failed_probe_cooldown"
        ):

            value = getattr(self, name)

            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):

                raise ValueError(
                    f"{name} must be a positive integer."
                )


class DisturbanceAwareDefenseController:

    def __init__(
        self,
        config=None,
        observation_config=None,
        memory_config=None
    ):

        self.config = (
            config
            if config is not None
            else DisturbanceAwareDefenseConfig()
        )

        self.config.validate()

        self.estimator = (
            ExposureAwareObservationEstimator(
                observation_config
            )
        )

        self.memory = (
            ExposureAwareDisturbanceMemory(
                memory_config
            )
        )

        self.reset()

    def reset(self):

        self.memory.reset()

        self.state = "NORMAL"

        self.protection_active = False

        self.reference_exposure = None

        self.safe_exposure = None

        self.probe_pending = False

        self.probe_origin = None

        self.clear_frames = 0

        self.persistent_frames = 0

        self.cooldown_remaining = 0

        self.last_command = None

    def _clip(self, exposure):

        return max(
            self.config.min_exposure_ms,
            min(
                self.config.max_exposure_ms,
                float(exposure)
            )
        )

    @staticmethod
    def _fraction(name, value):

        value = float(value)

        if (
            not math.isfinite(value)
            or not 0 <= value <= 1
        ):

            raise ValueError(
                f"{name} must be within [0, 1]."
            )

        return value

    def initialize_reference(
        self,
        sensor_result,
        sensor_config
    ):

        """
        Explicit trusted startup initialization.

        The caller must establish that the startup
        observation is suitable as a reference.
        """

        observation = self.estimator.estimate(
            sensor_result,
            sensor_config
        )

        if (
            observation["clipped_fraction"]
            >= self.config.saturation_trigger
        ):

            raise ValueError(
                "Unsafe startup observation."
            )

        self.memory.initialize_reference(
            observation
        )

    def _activate_protection(self, exposure):

        if not self.protection_active:

            self.reference_exposure = exposure

        self.protection_active = True

    def _clear_protection(self):

        self.state = "NORMAL"

        self.protection_active = False

        self.reference_exposure = None

        self.safe_exposure = None

        self.probe_pending = False

        self.probe_origin = None

        self.clear_frames = 0

        self.persistent_frames = 0

        self.cooldown_remaining = 0

        # Optical reference is intentionally preserved.

    def _result(
        self,
        exposure,
        action,
        evidence,
        observation,
        assessment_fraction,
        tracker_state
    ):

        exposure = self._clip(exposure)

        if (
            self.protection_active
            and self.reference_exposure is not None
        ):

            exposure = min(
                exposure,
                self.reference_exposure
            )

        self.last_command = exposure

        return {

            "exposure_ms": exposure,

            "action": action,

            "controller_state": self.state,

            "tracker_state": tracker_state,

            "protection_active":
                self.protection_active,

            "reference_exposure":
                self.reference_exposure,

            "safe_exposure":
                self.safe_exposure,

            "probe_pending":
                self.probe_pending,

            "clear_frames":
                self.clear_frames,

            "persistent_frames":
                self.persistent_frames,

            "cooldown_remaining":
                self.cooldown_remaining,

            "evidence_status":
                evidence["status"],

            "evidence_reason":
                evidence["reason"],

            "optical_changed_fraction":
                evidence["changed_fraction"],

            "global_scale":
                evidence["global_scale"],

            "common_valid_fraction":
                evidence["common_valid_fraction"],

            "physical_clipping_fraction":
                observation["clipped_fraction"],

            "censored_fraction":
                observation["censored_fraction"],

            "saturation_fraction":
                assessment_fraction
        }

    def update(
        self,
        sensor_result,
        sensor_config,
        assessment,
        tracker_state
    ):

        exposure = float(
            sensor_config.exposure_ms
        )

        if (
            not math.isfinite(exposure)
            or not (
                self.config.min_exposure_ms
                <= exposure
                <= self.config.max_exposure_ms
            )
        ):

            raise ValueError(
                "Current exposure outside valid bounds."
            )

        if tracker_state not in (
            "NORMAL",
            "TRANSIENT",
            "ACTIVE",
            "RECOVERY"
        ):

            raise ValueError(
                "Invalid Module 4 tracker state."
            )

        # During protection, enforce the one-frame
        # command contract. Module 3 arbitration will
        # be addressed during integration.

        if (
            self.protection_active
            and self.last_command is not None
            and not math.isclose(
                exposure,
                self.last_command,
                rel_tol=1e-6,
                abs_tol=1e-8
            )
        ):

            raise ValueError(
                "Applied exposure does not match "
                "previous defensive command."
            )

        assessment_fraction = self._fraction(
            "saturation_fraction",
            assessment["saturation_fraction"]
        )

        observation = self.estimator.estimate(
            sensor_result,
            sensor_config
        )

        evidence = self.memory.assess(
            observation
        )

        status = evidence["status"]

        # Keep RGB sample and spatial measurements
        # separate. Either may trigger protection.

        unsafe = (
            assessment_fraction
            >= self.config.saturation_trigger

            or observation["clipped_fraction"]
            >= self.config.saturation_trigger

            or observation["censored_fraction"]
            >= self.config.saturation_trigger
        )

        def finish(next_exposure, action):

            return self._result(
                next_exposure,
                action,
                evidence,
                observation,
                assessment_fraction,
                tracker_state
            )

        # ----------------------------------------------------
        # 1. Evaluate a pending exposure probe
        # ----------------------------------------------------

        successful_probe = False

        if self.probe_pending:

            origin = self.probe_origin

            self.probe_pending = False

            self.probe_origin = None

            if unsafe or status == "UNKNOWN":

                self.state = "HOLD_PROTECTION"

                self.clear_frames = 0

                self.persistent_frames = 0

                self.cooldown_remaining = (
                    self.config.failed_probe_cooldown
                )

                rollback = origin

                if unsafe:

                    rollback = min(
                        origin,
                        exposure
                        * self.config.reduction_factor
                    )

                    self.state = "PROTECT"

                    self.safe_exposure = None

                else:

                    self.safe_exposure = origin

                return finish(
                    rollback,
                    "PROBE_FAILED"
                )

            successful_probe = True

            self.safe_exposure = exposure

        # ----------------------------------------------------
        # 2. Safety-prioritized protection
        # ----------------------------------------------------

        if unsafe:

            self._activate_protection(exposure)

            self.state = "PROTECT"

            self.clear_frames = 0

            self.persistent_frames = 0

            self.safe_exposure = None

            return finish(
                exposure
                * self.config.reduction_factor,
                "PROTECT"
            )

        # ----------------------------------------------------
        # 3. Normal monitoring
        # ----------------------------------------------------

        if not self.protection_active:

            self.state = "NORMAL"

            if status == "PRESENT":

                # Disturbance evidence alone does not
                # justify reducing a currently safe
                # exposure.

                return finish(
                    exposure,
                    "MONITOR_DISTURBANCE"
                )

            if status == "UNKNOWN":

                return finish(
                    exposure,
                    "MONITOR_UNKNOWN"
                )

            return finish(
                exposure,
                "HOLD"
            )

        # ----------------------------------------------------
        # 4. Current exposure is below safety thresholds
        # ----------------------------------------------------

        self.safe_exposure = exposure

        if self.cooldown_remaining > 0:

            self.cooldown_remaining -= 1

            self.clear_frames = 0

            self.persistent_frames = 0

            self.state = "HOLD_PROTECTION"

            return finish(
                exposure,
                "HOLD_COOLDOWN"
            )

        # ----------------------------------------------------
        # 5. Insufficient environmental evidence
        # ----------------------------------------------------

        if status == "UNKNOWN":

            self.clear_frames = 0

            self.persistent_frames = 0

            self.state = "HOLD_PROTECTION"

            return finish(
                exposure,
                "HOLD_UNKNOWN"
            )

        # ----------------------------------------------------
        # 6. Persistent disturbance
        # ----------------------------------------------------

        if status == "PRESENT":

            self.clear_frames = 0

            self.state = "HOLD_PROTECTION"

            self.persistent_frames += 1

            if (
                self.persistent_frames
                >= self.config.persistent_probe_interval

                and exposure
                < self.reference_exposure - 1e-9
            ):

                candidate = min(
                    self.reference_exposure,
                    exposure
                    * self.config.persistent_probe_factor
                )

                self.probe_pending = True

                self.probe_origin = exposure

                self.persistent_frames = 0

                self.state = "PROBE"

                return finish(
                    candidate,
                    "PROBE"
                )

            return finish(
                exposure,
                "HOLD_PROTECTION"
            )

        # ----------------------------------------------------
        # 7. Clear environmental evidence
        # ----------------------------------------------------

        self.persistent_frames = 0

        # ACTIVE is treated as conflicting evidence.
        # Do not initiate aggressive recovery.

        if tracker_state == "ACTIVE":

            self.clear_frames = 0

            self.state = "HOLD_PROTECTION"

            return finish(
                exposure,
                "HOLD_CONFLICT"
            )

        self.clear_frames += 1

        if (
            self.clear_frames
            < self.config.clear_confirmation_frames
        ):

            self.state = "HOLD_PROTECTION"

            return finish(
                exposure,
                "CONFIRM_CLEAR"
            )

        # Restoration is declared only after an
        # observation at the reference exposure.

        if exposure >= self.reference_exposure - 1e-9:

            self._clear_protection()

            return finish(
                exposure,
                "RESTORED"
            )

        # ----------------------------------------------------
        # 8. Controlled recovery
        # ----------------------------------------------------

        recovering = (
            successful_probe
            or self.state == "RECOVER"
        )

        factor = (
            self.config.recovery_factor
            if recovering
            else self.config.clear_probe_factor
        )

        candidate = min(
            self.reference_exposure,
            exposure * factor
        )

        # Every upward command remains provisional
        # until the next sensor observation.

        self.probe_pending = True

        self.probe_origin = exposure

        self.state = (
            "RECOVER"
            if recovering
            else "PROBE"
        )

        return finish(
            candidate,
            "RECOVER" if recovering else "PROBE"
        )
