"""
AHDS closed-loop exposure controller.

Integrates existing Module 3, Module 4 and Module 5
components without modifying their implementations.

Systems:
    A: Static sensing.
    B: Unprotected adaptive exposure.
    C: Adaptive exposure with defensive arbitration.

Commands generated from frame t apply at frame t+1.

No optical ground truth, attack labels, evaluation
references or scenario timing enter this controller.
"""

from dataclasses import replace
import math

import torch

from ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
)

from ahds.module3_dynamic_sensing.exposure_controller import (
    ControllerConfig,
    AdaptiveExposureController,
)

from ahds.module4_anomaly_detection.disturbance_assessor import (
    DisturbanceConfig,
    PhysicalDisturbanceAssessor,
)

from ahds.module4_anomaly_detection.disturbance_state_tracker import (
    StateTrackerConfig,
    DisturbanceStateTracker,
)

from ahds.module5_defense.disturbance_aware_defense import (
    DisturbanceAwareDefenseConfig,
    DisturbanceAwareDefenseController,
)


class ClosedLoopExposureController:

    SYSTEMS = ("A", "B", "C")

    PERMITTED_FIELDS = frozenset({
        "normalized_output",
        "saturated",
        "adc_clipped",
    })

    def __init__(
        self,
        system,
        initial_sensor_config=None,
        adaptive_config=None,
        assessor_config=None,
        tracker_config=None,
        defense_config=None,
        observation_config=None,
        memory_config=None,
    ):

        if system not in self.SYSTEMS:
            raise ValueError("Unknown sensing system.")

        self.system = system

        self.initial_config = replace(
            initial_sensor_config or SensorConfig()
        )

        self.initial_config.validate()

        self.adaptive_config = (
            adaptive_config or ControllerConfig()
        )

        self.assessor_config = (
            assessor_config or DisturbanceConfig()
        )

        self.tracker_config = (
            tracker_config or StateTrackerConfig()
        )

        self.defense_config = (
            defense_config or DisturbanceAwareDefenseConfig()
        )

        self.observation_config = observation_config
        self.memory_config = memory_config

        self.adaptive_config.validate()
        self.defense_config.validate()

        initial_exposure = self.initial_config.exposure_ms

        if not (
            self.adaptive_config.min_exposure_ms
            <= initial_exposure
            <= self.adaptive_config.max_exposure_ms
        ):
            raise ValueError("Invalid initial adaptive exposure.")

        if system == "C":

            if not (
                self.defense_config.min_exposure_ms
                <= self.adaptive_config.min_exposure_ms
                <= self.adaptive_config.max_exposure_ms
                <= self.defense_config.max_exposure_ms
            ):
                raise ValueError(
                    "Adaptive and defensive bounds conflict."
                )

        self.reset()

    def reset(self):

        self.expected_config = replace(
            self.initial_config
        )

        self.frame_index = 0

        self.reference_initialized = (
            self.system != "C"
        )

        self.adaptive = None
        self.assessor = None
        self.tracker = None
        self.defense = None

        if self.system in ("B", "C"):

            self.adaptive = AdaptiveExposureController(
                config=self.adaptive_config,
                initial_exposure_ms=(
                    self.initial_config.exposure_ms
                ),
            )

        if self.system == "C":

            self.assessor = PhysicalDisturbanceAssessor(
                self.assessor_config
            )

            self.tracker = DisturbanceStateTracker(
                self.tracker_config
            )

            self.defense = DisturbanceAwareDefenseController(
                config=self.defense_config,
                observation_config=self.observation_config,
                memory_config=self.memory_config,
            )

    @classmethod
    def controller_view(cls, sensor_result):

        missing = (
            cls.PERMITTED_FIELDS - set(sensor_result)
        )

        if missing:
            raise ValueError(
                f"Missing sensor fields: {sorted(missing)}"
            )

        view = {
            name: sensor_result[name].detach().clone()
            for name in sorted(cls.PERMITTED_FIELDS)
        }

        image = view["normalized_output"]

        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected RGB observation.")

        if not torch.isfinite(image).all():
            raise ValueError("Non-finite observation.")

        if (image < 0).any() or (image > 1).any():
            raise ValueError("Observation outside [0,1].")

        for name in ("saturated", "adc_clipped"):

            mask = view[name]

            if mask.shape != image.shape:
                raise ValueError("Invalid clipping mask shape.")

            if mask.dtype != torch.bool:
                raise ValueError("Expected boolean clipping mask.")

        return view

    def _verify_applied_config(self, applied_config):

        applied_config.validate()

        if applied_config != self.expected_config:
            raise ValueError(
                "Applied sensor configuration differs "
                "from the previous authoritative command."
            )

    def initialize_reference(
        self,
        startup_result,
        startup_config,
    ):

        if self.system != "C":
            raise ValueError(
                "Only system C requires a trusted reference."
            )

        if self.frame_index != 0:
            raise RuntimeError(
                "Reference initialization is startup-only."
            )

        if self.reference_initialized:
            raise RuntimeError(
                "Reference has already been initialized."
            )

        self._verify_applied_config(startup_config)

        view = self.controller_view(startup_result)

        if (
            view["saturated"] | view["adc_clipped"]
        ).any().item():

            raise ValueError(
                "Trusted startup contains censored samples."
            )

        self.defense.initialize_reference(
            view,
            replace(startup_config),
        )

        self.reference_initialized = True

    def _adaptive_command(self, view, applied_config):

        # Synchronize with the physically applied exposure.
        # Never use a stale pre-protection exposure.
        self.adaptive.exposure_ms = float(
            applied_config.exposure_ms
        )

        return self.adaptive.update(
            view["normalized_output"]
        )

    def update(self, sensor_result, applied_config):

        if not self.reference_initialized:
            raise RuntimeError(
                "Trusted startup reference required."
            )

        self._verify_applied_config(applied_config)

        view = self.controller_view(sensor_result)

        current = float(applied_config.exposure_ms)

        next_exposure = current

        authority = "STATIC"
        action = "FIXED"

        adaptive_result = None
        defense_result = None
        assessment = None
        tracker_state = None

        if self.system == "B":

            adaptive_result = self._adaptive_command(
                view,
                applied_config,
            )

            next_exposure = float(
                adaptive_result["exposure_ms"]
            )

            authority = "M3"
            action = "ADAPT"

        elif self.system == "C":

            assessment = self.assessor.detect(view)

            tracker_state = self.tracker.update(
                assessment
            )["state"]

            defense_result = self.defense.update(
                view,
                replace(applied_config),
                assessment,
                tracker_state,
            )

            defense_action = defense_result["action"]

            evidence = defense_result["evidence_status"]

            protection_active = bool(
                defense_result["protection_active"]
            )

            # Active protection has exclusive authority.
            # No clipping or modification of M5 commands.
            if protection_active:

                next_exposure = float(
                    defense_result["exposure_ms"]
                )

                authority = "M5"
                action = defense_action

            # Preserve the restoration handover frame.
            elif defense_action == "RESTORED":

                next_exposure = float(
                    defense_result["exposure_ms"]
                )

                authority = "M5_HANDOVER"
                action = "RESTORED"

            # Resume normal adaptation only when both
            # optical evidence and tracker state permit it.
            elif (
                evidence == "CLEAR"
                and tracker_state == "NORMAL"
                and defense_action == "HOLD"
            ):

                adaptive_result = self._adaptive_command(
                    view,
                    applied_config,
                )

                next_exposure = float(
                    adaptive_result["exposure_ms"]
                )

                authority = "M3"
                action = "ADAPT"

            else:

                next_exposure = current

                authority = "ARBITRATION_HOLD"
                action = defense_action

        if not math.isfinite(next_exposure):
            raise ValueError("Non-finite exposure command.")

        if authority == "M3":

            lower = self.adaptive_config.min_exposure_ms
            upper = self.adaptive_config.max_exposure_ms

        elif self.system == "C":

            lower = self.defense_config.min_exposure_ms
            upper = self.defense_config.max_exposure_ms

        else:

            lower = upper = self.initial_config.exposure_ms

        if not lower <= next_exposure <= upper:
            raise ValueError("Exposure command outside bounds.")

        next_config = replace(
            applied_config,
            exposure_ms=next_exposure,
        )

        next_config.validate()

        result = {
            "system": self.system,
            "frame": self.frame_index,
            "authority": authority,
            "action": action,
            "applied_exposure_ms": current,
            "next_exposure_ms": next_exposure,
            "applied_gain": float(applied_config.gain),
            "next_gain": float(next_config.gain),
            "tracker_state": tracker_state,
            "disturbance_detected": (
                bool(assessment["disturbance_detected"])
                if assessment is not None
                else None
            ),
            "spatial_saturation_fraction": (
                float(assessment["saturation_fraction"])
                if assessment is not None
                else None
            ),
            "evidence_status": (
                defense_result["evidence_status"]
                if defense_result is not None
                else None
            ),
            "protection_active": (
                bool(defense_result["protection_active"])
                if defense_result is not None
                else False
            ),
            "defense_action": (
                defense_result["action"]
                if defense_result is not None
                else None
            ),
            "reference_exposure": (
                defense_result["reference_exposure"]
                if defense_result is not None
                else None
            ),
            "probe_pending": (
                bool(defense_result["probe_pending"])
                if defense_result is not None
                else False
            ),
            "adaptive_updated": adaptive_result is not None,
        }

        self.expected_config = replace(next_config)

        self.frame_index += 1

        return next_config, result
