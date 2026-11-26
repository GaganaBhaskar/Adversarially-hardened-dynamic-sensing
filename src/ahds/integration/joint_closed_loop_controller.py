"""
Joint nominal exposure-gain adaptation with
exclusive exposure-based defensive authority.

The validated exposure-only controller is preserved.

Module 3 controls exposure and gain during safe
normal operation.

Module 5 controls exposure during protection,
probing and recovery. Gain remains frozen.
"""

from dataclasses import replace
import math

from ahds.integration.closed_loop_controller import (
    ClosedLoopExposureController,
)

from ahds.module3_dynamic_sensing.joint_controller import (
    JointControllerConfig,
    JointExposureGainController,
)


class JointClosedLoopController(
    ClosedLoopExposureController
):

    def __init__(
        self,
        system,
        initial_sensor_config=None,
        joint_config=None,
        **kwargs,
    ):

        self.joint_config = (
            joint_config or JointControllerConfig()
        )

        self.joint_config.validate()

        super().__init__(
            system=system,
            initial_sensor_config=initial_sensor_config,
            **kwargs,
        )

        if not (
            self.joint_config.min_gain
            <= self.initial_config.gain
            <= self.joint_config.max_gain
        ):
            raise ValueError("Invalid initial gain.")

        if not (
            self.joint_config.min_exposure_ms
            <= self.initial_config.exposure_ms
            <= self.joint_config.max_exposure_ms
        ):
            raise ValueError("Invalid initial exposure.")

        if (
            self.joint_config.min_exposure_ms
            != self.adaptive_config.min_exposure_ms
            or self.joint_config.max_exposure_ms
            != self.adaptive_config.max_exposure_ms
        ):
            raise ValueError(
                "Joint and inherited exposure bounds differ."
            )

    def reset(self):

        super().reset()

        self._joint_command = None

        if self.system in ("B", "C"):

            self.adaptive = JointExposureGainController(
                config=self.joint_config,
                initial_exposure_ms=(
                    self.initial_config.exposure_ms
                ),
                initial_gain=self.initial_config.gain,
            )

    def _adaptive_command(
        self,
        view,
        applied_config,
    ):

        # Synchronize both state variables with the
        # physically applied sensor configuration.

        self.adaptive.exposure_ms = float(
            applied_config.exposure_ms
        )

        self.adaptive.gain = float(
            applied_config.gain
        )

        # ADC clipping is derived from an already
        # permitted physical sensor observation.

        adc_clip_ratio = float(
            view["adc_clipped"].double().mean()
        )

        result = self.adaptive.update(
            view["normalized_output"],
            adc_clip_ratio=adc_clip_ratio,
        )

        gain = float(result["gain"])

        if not math.isfinite(gain):
            raise ValueError("Non-finite gain command.")

        if not (
            self.joint_config.min_gain
            <= gain
            <= self.joint_config.max_gain
        ):
            raise ValueError("Gain outside joint bounds.")

        self._joint_command = result

        return result

    def update(
        self,
        sensor_result,
        applied_config,
    ):

        self._joint_command = None

        # Reuse the validated authority policy.
        # The inherited controller invokes our joint
        # adaptive command only when M3 owns authority.

        next_config, record = super().update(
            sensor_result,
            applied_config,
        )

        if record["authority"] == "M3":

            if self._joint_command is None:
                raise RuntimeError(
                    "Missing joint adaptive command."
                )

            next_gain = float(
                self._joint_command["gain"]
            )

            gain_authority = "M3"

        else:

            # Freeze gain during every non-adaptive
            # authority state, including restoration.

            next_gain = float(applied_config.gain)

            gain_authority = (
                "STATIC"
                if self.system == "A"
                else "HOLD"
            )

        next_config = replace(
            next_config,
            gain=next_gain,
        )

        next_config.validate()

        record["next_gain"] = next_gain

        record["gain_authority"] = gain_authority

        record["gain_changed"] = (
            next_gain != float(applied_config.gain)
        )

        record["joint_adaptation_enabled"] = (
            self.system in ("B", "C")
        )

        self.expected_config = replace(next_config)

        return next_config, record
