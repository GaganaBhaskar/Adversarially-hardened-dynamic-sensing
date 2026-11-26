"""
AHDS adaptive ROI output-selection integration.

The full-frame sensor observation remains available to
the sensing and defensive controllers.

ROI selection models downstream sample transmission.
It does not model physical sensor windowing, reduced
ADC conversion, hardware bandwidth, or energy savings.

Readout decisions apply to the next frame.
"""

from dataclasses import dataclass
import torch

from ahds.module3_dynamic_sensing.roi_allocator import (
    ROIConfig,
    allocate_readout,
)


@dataclass(frozen=True)
class AdaptiveROIConfig:

    # Normalized scene-prior ROI coordinates.
    x_min_fraction: float = 0.20
    y_min_fraction: float = 0.40
    x_max_fraction: float = 0.80
    y_max_fraction: float = 1.00

    background_stride: int = 4

    def validate(self):

        if not (
            0 <= self.x_min_fraction
            < self.x_max_fraction <= 1
            and
            0 <= self.y_min_fraction
            < self.y_max_fraction <= 1
        ):
            raise ValueError("Invalid normalized ROI.")

        if (
            type(self.background_stride) is not int
            or self.background_stride < 1
        ):
            raise ValueError("Invalid background stride.")

    def resolve(self, height, width):

        self.validate()

        roi = ROIConfig(
            x_min=int(self.x_min_fraction * width),
            y_min=int(self.y_min_fraction * height),
            x_max=int(self.x_max_fraction * width),
            y_max=int(self.y_max_fraction * height),
        )

        roi.validate(height, width)

        return roi


class ROIReadoutController:

    VALID_SYSTEMS = ("A", "B", "C")

    def __init__(self, system, config=None):

        if system not in self.VALID_SYSTEMS:
            raise ValueError("Unknown system.")

        self.system = system

        self.config = (
            config
            if config is not None
            else AdaptiveROIConfig()
        )

        self.config.validate()

        self.reset()

    def reset(self):

        # Begin conservatively with full-frame output.
        self.mode = "FULL_FRAME"

        self.reason = "STARTUP"

        self.frame_index = 0

    def readout(self, sensor_result):

        image = sensor_result["normalized_output"]

        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("Expected RGB sensor output.")

        _, height, width = image.shape

        if self.mode == "FULL_FRAME":

            mask = torch.ones(
                (height, width),
                dtype=torch.bool,
                device=image.device,
            )

            result = {
                "readout": image.clone(),
                "sampling_mask": mask,
                "sampling_ratio": 1.0,
                "bandwidth_reduction": 0.0,
            }

        else:

            result = allocate_readout(
                image,
                self.config.resolve(height, width),
                background_stride=(
                    self.config.background_stride
                ),
            )

        # Sampling masks are essential: unsampled zeros
        # must not be confused with measured dark pixels.

        result["sampled_adc_values"] = (
            sensor_result["adc_output"][
                :, result["sampling_mask"]
            ].clone()
        )

        result["readout_mode"] = self.mode
        result["readout_reason"] = self.reason
        result["frame"] = self.frame_index

        return result

    def update(self, control_record):

        if control_record["frame"] != self.frame_index:
            raise ValueError("ROI frame identity mismatch.")

        if control_record["system"] != self.system:
            raise ValueError("ROI system identity mismatch.")

        if self.system == "A":

            next_mode = "FULL_FRAME"
            reason = "STATIC_BASELINE"

        elif self.system == "B":

            next_mode = "ROI_PRIORITY"
            reason = "UNPROTECTED_DYNAMIC"

        else:

            safe_to_prioritize = (
                control_record["authority"] == "M3"
                and
                control_record["evidence_status"] == "CLEAR"
                and
                control_record["tracker_state"] == "NORMAL"
                and
                not control_record["protection_active"]
                and
                not control_record["probe_pending"]
            )

            if safe_to_prioritize:

                next_mode = "ROI_PRIORITY"
                reason = "NORMAL_ADAPTATION"

            else:

                next_mode = "FULL_FRAME"
                reason = "DEFENSIVE_OBSERVABILITY"

        self.mode = next_mode
        self.reason = reason

        self.frame_index += 1

        return {
            "next_readout_mode": next_mode,
            "next_readout_reason": reason,
        }
