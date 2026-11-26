
"""
AHDS physical and information-preservation metrics.

Evaluation-only utility.

Clean reference observations, counterfactual observations,
and optical ground truth must never be supplied to a
defensive controller through this module.

All image measurements operate on RGB channel samples.
Spatial masks are expanded across RGB channels.

Undefined regional metrics return None rather than zero.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InformationMetricConfig:

    epsilon: float = 1e-12

    def __post_init__(self):

        if (
            not math.isfinite(self.epsilon)
            or self.epsilon <= 0
        ):

            raise ValueError(
                "epsilon must be positive and finite."
            )


class InformationPreservationMetrics:

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else InformationMetricConfig()
        )

    @staticmethod
    def _image(value, name, normalized=False):

        if not isinstance(value, torch.Tensor):

            raise TypeError(
                f"{name} must be a tensor."
            )

        if (
            value.ndim != 3
            or value.shape[0] != 3
            or value.shape[1] < 1
            or value.shape[2] < 1
        ):

            raise ValueError(
                f"{name} must have shape (3, H, W)."
            )

        if not torch.is_floating_point(value):

            raise TypeError(
                f"{name} must be floating point."
            )

        value = value.detach().double()

        if not torch.isfinite(value).all():

            raise ValueError(
                f"{name} contains nonfinite values."
            )

        if (value < 0).any():

            raise ValueError(
                f"{name} contains negative values."
            )

        if normalized and (value > 1).any():

            raise ValueError(
                f"{name} must be within [0, 1]."
            )

        return value

    @staticmethod
    def _mask(value, image, name):

        if not isinstance(value, torch.Tensor):

            raise TypeError(
                f"{name} must be a tensor."
            )

        if value.dtype != torch.bool:

            raise TypeError(
                f"{name} must be boolean."
            )

        if value.device != image.device:

            raise ValueError(
                f"{name} device mismatch."
            )

        if value.shape != image.shape:

            raise ValueError(
                f"{name} shape mismatch."
            )

        return value.detach()

    @staticmethod
    def _region_mask(region_mask, image):

        if region_mask is None:

            return torch.ones_like(
                image,
                dtype=torch.bool
            )

        if not isinstance(
            region_mask,
            torch.Tensor
        ):

            raise TypeError(
                "region_mask must be a tensor."
            )

        if region_mask.dtype != torch.bool:

            raise TypeError(
                "region_mask must be boolean."
            )

        if region_mask.device != image.device:

            raise ValueError(
                "Region mask device mismatch."
            )

        height, width = image.shape[-2:]

        if region_mask.shape == (height, width):

            region_mask = region_mask.unsqueeze(0)

        if region_mask.shape != (1, height, width):

            raise ValueError(
                "Expected spatial mask (H, W) "
                "or (1, H, W)."
            )

        return region_mask.expand_as(image)

    @staticmethod
    def _calibration(sensor_config):

        sensor_config.validate()

        exposure = float(
            sensor_config.exposure_ms
        )

        gain = float(
            sensor_config.gain
        )

        full_well = float(
            sensor_config.full_well_electrons
        )

        conversion = float(
            sensor_config.electrons_per_unit_ms
        )

        bits = sensor_config.adc_bits

        if isinstance(bits, bool):

            raise ValueError(
                "adc_bits cannot be boolean."
            )

        adc_max = (2 ** bits) - 1

        scale = (
            exposure * gain * conversion
            / full_well
        )

        if (
            not math.isfinite(scale)
            or scale <= 0
        ):

            raise ValueError(
                "Invalid optical conversion scale."
            )

        return {

            "exposure": exposure,

            "gain": gain,

            "adc_max": adc_max,

            "scale": scale,

            "radiance_step": (
                1.0 / (adc_max * scale)
            )
        }

    def _observation(
        self,
        sensor_result,
        sensor_config,
        name
    ):

        image = self._image(
            sensor_result["normalized_output"],
            name,
            normalized=True
        )

        saturated = self._mask(
            sensor_result["saturated"],
            image,
            name + "_saturated"
        )

        adc_clipped = self._mask(
            sensor_result["adc_clipped"],
            image,
            name + "_adc_clipped"
        )

        calibration = self._calibration(
            sensor_config
        )

        adc = sensor_result["adc_output"]

        if not isinstance(adc, torch.Tensor):

            raise TypeError(
                "ADC output must be a tensor."
            )

        if (
            adc.shape != image.shape
            or adc.device != image.device
        ):

            raise ValueError(
                "ADC output shape or device mismatch."
            )

        if adc.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64
        ):

            raise TypeError(
                "ADC output must contain integers."
            )

        adc_max = calibration["adc_max"]

        if (
            (adc < 0).any()
            or (adc > adc_max).any()
        ):

            raise ValueError(
                "ADC output outside calibrated range."
            )

        exact_image = (
            adc.detach().double() / adc_max
        )

        if not torch.allclose(
            image,
            exact_image,
            atol=1e-7,
            rtol=1e-7
        ):

            raise ValueError(
                "Normalized output and ADC codes "
                "are inconsistent."
            )

        return {

            "image": exact_image,

            "adc": adc.detach(),

            "saturated": saturated,

            "adc_clipped": adc_clipped,

            "uncensored": ~(
                saturated | adc_clipped
            ),

            "calibration": calibration
        }

    @staticmethod
    def _mean(values, mask):

        selected = values[mask]

        if selected.numel() == 0:

            return None

        return float(
            selected.double().mean().item()
        )

    @staticmethod
    def _std(values, mask):

        selected = values[mask]

        if selected.numel() == 0:

            return None

        return float(
            selected.double().std(
                unbiased=False
            ).item()
        )

    @staticmethod
    def _unique_levels(adc, mask):

        selected = adc[mask]

        return int(
            torch.unique(selected).numel()
        )

    def _ratio(self, numerator, denominator):

        if (
            numerator is None
            or denominator is None
            or abs(denominator) <= self.config.epsilon
        ):

            return None

        return numerator / denominator

    @staticmethod
    def _gradient_mae(first, second, region):

        # Only evaluate edges whose two endpoints
        # are both inside the selected region.

        total_error = 0.0

        total_edges = 0

        horizontal_mask = (
            region[:, :, 1:]
            & region[:, :, :-1]
        )

        horizontal_error = (

            (
                first[:, :, 1:]
                - first[:, :, :-1]
            )

            -

            (
                second[:, :, 1:]
                - second[:, :, :-1]
            )

        ).abs()

        vertical_mask = (
            region[:, 1:, :]
            & region[:, :-1, :]
        )

        vertical_error = (

            (
                first[:, 1:, :]
                - first[:, :-1, :]
            )

            -

            (
                second[:, 1:, :]
                - second[:, :-1, :]
            )

        ).abs()

        for error, mask in (

            (horizontal_error, horizontal_mask),

            (vertical_error, vertical_mask)

        ):

            selected = error[mask]

            total_error += float(
                selected.sum().item()
            )

            total_edges += selected.numel()

        if total_edges == 0:

            return None

        return total_error / total_edges

    @torch.no_grad()
    def evaluate(
        self,
        reference_result,
        counterfactual_result,
        observed_result,
        reference_config,
        applied_config,
        clean_optical_scene,
        region_mask=None
    ):

        """
        Evaluate a single frame.

        reference_result:
            Clean capture at reference configuration.

        counterfactual_result:
            Clean capture at applied configuration.

        observed_result:
            Actual capture at applied configuration.

        clean_optical_scene:
            Evaluation-only optical ground truth.

        region_mask:
            Optional spatial evaluation region.

        Returns a flat dictionary of scalar metrics.
        """

        reference = self._observation(
            reference_result,
            reference_config,
            "reference"
        )

        counterfactual = self._observation(
            counterfactual_result,
            applied_config,
            "counterfactual"
        )

        observed = self._observation(
            observed_result,
            applied_config,
            "observed"
        )

        R = reference["image"]

        C = counterfactual["image"]

        Y = observed["image"]

        L = self._image(
            clean_optical_scene,
            "clean_optical_scene"
        )

        for image in (C, Y, L):

            if (
                image.shape != R.shape
                or image.device != R.device
            ):

                raise ValueError(
                    "Observation shape or device mismatch."
                )

        region = self._region_mask(
            region_mask,
            R
        )

        region_samples = int(
            region.sum().item()
        )

        valid_observed = (
            region & observed["uncensored"]
        )

        valid_clean = (
            region & counterfactual["uncensored"]
        )

        common_valid = (
            region
            & observed["uncensored"]
            & counterfactual["uncensored"]
        )

        reference_calibration = (
            reference["calibration"]
        )

        applied_calibration = (
            observed["calibration"]
        )

        applied_scale = (
            applied_calibration["scale"]
        )

        reference_scale = (
            reference_calibration["scale"]
        )

        reconstructed_observed = (
            Y / applied_scale
        )

        reconstructed_clean = (
            C / applied_scale
        )

        reconstructed_reference = (
            R / reference_scale
        )

        valid_reference = (
            region & reference["uncensored"]
        )

        reference_std = self._std(
            R,
            region
        )

        counterfactual_std = self._std(
            C,
            region
        )

        observed_std = self._std(
            Y,
            region
        )

        reference_levels = self._unique_levels(
            reference["adc"],
            region
        )

        counterfactual_levels = self._unique_levels(
            counterfactual["adc"],
            region
        )

        observed_levels = self._unique_levels(
            observed["adc"],
            region
        )

        metrics = {

            # Region accounting

            "region_sample_count":
                region_samples,

            "region_fraction":
                region_samples / R.numel(),

            # Physical protection

            "saturation_ratio":
                self._mean(
                    observed["saturated"],
                    region
                ),

            "adc_clipping_ratio":
                self._mean(
                    observed["adc_clipped"],
                    region
                ),

            "censored_ratio":
                self._mean(
                    ~observed["uncensored"],
                    region
                ),

            # Observation coverage

            "valid_observation_coverage":
                self._mean(
                    observed["uncensored"],
                    region
                ),

            "clean_valid_coverage":
                self._mean(
                    counterfactual["uncensored"],
                    region
                ),

            "reference_valid_coverage":
                self._mean(
                    reference["uncensored"],
                    region
                ),

            "common_valid_coverage":
                self._mean(
                    common_valid,
                    region
                ),

            "nonzero_valid_coverage":
                self._mean(
                    observed["uncensored"]
                    & (observed["adc"] > 0),
                    region
                ),

            "valid_observation_count":
                int(valid_observed.sum().item()),

            "common_valid_count":
                int(common_valid.sum().item()),

            # Raw sensor-output distortion

            "configuration_mae":
                self._mean(
                    (C - R).abs(),
                    region
                ),

            "disturbance_mae":
                self._mean(
                    (Y - C).abs(),
                    region
                ),

            "reference_mae":
                self._mean(
                    (Y - R).abs(),
                    region
                ),

            # Optical reconstruction

            "radiance_reconstruction_mae":
                self._mean(
                    (
                        reconstructed_observed - L
                    ).abs(),
                    valid_observed
                ),

            "clean_radiance_reconstruction_mae":
                self._mean(
                    (
                        reconstructed_clean - L
                    ).abs(),
                    valid_clean
                ),

            "reference_radiance_reconstruction_mae":
                self._mean(
                    (
                        reconstructed_reference - L
                    ).abs(),
                    valid_reference
                ),

            "common_valid_radiance_distortion_mae":
                self._mean(
                    (
                        reconstructed_observed
                        - reconstructed_clean
                    ).abs(),
                    common_valid
                ),

            # Exposure and gain utilization

            "exposure_utilization": (
                applied_calibration["exposure"]
                / reference_calibration["exposure"]
            ),

            "gain_utilization": (
                applied_calibration["gain"]
                / reference_calibration["gain"]
            ),

            "optical_scale_utilization": (
                applied_scale / reference_scale
            ),

            # Quantization diagnostics

            "radiance_quantization_step":
                applied_calibration["radiance_step"],

            "ideal_quantization_half_step": (
                applied_calibration["radiance_step"]
                / 2.0
            ),

            "relative_quantization_step": (
                applied_calibration["radiance_step"]
                / reference_calibration["radiance_step"]
            ),

            "reference_adc_levels":
                reference_levels,

            "clean_adc_levels":
                counterfactual_levels,

            "observed_adc_levels":
                observed_levels,

            "clean_adc_level_retention":
                self._ratio(
                    counterfactual_levels,
                    reference_levels
                ),

            "clean_zero_adc_ratio":
                self._mean(
                    counterfactual["adc"] == 0,
                    region
                ),

            "observed_zero_adc_ratio":
                self._mean(
                    observed["adc"] == 0,
                    region
                ),

            # Structural fidelity

            "gradient_mae":
                self._gradient_mae(
                    Y, R, region
                ),

            "configuration_gradient_mae":
                self._gradient_mae(
                    C, R, region
                ),

            "disturbance_gradient_mae":
                self._gradient_mae(
                    Y, C, region
                ),

            "reference_contrast":
                reference_std,

            "clean_contrast":
                counterfactual_std,

            "observed_contrast":
                observed_std,

            "contrast_retention":
                self._ratio(
                    observed_std,
                    reference_std
                ),

            "clean_contrast_retention":
                self._ratio(
                    counterfactual_std,
                    reference_std
                ),

            # Brightness diagnostics

            "reference_mean_output":
                self._mean(R, region),

            "clean_mean_output":
                self._mean(C, region),

            "observed_mean_output":
                self._mean(Y, region)
        }

        return metrics
