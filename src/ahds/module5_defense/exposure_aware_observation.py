
"""
AHDS Module 5
Exposure-Aware Physical Observation Estimator.

Reconstructs radiance-like optical intensity from
quantized sensor observations in the linear,
unsaturated operating region.

Does not access simulated optical ground truth.
"""

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class ExposureAwareObservationConfig:

    min_adc_counts: int = 4

    def validate(self):

        if (
            isinstance(self.min_adc_counts, bool)
            or not isinstance(self.min_adc_counts, int)
            or self.min_adc_counts < 1
        ):
            raise ValueError(
                "min_adc_counts must be a positive integer."
            )


class ExposureAwareObservationEstimator:

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else ExposureAwareObservationConfig()
        )

        self.config.validate()

    @staticmethod
    def _positive_finite(name, value):

        value = float(value)

        if not math.isfinite(value) or value <= 0:

            raise ValueError(
                f"{name} must be positive and finite."
            )

        return value

    @torch.no_grad()
    def estimate(self, sensor_result, sensor_config):

        # ----------------------------------------------------
        # 1. Validate physical acquisition parameters
        # ----------------------------------------------------

        exposure_ms = self._positive_finite(
            "exposure_ms",
            sensor_config.exposure_ms
        )

        gain = self._positive_finite(
            "gain",
            sensor_config.gain
        )

        conversion_rate = self._positive_finite(
            "electrons_per_unit_ms",
            sensor_config.electrons_per_unit_ms
        )

        full_well = self._positive_finite(
            "full_well_electrons",
            sensor_config.full_well_electrons
        )

        adc_bits = sensor_config.adc_bits

        if (
            isinstance(adc_bits, bool)
            or not isinstance(adc_bits, int)
            or not 1 <= adc_bits <= 24
        ):
            raise ValueError(
                "adc_bits must be an integer from 1 to 24."
            )

        adc_max = (2 ** adc_bits) - 1

        if self.config.min_adc_counts >= adc_max:

            raise ValueError(
                "Minimum ADC count exceeds usable ADC range."
            )

        # ----------------------------------------------------
        # 2. Extract observable sensor measurements
        # ----------------------------------------------------

        image = sensor_result["normalized_output"]

        saturated = sensor_result["saturated"]

        adc_clipped = sensor_result["adc_clipped"]

        if not isinstance(image, torch.Tensor):

            raise ValueError(
                "Sensor image must be a tensor."
            )

        if (
            image.ndim != 3
            or image.shape[0] != 3
            or image.numel() == 0
        ):
            raise ValueError(
                "Expected nonempty RGB tensor (3, H, W)."
            )

        if not torch.is_floating_point(image):

            raise ValueError(
                "Sensor image must be floating point."
            )

        if not torch.isfinite(image).all():

            raise ValueError(
                "Sensor image contains nonfinite values."
            )

        if (image < 0).any() or (image > 1).any():

            raise ValueError(
                "Normalized image must be within [0, 1]."
            )

        for name, mask in (
            ("saturated", saturated),
            ("adc_clipped", adc_clipped)
        ):

            if not isinstance(mask, torch.Tensor):

                raise ValueError(
                    f"{name} must be a tensor."
                )

            if mask.shape != image.shape:

                raise ValueError(
                    f"{name} shape mismatch."
                )

            if mask.dtype != torch.bool:

                raise ValueError(
                    f"{name} must be a boolean tensor."
                )

            if mask.device != image.device:

                raise ValueError(
                    f"{name} device mismatch."
                )

        # ----------------------------------------------------
        # 3. Identify unreliable observations
        # ----------------------------------------------------

        physical_clipping = (
            saturated | adc_clipped
        )

        # A top ADC code may also result from rounding.
        # Exclude conservatively without claiming that
        # physical saturation necessarily occurred.

        adc_ceiling = image >= 1.0

        censored_mask = (
            physical_clipping | adc_ceiling
        )

        min_signal = (
            self.config.min_adc_counts / adc_max
        )

        low_signal_mask = (
            (image < min_signal)
            & ~censored_mask
        )

        valid_mask = (
            ~censored_mask
            & ~low_signal_mask
        )

        # ----------------------------------------------------
        # 4. Invert the linear physical response
        # ----------------------------------------------------

        inverse_scale = (
            full_well
            / (
                exposure_ms
                * conversion_rate
                * gain
            )
        )

        if not math.isfinite(inverse_scale):

            raise ValueError(
                "Invalid inverse response scale."
            )

        raw_estimate = image * inverse_scale

        if not torch.isfinite(raw_estimate).all():

            raise ValueError(
                "Optical intensity reconstruction overflow."
            )

        # Invalid samples are zero-filled.
        # Consumers MUST use valid_mask.

        optical_estimate = torch.where(
            valid_mask,
            raw_estimate,
            torch.zeros_like(raw_estimate)
        )

        # ----------------------------------------------------
        # 5. Quantization uncertainty
        # ----------------------------------------------------

        quantization_error_bound = (
            inverse_scale / (2.0 * adc_max)
        )

        # ----------------------------------------------------
        # 6. Sample-level and pixel-level validity
        # ----------------------------------------------------

        valid_pixel_mask = valid_mask.all(dim=0)

        clipped_pixel_mask = (
            physical_clipping.any(dim=0)
        )

        censored_pixel_mask = (
            censored_mask.any(dim=0)
        )

        # ----------------------------------------------------
        # 7. Return observation and quality metadata
        # ----------------------------------------------------

        return {

            "optical_estimate":
                optical_estimate,

            "valid_mask":
                valid_mask,

            "valid_pixel_mask":
                valid_pixel_mask,

            "physical_clipping_mask":
                physical_clipping,

            "clipped_pixel_mask":
                clipped_pixel_mask,

            "censored_mask":
                censored_mask,

            "censored_pixel_mask":
                censored_pixel_mask,

            "low_signal_mask":
                low_signal_mask,

            "valid_fraction":
                valid_mask.float().mean().item(),

            "valid_pixel_fraction":
                valid_pixel_mask.float().mean().item(),

            "clipped_fraction":
                physical_clipping.float().mean().item(),

            "clipped_pixel_fraction":
                clipped_pixel_mask.float().mean().item(),

            "censored_fraction":
                censored_mask.float().mean().item(),

            "low_signal_fraction":
                low_signal_mask.float().mean().item(),

            "quantization_error_bound":
                quantization_error_bound,

            "inverse_response_scale":
                inverse_scale,

            "exposure_ms":
                exposure_ms,

            "gain":
                gain,

            "adc_max":
                adc_max
        }
