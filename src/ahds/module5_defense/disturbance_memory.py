
"""
AHDS Module 5
Exposure-Aware Disturbance Memory.

Compares exposure-normalized optical observations
against an explicitly initialized reference.

No simulated attack labels or attack maps are used.
"""

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class DisturbanceMemoryConfig:

    absolute_threshold: float = 0.05

    relative_threshold: float = 0.20

    minimum_changed_fraction: float = 0.005

    minimum_common_fraction: float = 0.50

    minimum_reference_intensity: float = 0.02

    minimum_global_scale: float = 0.10

    maximum_global_scale: float = 10.0

    maximum_censored_fraction: float = 0.005

    def validate(self):

        for name, value in vars(self).items():

            if not math.isfinite(value) or value <= 0:

                raise ValueError(
                    f"{name} must be positive and finite."
                )

        for name in (
            "minimum_changed_fraction",
            "minimum_common_fraction",
            "maximum_censored_fraction"
        ):

            if getattr(self, name) > 1:

                raise ValueError(
                    f"{name} must not exceed 1."
                )

        if not (
            self.minimum_global_scale
            <= 1.0
            <= self.maximum_global_scale
        ):

            raise ValueError(
                "Global scale bounds must include 1."
            )


class ExposureAwareDisturbanceMemory:

    def __init__(self, config=None):

        self.config = (
            config
            if config is not None
            else DisturbanceMemoryConfig()
        )

        self.config.validate()

        self.reset()

    def reset(self):

        self.reference = None

        self.reference_mask = None

        self.reference_uncertainty = None

    @staticmethod
    def _extract(observation):

        image = observation["optical_estimate"]

        mask = observation["valid_mask"]

        censored = observation["censored_mask"]

        uncertainty = float(
            observation["quantization_error_bound"]
        )

        if (
            image.ndim != 3
            or image.shape[0] != 3
            or image.numel() == 0
        ):

            raise ValueError(
                "Expected nonempty RGB observation."
            )

        if not torch.is_floating_point(image):

            raise ValueError(
                "Optical estimate must be floating point."
            )

        if (
            not torch.isfinite(image).all()
            or (image < 0).any()
        ):

            raise ValueError(
                "Invalid optical estimate."
            )

        for item in (mask, censored):

            if (
                item.shape != image.shape
                or item.dtype != torch.bool
                or item.device != image.device
            ):

                raise ValueError(
                    "Invalid observation mask."
                )

        if (mask & censored).any():

            raise ValueError(
                "Censored samples cannot be valid."
            )

        if (
            not math.isfinite(uncertainty)
            or uncertainty < 0
        ):

            raise ValueError(
                "Invalid quantization uncertainty."
            )

        return image, mask, censored, uncertainty

    @torch.no_grad()
    def initialize_reference(self, observation):

        if self.reference is not None:

            raise RuntimeError(
                "Reference already initialized. "
                "Explicit reset required."
            )

        image, mask, censored, uncertainty = (
            self._extract(observation)
        )

        eligible = (
            mask
            & (
                image
                >= self.config.minimum_reference_intensity
            )
        )

        if (
            eligible.float().mean().item()
            < self.config.minimum_common_fraction
        ):

            raise ValueError(
                "Insufficient reference coverage."
            )

        if (
            censored.float().mean().item()
            > self.config.maximum_censored_fraction
        ):

            raise ValueError(
                "Reference contains excessive censoring."
            )

        # Clone to prevent external mutation.

        self.reference = image.detach().clone()

        self.reference_mask = mask.detach().clone()

        self.reference_uncertainty = uncertainty

    @torch.no_grad()
    def assess(self, observation):

        image, mask, censored, uncertainty = (
            self._extract(observation)
        )

        censored_fraction = (
            censored.float().mean().item()
        )

        result = {
            "status": "UNKNOWN",
            "reason": "reference_unavailable",
            "global_scale": None,
            "common_valid_fraction": 0.0,
            "scale_support_fraction": 0.0,
            "changed_fraction": None,
            "changed_fraction_of_valid": None,
            "mean_positive_residual": None,
            "censored_fraction": censored_fraction,
            "candidate_mask": None,
            "residual_map": None
        }

        if self.reference is None:

            return result

        if (
            image.shape != self.reference.shape
            or image.device != self.reference.device
        ):

            raise ValueError(
                "Observation/reference mismatch."
            )

        common = mask & self.reference_mask

        coverage = common.float().mean().item()

        result["common_valid_fraction"] = coverage

        if coverage < self.config.minimum_common_fraction:

            result["reason"] = (
                "insufficient_common_coverage"
            )

            return result

        scale_support = (
            common
            & (
                self.reference
                >= self.config.minimum_reference_intensity
            )
        )

        support_fraction = (
            scale_support.float().mean().item()
        )

        result["scale_support_fraction"] = (
            support_fraction
        )

        if (
            support_fraction
            < self.config.minimum_common_fraction
        ):

            result["reason"] = (
                "insufficient_scale_support"
            )

            return result

        ratios = (
            image[scale_support]
            / self.reference[scale_support]
        )

        scale = ratios.median().item()

        result["global_scale"] = scale

        if (
            not math.isfinite(scale)
            or scale < self.config.minimum_global_scale
            or scale > self.config.maximum_global_scale
        ):

            result["reason"] = (
                "global_scale_out_of_range"
            )

            return result

        predicted = self.reference * scale

        residual = torch.where(
            common,
            (image - predicted).clamp_min(0),
            torch.zeros_like(image)
        )

        combined_uncertainty = (
            uncertainty
            + scale * self.reference_uncertainty
        )

        threshold = (
            self.config.absolute_threshold
            + self.config.relative_threshold * predicted
            + combined_uncertainty
        )

        candidate = (
            common
            & (residual > threshold)
        )

        changed_fraction = (
            candidate.float().mean().item()
        )

        changed_fraction_of_valid = (
            candidate.sum().item()
            / common.sum().item()
        )

        result.update({
            "changed_fraction":
                changed_fraction,

            "changed_fraction_of_valid":
                changed_fraction_of_valid,

            "mean_positive_residual":
                residual[common].mean().item(),

            "candidate_mask":
                candidate,

            "residual_map":
                residual
        })

        # Positive evidence can establish PRESENT
        # even when some other samples are censored.

        if (
            changed_fraction
            >= self.config.minimum_changed_fraction
        ):

            result["status"] = "PRESENT"

            result["reason"] = (
                "positive_optical_residual"
            )

        elif (
            censored_fraction
            > self.config.maximum_censored_fraction
        ):

            result["status"] = "UNKNOWN"

            result["reason"] = (
                "excessive_censoring"
            )

        else:

            result["status"] = "CLEAR"

            result["reason"] = (
                "residual_below_threshold"
            )

        return result
