
from dataclasses import dataclass
import math
import torch

from ahds.module4_anomaly_detection.saturation_detector import (
    SaturationAnomalyDetector
)

from ahds.module4_anomaly_detection.robust_temporal_detector import (
    RobustTemporalDetector
)


@dataclass
class DisturbanceConfig:
    residual_threshold: float = 0.15
    minimum_changed_fraction: float = 0.001
    minimum_saturation_fraction: float = 0.001

    def validate(self):
        values = (
            self.residual_threshold,
            self.minimum_changed_fraction,
            self.minimum_saturation_fraction
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Parameters must be finite.")

        if not 0 < self.residual_threshold <= 1:
            raise ValueError("Invalid residual threshold.")

        if not 0 <= self.minimum_changed_fraction <= 1:
            raise ValueError("Invalid changed fraction.")

        if not 0 <= self.minimum_saturation_fraction <= 1:
            raise ValueError("Invalid saturation fraction.")


class PhysicalDisturbanceAssessor:

    def __init__(self, config=None):
        self.config = config or DisturbanceConfig()
        self.config.validate()

        self.spatial = SaturationAnomalyDetector()
        self.temporal = RobustTemporalDetector()

    def reset(self):
        self.temporal.reset()

    def detect(self, sensor_result):

        spatial_result = self.spatial.detect(sensor_result)

        temporal_result = self.temporal.detect(
            sensor_result["normalized_output"]
        )

        residual = temporal_result["difference_map"]

        change_mask = (
            residual >= self.config.residual_threshold
        )

        changed_fraction = (
            change_mask.float().mean().item()
        )

        saturation_fraction = spatial_result[
            "global_saturation_ratio"
        ]

        mean_residual = residual.mean().item()
        max_residual = residual.max().item()

        temporal_disturbance = (
            changed_fraction >=
            self.config.minimum_changed_fraction
            and bool(change_mask.any().item())
        )

        saturation_disturbance = (
            saturation_fraction >=
            self.config.minimum_saturation_fraction
            and bool(spatial_result["saturation_mask"].any().item())
        )

        disturbance_detected = (
            temporal_disturbance or saturation_disturbance
        )

        # Descriptive severity proxy, not a calibrated
        # probability of damage or perception failure.
        severity_score = min(
            1.0,
            0.5 * changed_fraction
            + 0.3 * saturation_fraction
            + 0.2 * mean_residual
        )

        localization_mask = (
            change_mask | spatial_result["saturation_mask"]
        )

        return {
            "disturbance_detected": bool(disturbance_detected),
            "severity_score": severity_score,
            "changed_fraction": changed_fraction,
            "saturation_fraction": saturation_fraction,
            "mean_residual": mean_residual,
            "max_residual": max_residual,
            "localization_mask": localization_mask,
            "temporal_disturbance": bool(temporal_disturbance),
            "saturation_disturbance": bool(saturation_disturbance),
            "spatial_result": spatial_result,
            "temporal_result": temporal_result
        }
