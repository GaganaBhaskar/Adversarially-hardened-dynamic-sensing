
from dataclasses import dataclass
import math

from ahds.module4_anomaly_detection.saturation_detector import (
    SaturationAnomalyDetector
)
from ahds.module4_anomaly_detection.robust_temporal_detector import (
    RobustTemporalDetector
)


@dataclass
class ConcentrationFusionConfig:
    concentration_threshold: float = 0.30
    top_fraction: float = 0.05
    min_residual_energy: float = 1e-6

    def validate(self):
        values = (
            self.concentration_threshold,
            self.top_fraction,
            self.min_residual_energy
        )

        if not all(math.isfinite(v) for v in values):
            raise ValueError("Parameters must be finite.")

        if not 0 <= self.concentration_threshold <= 1:
            raise ValueError("Invalid concentration threshold.")

        if not 0 < self.top_fraction <= 1:
            raise ValueError("Invalid top fraction.")

        if self.min_residual_energy <= 0:
            raise ValueError("Invalid residual energy threshold.")


class ConcentrationFusionDetector:

    def __init__(self, config=None):
        self.config = config or ConcentrationFusionConfig()
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

        residual = (
            temporal_result["difference_map"]
            .flatten()
            .float()
        )

        k = max(
            1,
            math.ceil(
                residual.numel() * self.config.top_fraction
            )
        )

        total_energy = residual.sum().item()

        if total_energy > self.config.min_residual_energy:
            top_energy = residual.topk(k).values.sum().item()
            concentration = top_energy / total_energy
        else:
            concentration = 0.0

        spatial_flag = bool(spatial_result["detected"])
        temporal_flag = bool(temporal_result["detected"])

        # Sensor-reliability warning:
        # This is not necessarily an attack.
        optical_disturbance = (
            spatial_flag or temporal_flag
        )

        # Provisional concentrated anomaly candidate.
        concentrated_candidate = (
            temporal_flag
            and concentration >=
            self.config.concentration_threshold
        )

        return {
            # Backward-compatible candidate decision.
            "detected": bool(concentrated_candidate),

            "optical_disturbance": bool(optical_disturbance),
            "concentrated_candidate": bool(concentrated_candidate),

            "spatial_detected": spatial_flag,
            "temporal_detected": temporal_flag,
            "concentration": concentration,

            "spatial_result": spatial_result,
            "temporal_result": temporal_result
        }
