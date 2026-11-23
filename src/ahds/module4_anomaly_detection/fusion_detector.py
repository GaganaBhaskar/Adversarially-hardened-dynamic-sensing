
import torch

from ahds.module4_anomaly_detection.saturation_detector import (
    SaturationAnomalyDetector,
    AnomalyConfig
)

from ahds.module4_anomaly_detection.temporal_detector import (
    TemporalAnomalyDetector,
    TemporalConfig
)


class PhysicalAnomalyFusionDetector:
    """
    Combines spatial saturation and temporal-change indicators.

    OR fusion provides sensitivity to either indicator.
    AND fusion requires both indicators.

    Neither mode establishes adversarial intent.
    """

    def __init__(
        self,
        anomaly_config=None,
        temporal_config=None,
        fusion_mode="or"
    ):
        if fusion_mode not in ("or", "and"):
            raise ValueError("fusion_mode must be 'or' or 'and'.")

        self.spatial = SaturationAnomalyDetector(
            anomaly_config or AnomalyConfig()
        )

        self.temporal = TemporalAnomalyDetector(
            temporal_config or TemporalConfig()
        )

        self.fusion_mode = fusion_mode

    def reset(self):
        self.temporal.reset()

    def detect(self, sensor_result):
        spatial = self.spatial.detect(sensor_result)

        temporal = self.temporal.detect(
            sensor_result["normalized_output"]
        )

        spatial_flag = spatial["detected"]
        temporal_flag = temporal["detected"]

        if self.fusion_mode == "or":
            detected = spatial_flag or temporal_flag
            combined_mask = (
                spatial["suspicious_mask"]
                | temporal["change_mask"]
            )
        else:
            detected = spatial_flag and temporal_flag
            combined_mask = (
                spatial["suspicious_mask"]
                & temporal["change_mask"]
            )

        return {
            "detected": bool(detected),
            "spatial_detected": spatial_flag,
            "temporal_detected": temporal_flag,
            "combined_mask": combined_mask,
            "spatial_result": spatial,
            "temporal_result": temporal
        }
