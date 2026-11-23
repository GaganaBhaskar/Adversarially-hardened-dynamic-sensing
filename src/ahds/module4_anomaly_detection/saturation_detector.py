
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass
class AnomalyConfig:
    min_saturation_ratio: float = 0.005
    local_window: int = 15
    local_density_threshold: float = 0.5

    def validate(self):
        if not math.isfinite(self.min_saturation_ratio):
            raise ValueError("Invalid saturation ratio threshold.")

        if not 0 <= self.min_saturation_ratio <= 1:
            raise ValueError("Saturation ratio must be in [0,1].")

        if (
            not isinstance(self.local_window, int)
            or self.local_window < 1
            or self.local_window % 2 == 0
        ):
            raise ValueError("Local window must be a positive odd integer.")

        if not math.isfinite(self.local_density_threshold):
            raise ValueError("Invalid local density threshold.")

        if not 0 <= self.local_density_threshold <= 1:
            raise ValueError("Local density threshold must be in [0,1].")


class SaturationAnomalyDetector:
    """
    Detect concentrated sensor saturation.

    This is a physical anomaly indicator, not a complete
    adversarial attack classifier.
    """

    def __init__(self, config=None):
        self.config = config or AnomalyConfig()
        self.config.validate()

    def detect(self, sensor_result):
        saturated = sensor_result["saturated"]

        if saturated.ndim != 3 or saturated.shape[0] != 3:
            raise ValueError("Expected saturation tensor (3,H,W).")

        # A spatial pixel is saturated if any RGB channel saturates.
        spatial_mask = saturated.any(dim=0)

        global_ratio = spatial_mask.float().mean().item()

        # Calculate local saturation density.
        density = F.avg_pool2d(
            spatial_mask.float()[None, None],
            kernel_size=self.config.local_window,
            stride=1,
            padding=self.config.local_window // 2
        )[0, 0]

        suspicious_mask = (
            density >= self.config.local_density_threshold
        )

        suspicious_fraction = (
            suspicious_mask.float().mean().item()
        )

        detected = (
            global_ratio >= self.config.min_saturation_ratio
            and suspicious_mask.any().item()
        )

        return {
            "detected": bool(detected),
            "global_saturation_ratio": global_ratio,
            "suspicious_fraction": suspicious_fraction,
            "saturation_mask": spatial_mask,
            "local_density": density,
            "suspicious_mask": suspicious_mask
        }
