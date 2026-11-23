
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class ROIConfig:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def validate(self, height, width):
        values = (
            self.x_min, self.y_min,
            self.x_max, self.y_max
        )

        if any(type(v) is not int for v in values):
            raise ValueError("ROI coordinates must be integers.")

        if not (
            0 <= self.x_min < self.x_max <= width
            and 0 <= self.y_min < self.y_max <= height
        ):
            raise ValueError("ROI must lie within image bounds.")


def allocate_readout(image, roi, background_stride=4):
    """
    Simulate ROI-prioritized sensor readout.

    ROI: every pixel is sampled.
    Background: sample every Nth row and column.

    Returns:
        readout image with unsampled locations set to zero,
        sampling mask,
        sampling ratio.

    This models sample selection, not actual hardware readout.
    """

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected image shape (3,H,W).")

    if not isinstance(background_stride, int) or background_stride < 1:
        raise ValueError("background_stride must be a positive integer.")

    _, height, width = image.shape
    roi.validate(height, width)

    mask = torch.zeros(
        (height, width),
        dtype=torch.bool,
        device=image.device
    )

    # Sparse background sampling
    mask[::background_stride, ::background_stride] = True

    # Dense ROI sampling
    mask[
        roi.y_min:roi.y_max,
        roi.x_min:roi.x_max
    ] = True

    readout = image * mask.unsqueeze(0).to(image.dtype)

    sampling_ratio = mask.float().mean().item()

    return {
        "readout": readout,
        "sampling_mask": mask,
        "sampling_ratio": sampling_ratio,
        "bandwidth_reduction": 1.0 - sampling_ratio
    }
