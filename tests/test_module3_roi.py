
import pytest
import torch

from ahds.module3_dynamic_sensing.roi_allocator import (
    ROIConfig,
    allocate_readout
)


def test_roi_output_shape():
    image = torch.ones((3, 64, 64))
    roi = ROIConfig(10, 10, 30, 30)

    result = allocate_readout(image, roi)

    assert result["readout"].shape == image.shape
    assert result["sampling_mask"].shape == (64, 64)


def test_dense_roi_sampling():
    image = torch.ones((3, 64, 64))
    roi = ROIConfig(10, 10, 30, 30)

    result = allocate_readout(image, roi)

    mask = result["sampling_mask"]

    assert mask[10:30, 10:30].all()


def test_sparse_background_sampling():
    image = torch.ones((3, 64, 64))
    roi = ROIConfig(20, 20, 40, 40)

    result = allocate_readout(
        image, roi, background_stride=4
    )

    mask = result["sampling_mask"]

    assert mask[0, 0]
    assert not mask[1, 1]
    assert mask[4, 4]


def test_sampling_ratio():
    image = torch.ones((3, 64, 64))
    roi = ROIConfig(10, 10, 30, 30)

    result = allocate_readout(image, roi)

    expected = (
        result["sampling_mask"].float().mean().item()
    )

    assert result["sampling_ratio"] == pytest.approx(expected)

    assert result["bandwidth_reduction"] == pytest.approx(
        1.0 - expected
    )


def test_invalid_roi():
    image = torch.ones((3, 64, 64))

    roi = ROIConfig(-5, 10, 30, 30)

    with pytest.raises(ValueError):
        allocate_readout(image, roi)


def test_full_frame_roi():
    image = torch.ones((3, 64, 64))
    roi = ROIConfig(0, 0, 64, 64)

    result = allocate_readout(image, roi)

    assert result["sampling_ratio"] == 1.0
    assert result["bandwidth_reduction"] == 0.0
    assert torch.equal(result["readout"], image)
