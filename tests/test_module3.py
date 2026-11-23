
import pytest
import torch

from ahds.module3_dynamic_sensing.exposure_controller import (
    AdaptiveExposureController,
    ControllerConfig
)

from ahds.module3_dynamic_sensing.joint_controller import (
    JointExposureGainController,
    JointControllerConfig
)


def test_exposure_controller_increases_exposure():
    controller = AdaptiveExposureController(
        initial_exposure_ms=5.0
    )

    dark = torch.full((3, 32, 32), 0.05)

    result = controller.update(dark)

    assert result["exposure_ms"] > 5.0


def test_exposure_controller_reduces_exposure():
    controller = AdaptiveExposureController(
        initial_exposure_ms=20.0
    )

    bright = torch.full((3, 32, 32), 0.9)

    result = controller.update(bright)

    assert result["exposure_ms"] < 20.0


def test_exposure_limits():
    controller = AdaptiveExposureController(
        ControllerConfig(
            min_exposure_ms=1.0,
            max_exposure_ms=30.0
        ),
        initial_exposure_ms=30.0
    )

    dark = torch.zeros((3, 32, 32))

    for _ in range(10):
        controller.update(dark)

    assert controller.exposure_ms == 30.0


def test_joint_controller_increases_gain_at_exposure_limit():
    controller = JointExposureGainController(
        initial_exposure_ms=30.0,
        initial_gain=1.0
    )

    dark = torch.full((3, 32, 32), 0.05)

    controller.update(dark)

    assert controller.gain > 1.0


def test_joint_controller_respects_gain_limit():
    controller = JointExposureGainController(
        JointControllerConfig(max_gain=3.0),
        initial_exposure_ms=30.0,
        initial_gain=3.0
    )

    dark = torch.zeros((3, 32, 32))

    for _ in range(10):
        controller.update(dark)

    assert controller.gain <= 3.0


def test_joint_controller_reduces_gain_on_clipping():
    controller = JointExposureGainController(
        initial_exposure_ms=30.0,
        initial_gain=3.0
    )

    bright = torch.full((3, 32, 32), 0.8)

    controller.update(
        bright,
        adc_clip_ratio=0.30
    )

    assert controller.gain < 3.0


def test_joint_controller_invalid_brightness():
    controller = JointExposureGainController()

    invalid = torch.full((3, 32, 32), 1.5)

    with pytest.raises(ValueError):
        controller.update(invalid)


def test_joint_controller_invalid_config():
    with pytest.raises(ValueError):
        JointExposureGainController(
            JointControllerConfig(
                min_gain=3.0,
                max_gain=1.0
            )
        )


def test_joint_controller_invalid_clipping_ratio():
    controller = JointExposureGainController()

    image = torch.full((3, 32, 32), 0.5)

    with pytest.raises(ValueError):
        controller.update(
            image,
            adc_clip_ratio=1.5
        )


def test_joint_controller_exposure_and_gain_bounds():
    controller = JointExposureGainController(
        initial_exposure_ms=5.0,
        initial_gain=1.0
    )

    dark = torch.zeros((3, 32, 32))
    bright = torch.ones((3, 32, 32))

    for _ in range(20):
        controller.update(dark)

    for _ in range(20):
        controller.update(bright)

    assert 1.0 <= controller.exposure_ms <= 30.0
    assert 1.0 <= controller.gain <= 3.0
