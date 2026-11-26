
"""
AHDS — Module 5 Defensive Controller Tests

Task: M5-REC-02A
Purpose: Behavioral validation of recovered and
reconstructed defensive controller implementations.
"""

import pytest

from src.ahds.module5_defense.exposure_protection import (
    ExposureProtectionConfig,
    ExposureProtectionController,
)

from src.ahds.module5_defense.state_aware_exposure import (
    StateAwareExposureConfig,
    StateAwareExposureController,
)

from src.ahds.module5_defense.probed_exposure import (
    ProbedExposureConfig,
    ProbedExposureController,
)


def assessment(saturation=0.0):
    return {
        "saturation_fraction": saturation
    }


def activate_probe_controller():

    controller = ProbedExposureController()

    result = controller.update(
        10.0, assessment(0.03)
    )

    return controller, result["exposure_ms"]


def schedule_probe():

    controller, exposure = activate_probe_controller()

    for _ in range(
        controller.config.probe_interval_frames
    ):

        result = controller.update(
            exposure, assessment()
        )

        exposure = result["exposure_ms"]

    assert result["action"] == "PROBE"

    return controller, exposure


# ============================================================
# ORIGINAL EXPOSURE PROTECTION
# ============================================================


# M5-T01
def test_original_normal_hold():

    controller = ExposureProtectionController()

    result = controller.update(
        10.0, assessment()
    )

    assert result["action"] == "HOLD"
    assert result["exposure_ms"] == pytest.approx(10.0)


# M5-T02
def test_original_immediate_protection():

    controller = ExposureProtectionController()

    result = controller.update(
        10.0, assessment(0.03)
    )

    assert result["action"] == "PROTECT"
    assert result["exposure_ms"] == pytest.approx(5.0)
    assert result["clear_frames"] == 0


# M5-T03
def test_original_delayed_recovery():

    controller = ExposureProtectionController()

    exposure = controller.update(
        10.0, assessment(0.03)
    )["exposure_ms"]

    for _ in range(2):

        result = controller.update(
            exposure, assessment()
        )

        assert result["action"] == "HOLD"
        assert result["exposure_ms"] == pytest.approx(5.0)

    result = controller.update(
        exposure, assessment()
    )

    assert result["action"] == "RECOVER"
    assert result["exposure_ms"] == pytest.approx(5.5)


# M5-T04
def test_original_minimum_exposure():

    controller = ExposureProtectionController()

    result = controller.update(
        0.1, assessment(0.5)
    )

    assert result["exposure_ms"] == pytest.approx(0.1)


# M5-T05
def test_original_maximum_exposure():

    controller = ExposureProtectionController()

    exposure = 30.0

    for _ in range(10):

        result = controller.update(
            exposure, assessment()
        )

        exposure = result["exposure_ms"]

        assert exposure <= 30.0

    assert exposure == pytest.approx(30.0)


# M5-T06
def test_original_reset():

    controller = ExposureProtectionController()

    controller.update(10.0, assessment())
    controller.update(10.0, assessment())

    assert controller.clear_frames == 2

    controller.reset()

    assert controller.clear_frames == 0


# ============================================================
# STATE-AWARE EXPOSURE PROTECTION
# ============================================================


# M5-T07
def test_state_aware_normal_operation():

    controller = StateAwareExposureController()

    result = controller.update(
        10.0, assessment(), "NORMAL"
    )

    assert result["action"] == "HOLD"
    assert not result["protection_active"]
    assert result["exposure_ms"] == pytest.approx(10.0)


# M5-T08
def test_state_aware_protection_activation():

    controller = StateAwareExposureController()

    result = controller.update(
        10.0, assessment(0.03), "ACTIVE"
    )

    assert result["action"] == "PROTECT"
    assert result["protection_active"]
    assert result["reference_exposure"] == pytest.approx(10.0)
    assert result["exposure_ms"] == pytest.approx(5.0)


# M5-T09
def test_state_aware_disturbance_hold():

    controller = StateAwareExposureController()

    controller.update(
        10.0, assessment(0.03), "ACTIVE"
    )

    for state in (
        "ACTIVE",
        "RECOVERY",
        "TRANSIENT",
    ):

        result = controller.update(
            5.0, assessment(), state
        )

        assert result["action"] == "HOLD_PROTECTION"
        assert result["exposure_ms"] == pytest.approx(5.0)


# M5-T10
def test_state_aware_reference_restoration():

    controller = StateAwareExposureController()

    result = controller.update(
        10.0, assessment(0.03), "ACTIVE"
    )

    exposure = result["exposure_ms"]

    for _ in range(20):

        result = controller.update(
            exposure, assessment(), "NORMAL"
        )

        exposure = result["exposure_ms"]

        assert exposure <= 10.0

        if result["action"] == "RESTORED":
            break

    assert result["action"] == "RESTORED"
    assert exposure == pytest.approx(10.0)
    assert not result["protection_active"]
    assert result["reference_exposure"] is None


# M5-T11
def test_state_aware_reset():

    controller = StateAwareExposureController()

    controller.update(
        10.0, assessment(0.03), "ACTIVE"
    )

    controller.reset()

    assert not controller.protection_active
    assert controller.reference_exposure is None


# ============================================================
# PROBE-BASED EXPOSURE PROTECTION
# ============================================================


# M5-T12
def test_probe_normal_operation():

    controller = ProbedExposureController()

    result = controller.update(
        10.0, assessment()
    )

    assert result["action"] == "HOLD"
    assert not result["protection_active"]
    assert result["exposure_ms"] == pytest.approx(10.0)


# M5-T13
def test_probe_immediate_protection():

    controller = ProbedExposureController()

    result = controller.update(
        10.0, assessment(0.03)
    )

    assert result["action"] == "PROTECT"
    assert result["protection_active"]
    assert result["exposure_ms"] == pytest.approx(5.0)
    assert result["reference_exposure"] == pytest.approx(10.0)


# M5-T14
def test_probe_scheduling_interval():

    controller, exposure = activate_probe_controller()

    for _ in range(3):

        result = controller.update(
            exposure, assessment()
        )

        assert result["action"] == "HOLD_PROTECTION"
        assert result["exposure_ms"] == pytest.approx(5.0)

    result = controller.update(
        exposure, assessment()
    )

    assert result["action"] == "PROBE"
    assert result["probe_pending"]
    assert result["exposure_ms"] == pytest.approx(5.25)


# M5-T15
def test_probe_successful_acceptance():

    controller, exposure = schedule_probe()

    result = controller.update(
        exposure, assessment(0.001)
    )

    assert not result["probe_pending"]
    assert result["safe_exposure"] == pytest.approx(5.25)
    assert result["exposure_ms"] == pytest.approx(5.25)


# M5-T16
def test_probe_failed_rollback():

    controller, exposure = schedule_probe()

    result = controller.update(
        exposure, assessment(0.03)
    )

    assert result["action"] == "PROBE_FAILED"
    assert result["exposure_ms"] == pytest.approx(5.0)
    assert not result["probe_pending"]
    assert result["cooldown_remaining"] == 8


# M5-T17
def test_probe_failed_cooldown():

    controller, exposure = schedule_probe()

    result = controller.update(
        exposure, assessment(0.03)
    )

    exposure = result["exposure_ms"]

    for _ in range(8):

        result = controller.update(
            exposure, assessment()
        )

        assert result["action"] == "HOLD_PROTECTION"
        assert result["exposure_ms"] == pytest.approx(5.0)

    assert result["cooldown_remaining"] == 0

    # A fresh clear-frame interval is required.
    for _ in range(3):

        result = controller.update(
            exposure, assessment()
        )

        assert result["action"] == "HOLD_PROTECTION"

    result = controller.update(
        exposure, assessment()
    )

    assert result["action"] == "PROBE"


# M5-T18
def test_probe_persistent_saturation():

    controller = ProbedExposureController()

    exposure = 10.0

    for _ in range(20):

        result = controller.update(
            exposure, assessment(0.5)
        )

        exposure = result["exposure_ms"]

        assert result["action"] == "PROTECT"
        assert 0.1 <= exposure <= 10.0

    assert exposure == pytest.approx(0.1)


# M5-T19
def test_probe_reference_exposure_limit():

    controller, exposure = activate_probe_controller()

    for _ in range(100):

        result = controller.update(
            exposure, assessment()
        )

        exposure = result["exposure_ms"]

        assert exposure <= 10.0

        if result["action"] == "RESTORED":
            break

    assert result["action"] == "RESTORED"
    assert exposure == pytest.approx(10.0)
    assert not result["protection_active"]


# M5-T20
def test_probe_reset():

    controller, exposure = schedule_probe()

    controller.reset()

    assert not controller.protection_active
    assert not controller.probe_pending
    assert controller.reference_exposure is None
    assert controller.safe_exposure is None
    assert controller.clear_frames == 0
    assert controller.cooldown_remaining == 0


# ============================================================
# INPUT AND CONFIGURATION VALIDATION
# ============================================================


# M5-T21
def test_invalid_saturation_rejected():

    for controller in (
        ExposureProtectionController(),
        StateAwareExposureController(),
        ProbedExposureController(),
    ):

        for value in (
            -0.1, 1.1, float("nan"), float("inf")
        ):

            with pytest.raises(ValueError):

                if isinstance(
                    controller,
                    StateAwareExposureController
                ):

                    controller.update(
                        10.0,
                        assessment(value),
                        "NORMAL"
                    )

                else:

                    controller.update(
                        10.0, assessment(value)
                    )


# M5-T22
def test_invalid_exposure_rejected():

    for controller in (
        ExposureProtectionController(),
        StateAwareExposureController(),
        ProbedExposureController(),
    ):

        for exposure in (
            0.01, 31.0, float("nan"), float("inf")
        ):

            with pytest.raises(ValueError):

                if isinstance(
                    controller,
                    StateAwareExposureController
                ):

                    controller.update(
                        exposure,
                        assessment(),
                        "NORMAL"
                    )

                else:

                    controller.update(
                        exposure, assessment()
                    )


# M5-T23
def test_invalid_configuration_rejected():

    with pytest.raises(ValueError):

        ExposureProtectionController(
            ExposureProtectionConfig(
                reduction_factor=1.5
            )
        )

    with pytest.raises(ValueError):

        StateAwareExposureController(
            StateAwareExposureConfig(
                recovery_factor=0.5
            )
        )

    with pytest.raises(ValueError):

        ProbedExposureController(
            ProbedExposureConfig(
                probe_interval_frames=0
            )
        )

    with pytest.raises(ValueError):

        ProbedExposureController(
            ProbedExposureConfig(
                failed_probe_cooldown=-1
            )
        )


# M5-T24
def test_invalid_disturbance_state_rejected():

    controller = StateAwareExposureController()

    with pytest.raises(ValueError):

        controller.update(
            10.0,
            assessment(),
            "INVALID_STATE"
        )
