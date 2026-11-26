
"""
AHDS — M5-DEF-02B-01
DADRC behavioral and state-transition tests.

Isolated controller tests use controlled observation
and disturbance-evidence inputs.

Real sensor integration is tested separately.
"""

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from src.ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor
)

from src.ahds.module5_defense.disturbance_aware_defense import (
    DisturbanceAwareDefenseConfig,
    DisturbanceAwareDefenseController
)


# ============================================================
# TEST FIXTURES
# ============================================================

@pytest.fixture
def controller():

    return DisturbanceAwareDefenseController()


def drive(
    controller,
    exposure,
    status="CLEAR",
    clipping=0.0,
    censoring=0.0,
    spatial=0.0,
    tracker="NORMAL"
):

    """
    Inject controlled observation evidence to exercise
    individual state-machine branches deterministically.
    """

    controller.estimator.estimate = Mock(
        return_value={
            "clipped_fraction": clipping,
            "censored_fraction": censoring
        }
    )

    controller.memory.assess = Mock(
        return_value={
            "status": status,
            "reason": "controlled_test_evidence",
            "changed_fraction": (
                0.1 if status == "PRESENT"
                else None if status == "UNKNOWN"
                else 0.0
            ),
            "global_scale": 1.0,
            "common_valid_fraction": (
                0.1 if status == "UNKNOWN"
                else 1.0
            )
        }
    )

    return controller.update(
        sensor_result={},
        sensor_config=SensorConfig(
            exposure_ms=exposure,
            gain=1.0
        ),
        assessment={
            "saturation_fraction": spatial
        },
        tracker_state=tracker
    )


def activate(controller):

    decision = drive(
        controller,
        exposure=10.0,
        status="PRESENT",
        clipping=0.04,
        censoring=0.04,
        spatial=0.04,
        tracker="ACTIVE"
    )

    assert decision["action"] == "PROTECT"

    return decision["exposure_ms"]


def persistent_probe(controller):

    exposure = activate(controller)

    for _ in range(
        controller.config.persistent_probe_interval
    ):

        decision = drive(
            controller,
            exposure,
            status="PRESENT"
        )

        exposure = decision["exposure_ms"]

    assert decision["action"] == "PROBE"

    assert controller.probe_pending

    return exposure


def clear_probe(controller):

    exposure = activate(controller)

    for _ in range(
        controller.config.clear_confirmation_frames
    ):

        decision = drive(
            controller,
            exposure,
            status="CLEAR"
        )

        exposure = decision["exposure_ms"]

    assert decision["action"] == "PROBE"

    return exposure


# ============================================================
# A. CONFIGURATION VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "changes",
    [
        {"min_exposure_ms": 0.0},
        {"max_exposure_ms": 0.01},
        {"saturation_trigger": 1.1},
        {"reduction_factor": 1.0},
        {"persistent_probe_factor": 1.0},
        {"clear_probe_factor": 1.01},
        {"recovery_factor": 1.01},
        {"persistent_probe_interval": 0},
        {"clear_confirmation_frames": 0},
        {"failed_probe_cooldown": 0},
        {"recovery_factor": float("nan")}
    ]
)
def test_invalid_configuration(changes):

    config = replace(
        DisturbanceAwareDefenseConfig(),
        **changes
    )

    with pytest.raises(ValueError):

        DisturbanceAwareDefenseController(config)


# ============================================================
# B. NORMAL MONITORING
# ============================================================

@pytest.mark.parametrize(
    "status,expected_action",
    [
        ("CLEAR", "HOLD"),
        ("PRESENT", "MONITOR_DISTURBANCE"),
        ("UNKNOWN", "MONITOR_UNKNOWN")
    ]
)
def test_normal_monitoring(
    controller,
    status,
    expected_action
):

    decision = drive(
        controller,
        10.0,
        status=status
    )

    assert decision["action"] == expected_action

    assert decision["exposure_ms"] == 10.0

    assert not decision["protection_active"]


# ============================================================
# C. PHYSICAL PROTECTION
# ============================================================

@pytest.mark.parametrize(
    "signals",
    [
        {"clipping": 0.005},
        {"censoring": 0.005},
        {"spatial": 0.005}
    ]
)
def test_each_unsafe_signal_triggers_protection(
    controller,
    signals
):

    decision = drive(
        controller,
        10.0,
        status="CLEAR",
        **signals
    )

    assert decision["action"] == "PROTECT"

    assert decision["exposure_ms"] == 5.0

    assert decision["protection_active"]

    assert decision["reference_exposure"] == 10.0


def test_repeated_protection_preserves_reference(
    controller
):

    exposure = activate(controller)

    decision = drive(
        controller,
        exposure,
        status="PRESENT",
        clipping=0.04
    )

    assert decision["exposure_ms"] == 2.5

    assert decision["reference_exposure"] == 10.0


def test_minimum_exposure_bound(controller):

    decision = drive(
        controller,
        0.1,
        status="PRESENT",
        clipping=0.5
    )

    assert decision["exposure_ms"] == 0.1

    assert decision["action"] == "PROTECT"


# ============================================================
# D. PERSISTENT DISTURBANCE AND PROBING
# ============================================================

def test_persistent_disturbance_holds_exposure(
    controller
):

    exposure = activate(controller)

    decision = drive(
        controller,
        exposure,
        status="PRESENT"
    )

    assert decision["action"] == "HOLD_PROTECTION"

    assert decision["exposure_ms"] == exposure

    assert decision["protection_active"]


def test_persistent_probe_timing(controller):

    exposure = activate(controller)

    interval = (
        controller.config.persistent_probe_interval
    )

    for _ in range(interval - 1):

        decision = drive(
            controller,
            exposure,
            status="PRESENT"
        )

        assert decision["action"] == "HOLD_PROTECTION"

    decision = drive(
        controller,
        exposure,
        status="PRESENT"
    )

    assert decision["action"] == "PROBE"

    assert decision["exposure_ms"] == pytest.approx(
        5.25
    )

    assert decision["probe_pending"]


def test_successful_persistent_probe(controller):

    exposure = persistent_probe(controller)

    decision = drive(
        controller,
        exposure,
        status="PRESENT"
    )

    assert not decision["probe_pending"]

    assert decision["safe_exposure"] == pytest.approx(
        exposure
    )

    assert decision["protection_active"]

    assert decision["action"] == "HOLD_PROTECTION"


def test_unsafe_probe_failure(controller):

    exposure = persistent_probe(controller)

    decision = drive(
        controller,
        exposure,
        status="PRESENT",
        clipping=0.03
    )

    assert decision["action"] == "PROBE_FAILED"

    assert decision["exposure_ms"] == pytest.approx(
        2.625
    )

    assert not decision["probe_pending"]

    assert decision["cooldown_remaining"] == 8

    assert decision["controller_state"] == "PROTECT"

    assert decision["safe_exposure"] is None


def test_unknown_probe_rolls_back(controller):

    exposure = persistent_probe(controller)

    decision = drive(
        controller,
        exposure,
        status="UNKNOWN"
    )

    assert decision["action"] == "PROBE_FAILED"

    assert decision["exposure_ms"] == 5.0

    assert not decision["probe_pending"]

    assert decision["cooldown_remaining"] == 8


def test_failed_probe_cooldown(controller):

    exposure = persistent_probe(controller)

    failed = drive(
        controller,
        exposure,
        status="UNKNOWN"
    )

    exposure = failed["exposure_ms"]

    cooldown = controller.config.failed_probe_cooldown

    for remaining in range(cooldown - 1, -1, -1):

        decision = drive(
            controller,
            exposure,
            status="CLEAR"
        )

        assert decision["action"] == "HOLD_COOLDOWN"

        assert decision["exposure_ms"] == exposure

        assert (
            decision["cooldown_remaining"]
            == remaining
        )

    decision = drive(
        controller,
        exposure,
        status="CLEAR"
    )

    assert decision["action"] == "CONFIRM_CLEAR"


def test_unsafe_observation_overrides_cooldown(
    controller
):

    exposure = persistent_probe(controller)

    failed = drive(
        controller,
        exposure,
        status="UNKNOWN"
    )

    decision = drive(
        controller,
        failed["exposure_ms"],
        status="CLEAR",
        clipping=0.04
    )

    assert decision["action"] == "PROTECT"

    assert decision["exposure_ms"] == 2.5


# ============================================================
# E. UNCERTAINTY AND CONFLICTING EVIDENCE
# ============================================================

def test_unknown_evidence_holds_protection(
    controller
):

    exposure = activate(controller)

    for _ in range(12):

        decision = drive(
            controller,
            exposure,
            status="UNKNOWN"
        )

        assert decision["action"] == "HOLD_UNKNOWN"

        assert decision["exposure_ms"] == exposure

        assert decision["clear_frames"] == 0

        assert not decision["probe_pending"]


def test_active_tracker_blocks_clear_recovery(
    controller
):

    exposure = activate(controller)

    for _ in range(6):

        decision = drive(
            controller,
            exposure,
            status="CLEAR",
            tracker="ACTIVE"
        )

        assert decision["action"] == "HOLD_CONFLICT"

        assert decision["clear_frames"] == 0

        assert decision["exposure_ms"] == exposure


@pytest.mark.parametrize(
    "interruption",
    ["PRESENT", "UNKNOWN"]
)
def test_interrupted_clear_confirmation(
    controller,
    interruption
):

    exposure = activate(controller)

    for _ in range(2):

        drive(
            controller,
            exposure,
            status="CLEAR"
        )

    assert controller.clear_frames == 2

    drive(
        controller,
        exposure,
        status=interruption
    )

    assert controller.clear_frames == 0

    decision = drive(
        controller,
        exposure,
        status="CLEAR"
    )

    assert decision["action"] == "CONFIRM_CLEAR"

    assert decision["clear_frames"] == 1


# ============================================================
# F. CONTROLLED RECOVERY
# ============================================================

def test_clear_confirmation_and_probe(controller):

    exposure = clear_probe(controller)

    assert exposure == pytest.approx(5.5)

    assert controller.probe_pending

    assert controller.state == "PROBE"


def test_successful_clear_probe_enables_recovery(
    controller
):

    exposure = clear_probe(controller)

    decision = drive(
        controller,
        exposure,
        status="CLEAR"
    )

    assert decision["action"] == "RECOVER"

    assert decision["exposure_ms"] == pytest.approx(
        6.6
    )

    assert decision["probe_pending"]


def test_disturbance_returns_during_recovery(
    controller
):

    exposure = clear_probe(controller)

    decision = drive(
        controller,
        exposure,
        status="PRESENT"
    )

    assert decision["action"] == "HOLD_PROTECTION"

    assert decision["exposure_ms"] == exposure

    assert decision["clear_frames"] == 0

    assert decision["protection_active"]


def test_restoration_requires_observed_reference(
    controller
):

    exposure = clear_probe(controller)

    reached_reference = False

    for _ in range(20):

        decision = drive(
            controller,
            exposure,
            status="CLEAR"
        )

        assert decision["exposure_ms"] <= 10.0

        if decision["action"] == "RESTORED":

            assert exposure == pytest.approx(10.0)

            assert reached_reference

            assert not decision["protection_active"]

            assert not decision["probe_pending"]

            assert decision["controller_state"] == "NORMAL"

            return

        if decision["exposure_ms"] == 10.0:

            reached_reference = True

            assert decision["protection_active"]

            assert decision["probe_pending"]

        exposure = decision["exposure_ms"]

    pytest.fail("Reference exposure was not restored.")


# ============================================================
# G. INPUT AND COMMAND VALIDATION
# ============================================================

@pytest.mark.parametrize(
    "exposure",
    [0.01, 31.0, float("nan"), float("inf")]
)
def test_invalid_current_exposure(
    controller,
    exposure
):

    with pytest.raises(ValueError):

        drive(controller, exposure)


@pytest.mark.parametrize(
    "fraction",
    [-0.1, 1.1, float("nan")]
)
def test_invalid_spatial_saturation(
    controller,
    fraction
):

    with pytest.raises(ValueError):

        drive(
            controller,
            10.0,
            spatial=fraction
        )


def test_invalid_tracker_state(controller):

    with pytest.raises(ValueError):

        drive(
            controller,
            10.0,
            tracker="INVALID"
        )


def test_command_mismatch_rejected(controller):

    activate(controller)

    with pytest.raises(ValueError):

        drive(
            controller,
            6.0,
            status="PRESENT"
        )


# ============================================================
# H. REAL SENSOR REFERENCE AND RESET
# ============================================================

def test_real_sensor_reference_preservation(
    controller
):

    scene = torch.full(
        (3, 32, 32),
        0.5
    )

    config = SensorConfig(
        exposure_ms=10.0,
        gain=1.0
    )

    sensor_result = simulate_sensor(
        scene,
        config,
        add_noise=False
    )

    controller.initialize_reference(
        sensor_result,
        config
    )

    reference = controller.memory.reference.clone()

    mask = controller.memory.reference_mask.clone()

    exposure = activate(controller)

    for _ in range(3):

        drive(
            controller,
            exposure,
            status="UNKNOWN"
        )

    assert torch.equal(
        controller.memory.reference,
        reference
    )

    assert torch.equal(
        controller.memory.reference_mask,
        mask
    )


def test_duplicate_reference_rejected(controller):

    scene = torch.full((3, 32, 32), 0.5)

    config = SensorConfig(
        exposure_ms=10.0,
        gain=1.0
    )

    result = simulate_sensor(
        scene,
        config,
        add_noise=False
    )

    controller.initialize_reference(result, config)

    with pytest.raises(RuntimeError):

        controller.initialize_reference(result, config)


def test_unsafe_startup_reference_rejected(
    controller
):

    scene = torch.full((3, 32, 32), 20.0)

    config = SensorConfig(
        exposure_ms=10.0,
        gain=1.0
    )

    result = simulate_sensor(
        scene,
        config,
        add_noise=False
    )

    with pytest.raises(ValueError):

        controller.initialize_reference(result, config)


def test_controller_reset(controller):

    scene = torch.full((3, 32, 32), 0.5)

    config = SensorConfig(
        exposure_ms=10.0,
        gain=1.0
    )

    result = simulate_sensor(
        scene,
        config,
        add_noise=False
    )

    controller.initialize_reference(result, config)

    persistent_probe(controller)

    controller.reset()

    assert controller.state == "NORMAL"

    assert not controller.protection_active

    assert controller.reference_exposure is None

    assert controller.safe_exposure is None

    assert controller.probe_origin is None

    assert not controller.probe_pending

    assert controller.clear_frames == 0

    assert controller.persistent_frames == 0

    assert controller.cooldown_remaining == 0

    assert controller.last_command is None

    assert controller.memory.reference is None

    assert controller.memory.reference_mask is None
