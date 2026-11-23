
import pytest
import torch

from ahds.module1_environment.optical_environment import (
    AttackConfig,
    generate_clean_scene,
    apply_optical_attack,
)
from ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor,
)
from ahds.module4_anomaly_detection.concentration_fusion_detector import (
    ConcentrationFusionDetector,
    ConcentrationFusionConfig,
)
from ahds.module4_anomaly_detection.robust_temporal_detector import (
    RobustTemporalDetector,
)
from ahds.module4_anomaly_detection.disturbance_assessor import (
    PhysicalDisturbanceAssessor,
)
from ahds.module4_anomaly_detection.disturbance_state_tracker import (
    DisturbanceStateTracker,
    StateTrackerConfig,
)


@pytest.fixture
def setup_scene():
    scene = generate_clean_scene(256, 256, device="cpu")
    sensor = SensorConfig(
        exposure_ms=10.0,
        gain=1.0,
        electrons_per_unit_ms=100.0,
        full_well_electrons=10000.0,
        read_noise_electrons=3.0,
        adc_bits=12,
    )
    baseline = simulate_sensor(scene, sensor, add_noise=False)
    return scene, sensor, baseline


def make_measurement(scene, sensor, kind, intensity,
                     cx=0.5, cy=0.5, radius=0.1):
    config = AttackConfig(
        attack_type=kind,
        center_x=cx,
        center_y=cy,
        radius=radius,
        intensity=intensity,
    )
    modified, _ = apply_optical_attack(scene, config)
    return simulate_sensor(modified, sensor, add_noise=False)


def test_clean_scene_not_flagged(setup_scene):
    _, _, baseline = setup_scene
    detector = ConcentrationFusionDetector()
    detector.detect(baseline)
    result = detector.detect(baseline)

    assert result["detected"] is False
    assert result["optical_disturbance"] is False


def test_strong_benign_glare_not_attack_candidate(setup_scene):
    scene, sensor, baseline = setup_scene
    measurement = make_measurement(
        scene, sensor, "glare", 10.0
    )

    detector = ConcentrationFusionDetector()
    detector.detect(baseline)
    result = detector.detect(measurement)

    assert result["optical_disturbance"] is True
    assert result["detected"] is False


def test_localized_laser_candidate(setup_scene):
    scene, sensor, baseline = setup_scene
    measurement = make_measurement(
        scene, sensor, "laser", 15.0
    )

    detector = ConcentrationFusionDetector()
    detector.detect(baseline)
    result = detector.detect(measurement)

    assert result["detected"] is True
    assert result["concentration"] >= 0.30


def test_weak_laser_disturbance(setup_scene):
    scene, sensor, baseline = setup_scene
    measurement = make_measurement(
        scene, sensor, "laser", 2.0,
        cx=0.25, cy=0.70, radius=0.05
    )

    assessor = PhysicalDisturbanceAssessor()
    assessor.detect(baseline)
    result = assessor.detect(measurement)

    assert result["disturbance_detected"] is True
    assert result["severity_score"] > 0


def test_global_brightness_compensation(setup_scene):
    scene, sensor, baseline = setup_scene
    bright = simulate_sensor(
        scene * 3.0, sensor, add_noise=False
    )

    detector = RobustTemporalDetector()
    detector.detect(baseline["normalized_output"])
    result = detector.detect(bright["normalized_output"])

    assert result["detected"] is False
    assert result["global_scale"] == pytest.approx(
        3.0, abs=0.05
    )


def test_state_tracker_sustained_attack_and_recovery(setup_scene):
    scene, sensor, baseline = setup_scene
    attack = make_measurement(
        scene, sensor, "laser", 15.0
    )

    assessor = PhysicalDisturbanceAssessor()
    tracker = DisturbanceStateTracker()

    states = []
    frames = [baseline] * 5 + [attack] * 10 + [baseline] * 5

    for measurement in frames:
        assessment = assessor.detect(measurement)
        states.append(tracker.update(assessment)["state"])

    assert states[:5] == ["NORMAL"] * 5
    assert states[5:15] == ["ACTIVE"] * 10
    assert states[15] == "RECOVERY"
    assert states[16:] == ["NORMAL"] * 4


def test_detector_reset(setup_scene):
    _, _, baseline = setup_scene
    detector = ConcentrationFusionDetector()

    detector.detect(baseline)
    detector.reset()

    result = detector.detect(baseline)

    assert result["temporal_result"]["initialized"] is True


def test_invalid_concentration_threshold():
    with pytest.raises(ValueError):
        ConcentrationFusionDetector(
            ConcentrationFusionConfig(
                concentration_threshold=1.5
            )
        )


def test_invalid_recovery_frames():
    with pytest.raises(ValueError):
        DisturbanceStateTracker(
            StateTrackerConfig(recovery_frames=0)
        )


def test_invalid_temporal_input():
    detector = RobustTemporalDetector()

    with pytest.raises(ValueError):
        detector.detect(torch.zeros(1, 32, 32))
