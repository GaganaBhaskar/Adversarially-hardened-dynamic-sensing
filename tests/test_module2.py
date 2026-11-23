
import torch
import pytest

from ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor
)


def test_sensor_output_shape():
    scene = torch.full((3, 64, 64), 0.5)

    result = simulate_sensor(
        scene, add_noise=False
    )

    assert result["adc_output"].shape == (3, 64, 64)
    assert result["normalized_output"].shape == (3, 64, 64)


def test_sensor_output_range():
    scene = torch.full((3, 64, 64), 0.5)

    result = simulate_sensor(
        scene, add_noise=False
    )

    assert result["adc_output"].min() >= 0
    assert result["adc_output"].max() <= 4095


def test_full_well_saturation():
    scene = torch.full((3, 64, 64), 100.0)

    result = simulate_sensor(
        scene, add_noise=False
    )

    assert result["saturation_ratio"] == 1.0
    assert result["adc_output"].max() == 4095


def test_no_saturation_for_dark_scene():
    scene = torch.zeros((3, 64, 64))

    result = simulate_sensor(
        scene, add_noise=False
    )

    assert result["saturation_ratio"] == 0.0
    assert result["adc_output"].max() == 0


def test_exposure_increases_signal():
    scene = torch.full((3, 64, 64), 0.5)

    short = simulate_sensor(
        scene,
        SensorConfig(exposure_ms=2.0),
        add_noise=False
    )

    long = simulate_sensor(
        scene,
        SensorConfig(exposure_ms=10.0),
        add_noise=False
    )

    assert (
        long["expected_electrons"].mean()
        > short["expected_electrons"].mean()
    )


def test_noise_reproducibility():
    scene = torch.full((3, 64, 64), 0.5)

    a = simulate_sensor(scene, seed=42)
    b = simulate_sensor(scene, seed=42)
    c = simulate_sensor(scene, seed=43)

    assert torch.equal(a["electrons"], b["electrons"])
    assert not torch.equal(a["electrons"], c["electrons"])


def test_invalid_exposure():
    scene = torch.ones((3, 64, 64))

    with pytest.raises(ValueError):
        simulate_sensor(
            scene,
            SensorConfig(exposure_ms=-1.0)
        )


def test_invalid_input_shape():
    scene = torch.ones((64, 64))

    with pytest.raises(ValueError):
        simulate_sensor(scene)


def test_invalid_negative_radiance():
    scene = torch.full((3, 64, 64), -0.5)

    with pytest.raises(ValueError):
        simulate_sensor(scene)


def test_adc_bit_depth():
    scene = torch.full((3, 64, 64), 100.0)

    result = simulate_sensor(
        scene,
        SensorConfig(adc_bits=8),
        add_noise=False
    )

    assert result["adc_output"].max() == 255
