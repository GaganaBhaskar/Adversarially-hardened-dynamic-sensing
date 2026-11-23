
import sys
import torch
import pytest

sys.path.insert(
    0,
    "/content/adversarially-hardened-dynamic-sensing/src"
)

from ahds.module1_environment.optical_environment import (
    generate_clean_scene,
    AttackConfig
)

from ahds.module1_environment.optical_scenarios import (
    ILLUMINATION_PRESETS,
    apply_illumination_condition,
    apply_benign_lighting,
    apply_multiple_attacks,
    generate_temporal_sequence
)


def test_illumination_conditions():
    scene = generate_clean_scene(64, 64)

    for config in ILLUMINATION_PRESETS.values():
        result = apply_illumination_condition(scene, config)

        assert result.shape == scene.shape
        assert torch.isfinite(result).all()


def test_invalid_illumination():
    scene = generate_clean_scene(64, 64)

    from ahds.module1_environment.optical_scenarios import (
        IlluminationConfig
    )

    config = IlluminationConfig(condition="invalid")

    with pytest.raises(ValueError):
        apply_illumination_condition(scene, config)


def test_benign_light_generation():
    scene = generate_clean_scene(64, 64)

    result, light_map = apply_benign_lighting(
        scene,
        intensity=1.0,
        center_x=0.5,
        center_y=0.5
    )

    assert result.shape == scene.shape
    assert light_map.max() > 0
    assert torch.all(result >= scene)


def test_multiple_attacks():
    scene = generate_clean_scene(64, 64)

    attacks = [
        AttackConfig(
            attack_type="laser",
            center_x=0.25,
            center_y=0.5,
            intensity=1.0
        ),
        AttackConfig(
            attack_type="glare",
            center_x=0.75,
            center_y=0.5,
            intensity=0.8
        )
    ]

    result, attack_map = apply_multiple_attacks(
        scene, attacks
    )

    assert result.shape == scene.shape
    assert attack_map.shape == (1, 64, 64)
    assert torch.all(result >= scene)


def test_temporal_sequence():
    scene = generate_clean_scene(64, 64)

    config = AttackConfig(
        attack_type="pulse",
        pulse_frequency=0.25,
        intensity=2.0
    )

    scenes, maps = generate_temporal_sequence(
        scene,
        config,
        num_frames=8
    )

    assert scenes.shape == (8, 3, 64, 64)
    assert maps.shape == (8, 1, 64, 64)

    # The modulated attack should change over time.
    assert not torch.allclose(maps[0], maps[1])

    # Frame 0 and Frame 2 should have equal intensity.
    assert torch.allclose(maps[0], maps[2], atol=1e-6)

    # Pulse repeats after four frames.
    assert torch.allclose(maps[0], maps[4], atol=1e-6)

    # Frame 1 should have greater intensity than Frame 0.
    assert maps[1].max() > maps[0].max()

    # Frame 3 should have approximately zero attack intensity.
    assert maps[3].abs().max() < 1e-5
