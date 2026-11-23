
import sys
import torch

sys.path.insert(
    0,
    "/content/adversarially-hardened-dynamic-sensing/src"
)

from ahds.module1_environment.optical_environment import (
    AttackConfig,
    generate_clean_scene,
    generate_attack_map,
    apply_optical_attack
)


def test_clean_scene_shape():
    scene = generate_clean_scene(128, 128)

    assert scene.shape == (3, 128, 128)
    assert torch.isfinite(scene).all()


def test_attack_map_shape():
    config = AttackConfig()

    attack = generate_attack_map(
        128, 128, config
    )

    assert attack.shape == (1, 128, 128)
    assert torch.isfinite(attack).all()


def test_attack_increases_illumination():
    scene = generate_clean_scene(128, 128)

    config = AttackConfig(intensity=2.0)

    attacked, attack_map = apply_optical_attack(
        scene, config
    )

    assert torch.all(attacked >= scene)
    assert attack_map.max() > 0


def test_invalid_attack():
    config = AttackConfig(attack_type="invalid")

    try:
        config.validate()
        assert False
    except ValueError:
        pass
