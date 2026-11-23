
from dataclasses import dataclass
import torch

from ahds.module1_environment.optical_environment import (
    AttackConfig,
    generate_attack_map
)


@dataclass
class IlluminationConfig:
    condition: str = "daylight"

    # Multiplicative scene illumination
    scale: float = 1.0

    # Additional uniform ambient illumination
    ambient: float = 0.0

    def validate(self):
        if self.condition not in [
            "daylight",
            "dusk",
            "night",
            "low_light"
        ]:
            raise ValueError("Invalid illumination condition.")

        if self.scale < 0 or self.ambient < 0:
            raise ValueError("Illumination values must be nonnegative.")


ILLUMINATION_PRESETS = {
    "daylight": IlluminationConfig(
        "daylight", 1.0, 0.02
    ),
    "dusk": IlluminationConfig(
        "dusk", 0.45, 0.01
    ),
    "night": IlluminationConfig(
        "night", 0.22, 0.005
    ),
    "low_light": IlluminationConfig(
        "low_light", 0.08, 0.002
    )
}


def apply_illumination_condition(scene, config):
    """
    Apply an environmental illumination condition.

    Output remains unclipped. This represents an approximate
    normalized optical input, not calibrated radiance.
    """

    config.validate()

    return (
        scene * config.scale + config.ambient
    ).clamp(min=0.0)


def generate_benign_light(
    height,
    width,
    center_x=0.5,
    center_y=0.5,
    intensity=0.5,
    radius=0.08,
    device="cpu"
):
    """
    Generate a legitimate localized light source.

    Examples:
    - Vehicle headlights
    - Streetlights
    - Strong reflections

    This is an environmental hard-negative, not an attack.
    """

    if intensity < 0 or radius <= 0:
        raise ValueError("Invalid benign light parameters.")

    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)

    yy, xx = torch.meshgrid(y, x, indexing="ij")

    distance_squared = (
        (xx - center_x) ** 2 +
        (yy - center_y) ** 2
    )

    light = intensity * torch.exp(
        -distance_squared / (2 * radius ** 2)
    )

    return light.unsqueeze(0)


def apply_benign_lighting(scene, **kwargs):
    """
    Apply legitimate environmental illumination.

    Returns:
        illuminated_scene
        benign_light_map
    """

    _, height, width = scene.shape

    light_map = generate_benign_light(
        height,
        width,
        device=scene.device,
        **kwargs
    )

    return scene + light_map, light_map


def apply_multiple_attacks(
    clean_scene,
    attack_configs,
    time_step=0
):
    """
    Apply multiple independent optical attacks.

    Returns:
        attacked_scene
        combined_attack_map
    """

    if not attack_configs:
        raise ValueError("At least one attack is required.")

    _, height, width = clean_scene.shape

    combined_map = torch.zeros(
        (1, height, width),
        dtype=clean_scene.dtype,
        device=clean_scene.device
    )

    for config in attack_configs:

        attack_map = generate_attack_map(
            height,
            width,
            config,
            device=clean_scene.device,
            time_step=time_step
        )

        combined_map = combined_map + attack_map

    attacked_scene = clean_scene + combined_map

    return attacked_scene, combined_map


def generate_temporal_sequence(
    clean_scene,
    attack_config,
    num_frames=10
):
    """
    Generate successive optical frames under one attack.

    Returns:
        scenes: (T, 3, H, W)
        attack_maps: (T, 1, H, W)
    """

    if num_frames < 1:
        raise ValueError("num_frames must be positive.")

    scenes = []
    attack_maps = []

    for t in range(num_frames):

        attack_map = generate_attack_map(
            clean_scene.shape[1],
            clean_scene.shape[2],
            attack_config,
            device=clean_scene.device,
            time_step=t
        )

        scenes.append(clean_scene + attack_map)
        attack_maps.append(attack_map)

    return (
        torch.stack(scenes),
        torch.stack(attack_maps)
    )
