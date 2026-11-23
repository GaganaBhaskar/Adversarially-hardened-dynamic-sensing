
from dataclasses import dataclass
import math
import torch


@dataclass
class AttackConfig:
    attack_type: str = "laser"

    center_x: float = 0.5
    center_y: float = 0.5

    radius: float = 0.12
    intensity: float = 2.0

    pulse_frequency: float = 2.0
    pulse_phase: float = 0.0

    def validate(self):
        valid_types = ["laser", "pulse", "glare"]

        if self.attack_type not in valid_types:
            raise ValueError("Unsupported attack type.")

        if self.radius <= 0:
            raise ValueError("Radius must be positive.")

        if self.intensity < 0:
            raise ValueError("Intensity cannot be negative.")


def generate_clean_scene(
    height=256,
    width=256,
    device="cpu"
):
    """
    Generate a simple synthetic road environment.

    Output:
        Tensor of shape (3, H, W)

    Values represent approximate linear RGB radiance.
    They are not calibrated physical radiance measurements.
    """

    scene = torch.zeros(
        (3, height, width),
        dtype=torch.float32,
        device=device
    )

    # Coordinate grids
    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)

    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Sky
    sky_mask = yy < 0.42

    scene[0][sky_mask] = 0.45
    scene[1][sky_mask] = 0.65
    scene[2][sky_mask] = 0.85

    # Road
    road_mask = yy >= 0.42

    road_value = 0.18 + 0.12 * yy

    for channel in range(3):
        scene[channel][road_mask] = road_value[road_mask]

    # Road boundaries
    left_boundary = 0.25 + 0.12 * (yy - 0.42)
    right_boundary = 0.75 - 0.12 * (yy - 0.42)

    road_edges = (
        (torch.abs(xx - left_boundary) < 0.008) |
        (torch.abs(xx - right_boundary) < 0.008)
    ) & road_mask

    scene[:, road_edges] = 0.8

    # Lane markings
    lane_center = torch.abs(xx - 0.5) < 0.006

    lane_mask = lane_center & road_mask & (
        torch.sin(yy * 75) > 0
    )

    scene[:, lane_mask] = 0.95

    # Vehicle-like object
    vehicle_mask = (
        (xx > 0.62) & (xx < 0.79) &
        (yy > 0.61) & (yy < 0.76)
    )

    scene[0][vehicle_mask] = 0.12
    scene[1][vehicle_mask] = 0.15
    scene[2][vehicle_mask] = 0.18

    # Pedestrian-like silhouette
    px, py = 0.36, 0.68

    head = (
        (xx - px) ** 2 + (yy - (py - 0.055)) ** 2
    ) < 0.00035

    body = (
        (torch.abs(xx - px) < 0.018) &
        (yy > py - 0.04) &
        (yy < py + 0.055)
    )

    legs = (
        (yy >= py + 0.04) &
        (yy <= py + 0.12) &
        (
            (torch.abs(xx - (px - 0.014)) < 0.008) |
            (torch.abs(xx - (px + 0.014)) < 0.008)
        )
    )

    pedestrian_mask = head | body | legs

    scene[:, pedestrian_mask] = 0.04

    return scene.clamp(min=0.0)


def generate_attack_map(
    height,
    width,
    config,
    device="cpu",
    time_step=0
):
    """
    Generate a spatial optical interference map.

    The output is additive incident illumination.
    Sensor clipping and saturation belong to Module 2.
    """

    config.validate()

    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)

    yy, xx = torch.meshgrid(y, x, indexing="ij")

    distance_squared = (
        (xx - config.center_x) ** 2 +
        (yy - config.center_y) ** 2
    )

    sigma = config.radius

    gaussian = torch.exp(
        -distance_squared / (2 * sigma ** 2)
    )

    if config.attack_type == "laser":
        intensity = config.intensity
        attack_map = intensity * gaussian

    elif config.attack_type == "pulse":
        modulation = (
            0.5 + 0.5 * math.sin(
                2 * math.pi *
                config.pulse_frequency *
                time_step +
                config.pulse_phase
            )
        )

        attack_map = (
            config.intensity * modulation * gaussian
        )

    elif config.attack_type == "glare":
        broad_sigma = sigma * 2.5

        broad_glare = torch.exp(
            -distance_squared / (2 * broad_sigma ** 2)
        )

        attack_map = config.intensity * broad_glare

    return attack_map.unsqueeze(0)


def apply_optical_attack(
    clean_scene,
    config,
    time_step=0
):
    """
    Apply optical interference to a clean scene.

    Args:
        clean_scene: Tensor (3, H, W)
        config: AttackConfig
        time_step: Temporal simulation step

    Returns:
        attacked_scene: Tensor (3, H, W)
        attack_map: Tensor (1, H, W)
    """

    if clean_scene.ndim != 3 or clean_scene.shape[0] != 3:
        raise ValueError("Expected an RGB scene with shape (3,H,W).")

    _, height, width = clean_scene.shape

    attack_map = generate_attack_map(
        height=height,
        width=width,
        config=config,
        device=clean_scene.device,
        time_step=time_step
    )

    attacked_scene = clean_scene + attack_map

    return attacked_scene, attack_map
