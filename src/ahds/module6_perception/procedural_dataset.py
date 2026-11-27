
"""
AHDS procedural semantic perception benchmark.

Synthetic linear RGB radiance and visible semantic labels
are rendered together from scene geometry.

No sensor observations, disturbance maps or controller
states are used to construct semantic ground truth.

The original Module 1 environment remains unchanged.
"""

from functools import lru_cache

import numpy as np
import torch

from torch.utils.data import Dataset

from ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor,
)

from ahds.module3_dynamic_sensing.roi_allocator import (
    allocate_readout,
)

from ahds.integration.roi_closed_loop import (
    AdaptiveROIConfig,
)


CLASS_NAMES = (
    "background",
    "lane_markings",
    "vehicle",
    "pedestrian",
)

NUM_CLASSES = len(CLASS_NAMES)

SPLITS = {
    "train": (100000, 800),
    "validation": (200000, 160),
    "test": (300000, 160),
}


@lru_cache(maxsize=4)
def coordinate_grid(height, width):

    y = torch.linspace(0, 1, height)
    x = torch.linspace(0, 1, width)

    return torch.meshgrid(y, x, indexing="ij")


def render_scene(seed, height=256, width=256):

    if height < 64 or width < 64:
        raise ValueError("Minimum scene resolution is 64x64.")

    rng = np.random.default_rng(seed)

    yy, xx = coordinate_grid(height, width)

    scene = torch.zeros(
        (3, height, width),
        dtype=torch.float32,
    )

    labels = torch.zeros(
        (height, width),
        dtype=torch.long,
    )

    def paint(mask, color, class_id):

        color = torch.tensor(
            color,
            dtype=torch.float32,
        ).reshape(3, 1)

        scene[:, mask] = color
        labels[mask] = class_id

    # -------------------------------------------------------------
    # Road and sky geometry
    # -------------------------------------------------------------

    horizon = float(rng.uniform(0.35, 0.46))

    road_center = float(rng.uniform(0.44, 0.56))

    half_width = float(rng.uniform(0.22, 0.29))

    boundary_slope = float(rng.uniform(-0.10, 0.16))

    sky = yy < horizon
    road = ~sky

    sky_color = [
        rng.uniform(0.35, 0.55),
        rng.uniform(0.50, 0.75),
        rng.uniform(0.70, 0.95),
    ]

    paint(sky, sky_color, 0)

    road_base = float(rng.uniform(0.14, 0.23))
    road_gradient = float(rng.uniform(0.08, 0.15))

    road_value = road_base + road_gradient * yy

    for channel in range(3):
        scene[channel, road] = road_value[road]

    # -------------------------------------------------------------
    # Lane markings and road boundaries
    # -------------------------------------------------------------

    left = (
        road_center - half_width
        + boundary_slope * (yy - horizon)
    )

    right = (
        road_center + half_width
        - boundary_slope * (yy - horizon)
    )

    edge_width = float(rng.uniform(0.005, 0.010))

    edge_mask = (
        ((xx - left).abs() < edge_width)
        |
        ((xx - right).abs() < edge_width)
    ) & road

    edge_value = float(rng.uniform(0.70, 0.90))

    paint(edge_mask, [edge_value] * 3, 1)

    lane_width = float(rng.uniform(0.004, 0.009))
    lane_frequency = float(rng.uniform(55, 90))
    lane_phase = float(rng.uniform(0, 2 * np.pi))

    lane_mask = (
        ((xx - road_center).abs() < lane_width)
        & road
        & (torch.sin(yy * lane_frequency + lane_phase) > 0)
    )

    lane_value = float(rng.uniform(0.82, 1.0))

    paint(lane_mask, [lane_value] * 3, 1)

    # -------------------------------------------------------------
    # Vehicle geometry
    # -------------------------------------------------------------

    vehicle_present = bool(rng.random() < 0.85)
    pedestrian_present = bool(rng.random() < 0.85)

    vehicle_x = float(rng.uniform(0.27, 0.76))
    vehicle_y = float(rng.uniform(0.60, 0.83))

    vehicle_width = float(rng.uniform(0.10, 0.19))
    vehicle_height = float(rng.uniform(0.09, 0.17))

    if vehicle_present:

        vehicle_mask = (
            ((xx - vehicle_x).abs() < vehicle_width / 2)
            &
            ((yy - vehicle_y).abs() < vehicle_height / 2)
        )

        vehicle_color = rng.uniform(
            0.06, 0.20, size=3
        ).tolist()

        paint(vehicle_mask, vehicle_color, 2)

        # Appearance variation within the same semantic object.
        window_mask = vehicle_mask & (
            yy < vehicle_y - vehicle_height * 0.12
        ) & (
            (xx - vehicle_x).abs() < vehicle_width * 0.36
        )

        window_color = rng.uniform(
            0.20, 0.35, size=3
        ).tolist()

        paint(window_mask, window_color, 2)

    # -------------------------------------------------------------
    # Pedestrian geometry
    # -------------------------------------------------------------

    pedestrian_x = float(rng.uniform(0.23, 0.80))
    pedestrian_y = float(rng.uniform(0.62, 0.81))
    scale = float(rng.uniform(0.75, 1.25))

    if pedestrian_present:

        px = pedestrian_x
        py = pedestrian_y

        head = (
            (xx - px) ** 2
            + (yy - (py - 0.055 * scale)) ** 2
        ) < (0.018 * scale) ** 2

        body = (
            ((xx - px).abs() < 0.018 * scale)
            & (yy > py - 0.04 * scale)
            & (yy < py + 0.055 * scale)
        )

        legs = (
            (yy >= py + 0.04 * scale)
            & (yy <= py + 0.12 * scale)
            & (
                (
                    (xx - (px - 0.014 * scale)).abs()
                    < 0.008 * scale
                )
                |
                (
                    (xx - (px + 0.014 * scale)).abs()
                    < 0.008 * scale
                )
            )
        )

        pedestrian_mask = head | body | legs

        pedestrian_color = rng.uniform(
            0.025, 0.085, size=3
        ).tolist()

        # Visible-surface semantics:
        # pedestrian pixels overwrite occluded vehicle pixels.
        paint(pedestrian_mask, pedestrian_color, 3)

    # -------------------------------------------------------------
    # Benign illumination variation
    # -------------------------------------------------------------

    illumination = float(rng.uniform(0.22, 1.25))
    ambient = float(rng.uniform(0.002, 0.025))

    scene = (
        scene * illumination + ambient
    ).clamp(min=0.0)

    metadata = {
        "scene_seed": int(seed),
        "horizon": horizon,
        "road_center": road_center,
        "illumination": illumination,
        "ambient": ambient,
        "vehicle_present": vehicle_present,
        "pedestrian_present": pedestrian_present,
        "vehicle_visible": bool((labels == 2).any()),
        "pedestrian_visible": bool((labels == 3).any()),
    }

    return scene, labels, metadata


class ProceduralPerceptionDataset(Dataset):

    def __init__(self, split="train"):

        if split not in SPLITS:
            raise ValueError("Unknown dataset split.")

        self.split = split
        self.start_seed, self.count = SPLITS[split]

    def __len__(self):
        return self.count

    def scene_seed(self, index):

        if not 0 <= index < self.count:
            raise IndexError(index)

        return self.start_seed + index

    def optical_sample(self, index):

        return render_scene(self.scene_seed(index))

    def __getitem__(self, index):

        scene, labels, metadata = self.optical_sample(index)

        seed = self.scene_seed(index)

        # Reproducible acquisition augmentation.
        # Independent from the scene geometry generator.
        rng = np.random.default_rng(seed + 1000000)

        exposure = float(
            np.exp(rng.uniform(np.log(1.0), np.log(30.0)))
        )

        gain = float(rng.uniform(1.0, 3.0))

        noise_enabled = bool(rng.random() < 0.5)

        config = SensorConfig(
            exposure_ms=exposure,
            gain=gain,
        )

        observed = simulate_sensor(
            scene,
            config=config,
            add_noise=noise_enabled,
            seed=seed + 2000000,
        )

        image = observed["normalized_output"]

        # Train on both full and ROI-priority observations.
        # No optical attacks are included in this dataset.
        roi_enabled = bool(rng.random() < 0.5)

        if roi_enabled:

            roi_config = AdaptiveROIConfig()

            result = allocate_readout(
                image,
                roi_config.resolve(
                    image.shape[1],
                    image.shape[2],
                ),
                background_stride=roi_config.background_stride,
            )

            readout = result["readout"]
            mask = result["sampling_mask"]

        else:

            readout = image.clone()

            mask = torch.ones(
                image.shape[1:],
                dtype=torch.bool,
            )

        model_input = torch.cat(
            [
                readout,
                mask.unsqueeze(0).float(),
            ],
            dim=0,
        )

        return {
            "input": model_input,
            "target": labels,
            "scene_seed": seed,
            "sampling_ratio": float(mask.float().mean()),
            "exposure_ms": exposure,
            "gain": gain,
            "noise_enabled": noise_enabled,
            "roi_enabled": roi_enabled,
        }
