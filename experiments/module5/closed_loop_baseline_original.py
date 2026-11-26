# ============================================================
# AHDS — MODULE 5
# EXPERIMENT: CLOSED-LOOP DEFENSE COMPARISON
#
# Strategies:
# 1. Fixed exposure
# 2. Original exposure protection
# 3. State-aware exposure protection
# 4. Probe-based exposure protection
#
# Frames 0–4:   Clean baseline
# Frames 5–19:  Persistent laser disturbance
# Frames 20–29: Recovery
#
# Noise-free, fixed gain, 256x256 scene
# ============================================================

import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import replace, asdict
from IPython.display import display as show_table

from src.ahds.module1_environment.optical_environment import (
    AttackConfig,
    generate_clean_scene,
    apply_optical_attack,
)

from src.ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor,
)

from src.ahds.module4_anomaly_detection.disturbance_assessor import (
    PhysicalDisturbanceAssessor,
)

from src.ahds.module4_anomaly_detection.disturbance_state_tracker import (
    DisturbanceStateTracker,
)

from src.ahds.module5_defense.exposure_protection import (
    ExposureProtectionController,
)

from src.ahds.module5_defense.state_aware_exposure import (
    StateAwareExposureController,
)

from src.ahds.module5_defense.probed_exposure import (
    ProbedExposureController,
)


# ============================================================
# 1. REPOSITORY AND RESULTS DIRECTORIES
# ============================================================

ROOT = Path(
    "/content/adversarially-hardened-dynamic-sensing"
)

assert ROOT.is_dir(), "AHDS repository not found."

OUT5 = ROOT / "results/module5"
FIG5 = OUT5 / "figures"

FIG5.mkdir(parents=True, exist_ok=True)

torch.manual_seed(42)


# ============================================================
# 2. GENERATE CLEAN OPTICAL SCENE
# ============================================================

HEIGHT = 256
WIDTH = 256

scene = generate_clean_scene(
    HEIGHT,
    WIDTH,
    device="cpu",
)

assert scene.shape == (3, HEIGHT, WIDTH)
assert torch.isfinite(scene).all()
assert (scene >= 0).all()

print("Clean scene:", tuple(scene.shape))


# ============================================================
# 3. CONFIGURE LASER DISTURBANCE
# ============================================================

attack_config = AttackConfig(
    attack_type="laser",
    center_x=0.5,
    center_y=0.5,
    radius=0.12,
    intensity=15.0,
)

base_sensor_config = SensorConfig(
    exposure_ms=10.0,
    gain=1.0,
)

INITIAL_EXPOSURE = 10.0
TOTAL_FRAMES = 30
ATTACK_START = 5
ATTACK_END = 20


# ============================================================
# 4. DEFINE BACKGROUND EVALUATION REGION
# ============================================================

# The laser may have nonzero spatial tails.
# Therefore, avoid requiring attack_map == 0.
#
# Instead, exclude the central laser region using
# a geometric mask with a safety margin.
#
# This is a background-region metric, not a guarantee
# that every included pixel is optically unaffected.

_, height, width = scene.shape

yy, xx = torch.meshgrid(
    torch.arange(height, device=scene.device),
    torch.arange(width, device=scene.device),
    indexing="ij",
)

x_normalized = xx.float() / (width - 1)
y_normalized = yy.float() / (height - 1)

distance = torch.sqrt(
    (x_normalized - attack_config.center_x) ** 2
    + (y_normalized - attack_config.center_y) ** 2
)

background_exclusion_radius = max(
    2.0 * attack_config.radius,
    0.20,
)

background_mask = (
    distance >= background_exclusion_radius
)

assert background_mask.any().item(), (
    "Background evaluation region is empty."
)

background_pixels = int(
    background_mask.sum().item()
)

background_fraction = float(
    background_mask.float().mean().item()
)

print("Background pixels:", background_pixels)
print("Background fraction:", round(background_fraction, 4))


# ============================================================
# 5. GENERATE CLEAN REFERENCE SENSOR OUTPUT
# ============================================================

clean_reference_sensor = simulate_sensor(
    scene,
    base_sensor_config,
    add_noise=False,
)

clean_reference_output = (
    clean_reference_sensor["normalized_output"]
)

reference_background_brightness = float(
    clean_reference_output[:, background_mask]
    .mean()
    .item()
)

assert reference_background_brightness > 0, (
    "Reference background brightness must be positive."
)

print(
    "Reference background brightness:",
    round(reference_background_brightness, 6),
)


# ============================================================
# 6. INITIALIZE FOUR STRATEGIES
# ============================================================

strategies = {
    "fixed": None,
    "original": ExposureProtectionController(),
    "state_aware": StateAwareExposureController(),
    "probed": ProbedExposureController(),
}

records = []


# ============================================================
# 7. EXECUTE CLOSED-LOOP EXPERIMENT
# ============================================================

for strategy_name, controller in strategies.items():

    print(f"\nRunning strategy: {strategy_name}")

    # Independent Module 4 instances for each strategy.
    assessor = PhysicalDisturbanceAssessor()
    tracker = DisturbanceStateTracker()

    assessor.reset()
    tracker.reset()

    if controller is not None:
        controller.reset()

    exposure = INITIAL_EXPOSURE

    for frame in range(TOTAL_FRAMES):

        # ----------------------------------------------------
        # A. Generate optical environment
        # ----------------------------------------------------

        attack_active = (
            ATTACK_START <= frame < ATTACK_END
        )

        if attack_active:

            optical_scene, _ = apply_optical_attack(
                scene,
                attack_config,
                time_step=frame,
            )

        else:
            optical_scene = scene.clone()

        # ----------------------------------------------------
        # B. Simulate sensor using current exposure
        # ----------------------------------------------------

        current_exposure = float(exposure)

        sensor_config = replace(
            base_sensor_config,
            exposure_ms=current_exposure,
        )

        sensor = simulate_sensor(
            optical_scene,
            sensor_config,
            add_noise=False,
        )

        # ----------------------------------------------------
        # C. Module 4 disturbance assessment
        # ----------------------------------------------------

        assessment = assessor.detect(sensor)

        state_result = tracker.update(assessment)

        # ----------------------------------------------------
        # D. Module 5 defensive controller
        # ----------------------------------------------------

        if strategy_name == "fixed":

            decision = {
                "exposure_ms": current_exposure,
                "action": "FIXED",
            }

        elif strategy_name == "state_aware":

            decision = controller.update(
                current_exposure,
                assessment,
                state_result["state"],
            )

        else:

            decision = controller.update(
                current_exposure,
                assessment,
            )

        next_exposure = float(
            decision["exposure_ms"]
        )

        assert np.isfinite(next_exposure), (
            "Controller produced non-finite exposure."
        )

        assert 0.1 <= next_exposure <= 30.0, (
            "Controller exposure outside expected limits."
        )

        # ----------------------------------------------------
        # E. Measure background brightness preservation
        # ----------------------------------------------------

        sensor_output = sensor["normalized_output"]

        background_brightness = float(
            sensor_output[:, background_mask]
            .mean()
            .item()
        )

        brightness_retention = (
            background_brightness
            / reference_background_brightness
        )

        # ----------------------------------------------------
        # F. Record numerical results
        # ----------------------------------------------------

        records.append({

            "strategy": strategy_name,

            "frame": frame,

            "attack_active": int(attack_active),

            "exposure_ms": current_exposure,

            "next_exposure_ms": next_exposure,

            "action": decision["action"],

            "tracker_state": state_result["state"],

            "disturbance_detected": int(
                assessment["disturbance_detected"]
            ),

            "saturation_fraction": float(
                assessment["saturation_fraction"]
            ),

            "full_well_saturation_ratio": float(
                sensor["saturation_ratio"]
            ),

            "adc_clip_ratio": float(
                sensor["adc_clip_ratio"]
            ),

            "background_brightness": background_brightness,

            "brightness_retention": brightness_retention,

            "protection_active": (
                decision.get("protection_active", None)
            ),
        })

        # Exposure command applies to the NEXT frame.
        exposure = next_exposure


# ============================================================
# 8. CREATE RESULTS DATAFRAME
# ============================================================

closed_loop_df = pd.DataFrame(records)

assert len(closed_loop_df) == 4 * TOTAL_FRAMES

assert closed_loop_df.groupby("strategy").size().eq(
    TOTAL_FRAMES
).all()

assert np.isfinite(
    closed_loop_df[
        [
            "exposure_ms",
            "next_exposure_ms",
            "saturation_fraction",
            "full_well_saturation_ratio",
            "adc_clip_ratio",
            "background_brightness",
            "brightness_retention",
        ]
    ].to_numpy(dtype=float)
).all()

csv_path = OUT5 / "closed_loop_comparison.csv"

closed_loop_df.to_csv(
    csv_path,
    index=False,
)

print("\nFull experiment saved:", csv_path)


# ============================================================
# 9. GENERATE STRATEGY COMPARISON SUMMARY
# ============================================================

summary_records = []

for strategy_name, group in closed_loop_df.groupby(
    "strategy"
):

    attack_rows = group[
        group["attack_active"] == 1
    ]

    recovery_rows = group[
        group["frame"] >= ATTACK_END
    ]

    summary_records.append({

        "strategy": strategy_name,

        "mean_attack_saturation": float(
            attack_rows[
                "full_well_saturation_ratio"
            ].mean()
        ),

        "mean_attack_adc_clipping": float(
            attack_rows[
                "adc_clip_ratio"
            ].mean()
        ),

        "mean_attack_brightness_retention": float(
            attack_rows[
                "brightness_retention"
            ].mean()
        ),

        "minimum_exposure_ms": float(
            group["exposure_ms"].min()
        ),

        "maximum_exposure_ms": float(
            group["exposure_ms"].max()
        ),

        "final_exposure_ms": float(
            group.iloc[-1]["next_exposure_ms"]
        ),

        "recovery_mean_brightness_retention": float(
            recovery_rows[
                "brightness_retention"
            ].mean()
        ),

        "protect_actions": int(
            (group["action"] == "PROTECT").sum()
        ),

        "probe_actions": int(
            (group["action"] == "PROBE").sum()
        ),

        "failed_probe_actions": int(
            (group["action"] == "PROBE_FAILED").sum()
        ),
    })

summary_df = pd.DataFrame(summary_records)

summary_path = OUT5 / "closed_loop_summary.csv"

summary_df.to_csv(
    summary_path,
    index=False,
)

print("\n" + "=" * 65)
print("MODULE 5 — CLOSED-LOOP COMPARISON")
print("=" * 65)

show_table(summary_df.round(6))


# ============================================================
# 10. SAVE EXPERIMENT CONFIGURATION
# ============================================================

experiment_config = {

    "experiment": "module5_initial_closed_loop_comparison",

    "random_seed": 42,

    "image_size": [HEIGHT, WIDTH],

    "total_frames": TOTAL_FRAMES,

    "attack_start_frame": ATTACK_START,

    "attack_end_frame_exclusive": ATTACK_END,

    "attack_config": asdict(attack_config),

    "sensor_config": asdict(base_sensor_config),

    "noise_enabled": False,

    "strategies": list(strategies.keys()),

    "background_exclusion_radius_normalized":
        background_exclusion_radius,

    "background_pixel_count": background_pixels,

    "background_fraction": background_fraction,

    "reference_background_brightness":
        reference_background_brightness,

    "exposure_command_delay_frames": 1,

    "notes": [
        "Fixed gain across all strategies.",
        "Noise-free initial comparison.",
        "Background mask excludes nominal laser center.",
        "Background region may contain weak laser tails.",
        "Brightness retention is relative to clean 10 ms capture.",
        "Controller decisions affect the following frame.",
        "No hardware bandwidth or energy measurements.",
        "Module 3 adaptive gain and ROI are not integrated yet."
    ]
}

config_path = OUT5 / "closed_loop_experiment_config.json"

with open(config_path, "w") as f:
    json.dump(
        experiment_config,
        f,
        indent=2,
    )

print("\nConfiguration saved:", config_path)


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("EXPERIMENT COMPLETED")
print("=" * 65)

print("Strategies:", len(strategies))
print("Frames per strategy:", TOTAL_FRAMES)
print("Total measurements:", len(closed_loop_df))
print("Results directory:", OUT5)
