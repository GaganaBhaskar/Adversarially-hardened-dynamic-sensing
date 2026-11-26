
# ============================================================
# AHDS — M5-DEF-02A-04B
# DADRC Initial Closed-Loop Smoke Test
# ============================================================

from pathlib import Path
from dataclasses import replace, asdict
import importlib
import json
import sys
import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

ROOT = Path(
    "/content/adversarially-hardened-dynamic-sensing"
)

os.chdir(ROOT)

for path in (ROOT, ROOT / "src"):

    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.ahds.module1_environment.optical_environment import (
    generate_clean_scene,
    AttackConfig,
    apply_optical_attack
)

from src.ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor
)

from src.ahds.module4_anomaly_detection.disturbance_assessor import (
    PhysicalDisturbanceAssessor
)

from src.ahds.module4_anomaly_detection.disturbance_state_tracker import (
    DisturbanceStateTracker
)

import src.ahds.module5_defense.disturbance_aware_defense as defense_module

importlib.reload(defense_module)

DisturbanceAwareDefenseController = (
    defense_module.DisturbanceAwareDefenseController
)

print("=" * 70)
print("AHDS — M5-DEF-02A-04B")
print("DADRC Initial Closed-Loop Smoke Test")
print("=" * 70)

torch.manual_seed(42)

TOTAL_FRAMES = 60

ATTACK_START = 10

ATTACK_END = 30

INITIAL_EXPOSURE = 10.0

scene = generate_clean_scene(
    256, 256, device="cpu"
)

attack_config = AttackConfig(
    attack_type="laser",
    center_x=0.5,
    center_y=0.5,
    radius=0.12,
    intensity=15.0
)

base_config = SensorConfig(
    exposure_ms=INITIAL_EXPOSURE,
    gain=1.0
)

controller = DisturbanceAwareDefenseController()

assessor = PhysicalDisturbanceAssessor()

tracker = DisturbanceStateTracker()

# ------------------------------------------------------------
# 1. Trusted startup reference
# ------------------------------------------------------------

reference_result = simulate_sensor(
    scene,
    base_config,
    add_noise=False
)

controller.initialize_reference(
    reference_result,
    base_config
)

reference_snapshot = (
    controller.memory.reference.clone()
)

reference_mask_snapshot = (
    controller.memory.reference_mask.clone()
)

print("\nTrusted optical reference: INITIALIZED")

# ------------------------------------------------------------
# 2. Closed-loop execution
# ------------------------------------------------------------

exposure = INITIAL_EXPOSURE

records = []

for frame in range(TOTAL_FRAMES):

    attack_active = (
        ATTACK_START <= frame < ATTACK_END
    )

    if attack_active:

        optical_scene, _ = apply_optical_attack(
            scene,
            attack_config,
            time_step=frame
        )

    else:

        optical_scene = scene

    sensor_config = replace(
        base_config,
        exposure_ms=exposure
    )

    sensor_result = simulate_sensor(
        optical_scene,
        sensor_config,
        add_noise=False
    )

    assessment = assessor.detect(
        sensor_result
    )

    tracker_result = tracker.update(
        assessment
    )

    # The controller receives only permitted
    # sensor observations and acquisition metadata.

    permitted_result = {

        key: sensor_result[key]

        for key in (
            "normalized_output",
            "saturated",
            "adc_clipped"
        )
    }

    decision = controller.update(
        permitted_result,
        sensor_config,
        assessment,
        tracker_result["state"]
    )

    records.append({

        "frame": frame,

        # Ground truth is logged for evaluation only.
        "attack_active": int(attack_active),

        "exposure_ms": exposure,

        "next_exposure_ms":
            decision["exposure_ms"],

        "action":
            decision["action"],

        "controller_state":
            decision["controller_state"],

        "tracker_state":
            decision["tracker_state"],

        "evidence_status":
            decision["evidence_status"],

        "evidence_reason":
            decision["evidence_reason"],

        "saturation_ratio":
            sensor_result["saturation_ratio"],

        "adc_clip_ratio":
            sensor_result["adc_clip_ratio"],

        "physical_clipping_fraction":
            decision["physical_clipping_fraction"],

        "censored_fraction":
            decision["censored_fraction"],

        "spatial_saturation_fraction":
            decision["saturation_fraction"],

        "optical_changed_fraction":
            decision["optical_changed_fraction"],

        "common_valid_fraction":
            decision["common_valid_fraction"],

        "protection_active":
            decision["protection_active"],

        "reference_exposure":
            decision["reference_exposure"],

        "probe_pending":
            decision["probe_pending"],

        "clear_frames":
            decision["clear_frames"],

        "cooldown_remaining":
            decision["cooldown_remaining"]
    })

    # Apply command to the NEXT frame.

    exposure = decision["exposure_ms"]

dadrc_results = pd.DataFrame(records)

# ------------------------------------------------------------
# 3. Display important transitions
# ------------------------------------------------------------

important_columns = [

    "frame",

    "attack_active",

    "exposure_ms",

    "next_exposure_ms",

    "action",

    "controller_state",

    "tracker_state",

    "evidence_status",

    "saturation_ratio",

    "optical_changed_fraction"
]

print("\n[CLOSED-LOOP TIMELINE]")

display(
    dadrc_results[
        important_columns
    ].round(6)
)

# ------------------------------------------------------------
# 4. Calculate preliminary metrics
# ------------------------------------------------------------

attack_rows = dadrc_results[
    dadrc_results["attack_active"] == 1
]

pre_attack_rows = dadrc_results[
    dadrc_results["frame"] < ATTACK_START
]

recovery_rows = dadrc_results[
    dadrc_results["frame"] >= ATTACK_END
]

persistent_unsaturated = attack_rows[

    (
        attack_rows["physical_clipping_fraction"]
        == 0
    )

    & (
        attack_rows["censored_fraction"]
        == 0
    )
]

restored_rows = recovery_rows[
    recovery_rows["action"] == "RESTORED"
]

restoration_frame = (

    int(restored_rows.iloc[0]["frame"])

    if not restored_rows.empty

    else None
)

summary = {

    "total_frames": len(dadrc_results),

    "attack_frames": len(attack_rows),

    "mean_attack_saturation":
        float(
            attack_rows["saturation_ratio"].mean()
        ),

    "minimum_exposure_ms":
        float(
            dadrc_results["exposure_ms"].min()
        ),

    "final_applied_exposure_ms":
        float(
            dadrc_results.iloc[-1]["exposure_ms"]
        ),

    "final_commanded_exposure_ms":
        float(
            dadrc_results.iloc[-1]["next_exposure_ms"]
        ),

    "persistent_unsaturated_frames":
        len(persistent_unsaturated),

    "persistent_unsaturated_present_frames":
        int(
            (
                persistent_unsaturated[
                    "evidence_status"
                ] == "PRESENT"
            ).sum()
        ),

    "protect_actions":
        int(
            (
                dadrc_results["action"]
                == "PROTECT"
            ).sum()
        ),

    "probe_actions":
        int(
            (
                dadrc_results["action"]
                == "PROBE"
            ).sum()
        ),

    "failed_probe_actions":
        int(
            (
                dadrc_results["action"]
                == "PROBE_FAILED"
            ).sum()
        ),

    "recovery_actions":
        int(
            (
                dadrc_results["action"]
                == "RECOVER"
            ).sum()
        ),

    "restoration_frame":
        restoration_frame,

    "restoration_delay_frames":
        (
            restoration_frame - ATTACK_END

            if restoration_frame is not None

            else None
        )
}

print("\n[PRELIMINARY CONTROLLER METRICS]")

for name, value in summary.items():

    print(f"{name}: {value}")

# ------------------------------------------------------------
# 5. Smoke-test validation
# ------------------------------------------------------------

unsafe_rows = dadrc_results[

    (
        dadrc_results[
            "physical_clipping_fraction"
        ] >= controller.config.saturation_trigger
    )

    | (
        dadrc_results[
            "censored_fraction"
        ] >= controller.config.saturation_trigger
    )

    | (
        dadrc_results[
            "spatial_saturation_fraction"
        ] >= controller.config.saturation_trigger
    )
]

unknown_protected = dadrc_results[

    (
        dadrc_results["evidence_status"]
        == "UNKNOWN"
    )

    & dadrc_results["protection_active"]
]

protected_rows = dadrc_results[
    dadrc_results["protection_active"]
]

smoke_checks = {

    "all_sixty_frames_completed":
        len(dadrc_results) == TOTAL_FRAMES,

    "exposure_commands_within_bounds":
        bool(
            dadrc_results[
                "next_exposure_ms"
            ].between(
                controller.config.min_exposure_ms,
                controller.config.max_exposure_ms
            ).all()
        ),

    "one_frame_command_delay":
        bool(
            np.allclose(
                dadrc_results[
                    "next_exposure_ms"
                ].to_numpy()[:-1],

                dadrc_results[
                    "exposure_ms"
                ].to_numpy()[1:]
            )
        ),

    "clean_startup_exposure_preserved":
        bool(
            np.allclose(
                pre_attack_rows["next_exposure_ms"],
                INITIAL_EXPOSURE
            )
        ),

    "unsafe_observation_encountered":
        len(unsafe_rows) > 0,

    "unsafe_observations_never_increase_exposure":
        bool(
            (
                unsafe_rows["next_exposure_ms"]
                <= unsafe_rows["exposure_ms"]
            ).all()
        ),

    "protective_action_observed":
        summary["protect_actions"] > 0,

    "persistent_unsaturated_evidence_observed":
        (
            summary[
                "persistent_unsaturated_present_frames"
            ] > 0
        ),

    "unknown_evidence_never_increases_exposure":
        bool(
            (
                unknown_protected["next_exposure_ms"]
                <= unknown_protected["exposure_ms"]
            ).all()
        ),

    "recovery_does_not_exceed_reference":
        bool(
            (
                protected_rows["next_exposure_ms"]
                <= protected_rows["reference_exposure"]
                + 1e-8
            ).all()
        ),

    "optical_reference_preserved":
        torch.equal(
            controller.memory.reference,
            reference_snapshot
        ),

    "optical_reference_mask_preserved":
        torch.equal(
            controller.memory.reference_mask,
            reference_mask_snapshot
        ),

    "restoration_observed":
        restoration_frame is not None,

    "final_exposure_restored":
        bool(
            np.isclose(
                dadrc_results.iloc[-1]["exposure_ms"],
                INITIAL_EXPOSURE
            )
        ),

    "final_protection_inactive":
        not controller.protection_active
}

print("\n[SMOKE TEST CHECKS]")

for name, passed in smoke_checks.items():

    print(
        f"[{'PASS' if passed else 'FAIL'}] {name}"
    )

# ------------------------------------------------------------
# 6. Archive numerical results and configuration
# ------------------------------------------------------------

OUT = ROOT / "results/module5/dadrc_smoke_test"

FIG = OUT / "figures"

FIG.mkdir(parents=True, exist_ok=True)

dadrc_results.to_csv(
    OUT / "closed_loop_timeline.csv",
    index=False
)

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2)
)

report = {

    "task": "M5-DEF-02A-04B",

    "checks": {
        k: bool(v)
        for k, v in smoke_checks.items()
    },

    "passed_checks": sum(
        bool(v)
        for v in smoke_checks.values()
    ),

    "total_checks": len(smoke_checks),

    "all_checks_passed":
        all(smoke_checks.values()),

    "controller_config":
        asdict(controller.config),

    "observation_config":
        asdict(controller.estimator.config),

    "memory_config":
        asdict(controller.memory.config),

    "sensor_config":
        asdict(base_config),

    "attack_config":
        asdict(attack_config),

    "total_frames": TOTAL_FRAMES,

    "attack_start_frame": ATTACK_START,

    "attack_end_frame_exclusive": ATTACK_END,

    "random_seed": 42,

    "noise_enabled": False,

    "image_size": [256, 256],

    "command_delay_frames": 1,

    "reference_policy":
        "frozen_trusted_startup",

    "limitations": [

        "Initial functional smoke test.",

        "Static scene and fixed gain.",

        "Noise-free simulation.",

        "Single disturbance configuration.",

        "No baseline superiority claim.",

        "No Module 3 arbitration yet.",

        "No downstream perception evaluation."
    ]
}

(OUT / "validation_report.json").write_text(
    json.dumps(report, indent=2)
)

# ------------------------------------------------------------
# 7. Plot closed-loop behavior
# ------------------------------------------------------------

fig, axes = plt.subplots(
    4, 1,
    figsize=(13, 12),
    sharex=True
)

for ax in axes:

    ax.axvspan(
        ATTACK_START,
        ATTACK_END,
        color="red",
        alpha=0.08
    )

    ax.grid(alpha=0.25)

axes[0].plot(
    dadrc_results["frame"],
    dadrc_results["exposure_ms"],
    label="Applied exposure",
    linewidth=2
)

axes[0].plot(
    dadrc_results["frame"],
    dadrc_results["next_exposure_ms"],
    linestyle="--",
    label="Next-frame command"
)

axes[0].set_ylabel("Exposure (ms)")
axes[0].legend()

axes[1].plot(
    dadrc_results["frame"],
    100 * dadrc_results["saturation_ratio"],
    label="Full-well saturation"
)

axes[1].plot(
    dadrc_results["frame"],
    100 * dadrc_results["adc_clip_ratio"],
    linestyle="--",
    label="ADC clipping"
)

axes[1].set_ylabel("Clipping (%)")
axes[1].legend()

axes[2].plot(
    dadrc_results["frame"],
    100 * dadrc_results["optical_changed_fraction"],
    label="Reference-based disturbance candidates"
)

axes[2].set_ylabel("Changed samples (%)")
axes[2].legend()

status_mapping = {

    "CLEAR": 0,

    "UNKNOWN": 1,

    "PRESENT": 2
}

axes[3].step(
    dadrc_results["frame"],
    dadrc_results["evidence_status"].map(
        status_mapping
    ),
    where="post"
)

axes[3].set_yticks(
    [0, 1, 2],
    ["CLEAR", "UNKNOWN", "PRESENT"]
)

axes[3].set_xlabel("Frame")
axes[3].set_ylabel("Memory evidence")

fig.suptitle(
    "AHDS — Initial DADRC Closed-Loop Behavior",
    fontsize=15
)

fig.tight_layout()

fig.savefig(
    FIG / "dadrc_closed_loop_timeline.png",
    dpi=180,
    bbox_inches="tight"
)

plt.show()

# ------------------------------------------------------------
# 8. Final verification
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DADRC SMOKE TEST SUMMARY")
print("=" * 70)

print(
    "Validation checks:",
    f"{report['passed_checks']}/{report['total_checks']}"
)

print("Restoration frame:", restoration_frame)

print("Results directory:", OUT)

assert all(smoke_checks.values()), (
    "Smoke test failed. "
    "Inspect archived results before proceeding."
)

print("\nM5-DEF-02A-04B: PASSED")