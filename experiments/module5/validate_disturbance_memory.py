# AHDS standalone reproduction experiment
# Generated during M5-DEF-02C-02B-03
# Original tasks: M5-DEF-02A-03C
# Original source cells are preserved in original_notebook_cells/.
# Historical output directories must not be used.
# Portability adaptations are documented in standalone_conversion_report.json.

if __name__ != '__main__':
    raise RuntimeError('Execute this experiment as a script.')


# ============================================================
# AHDS — M5-DEF-02A-03C
# Persistent Disturbance Evidence Validation
# ============================================================

from pathlib import Path
from dataclasses import replace, asdict
import importlib
import json
import sys
import os
import torch
import pandas as pd
def display(*objects):
    for obj in objects:
        if hasattr(obj, 'to_string'):
            print(obj.to_string())
        else:
            print(obj)

ROOT = Path(__file__).resolve().parents[2]

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

from src.ahds.module5_defense.exposure_aware_observation import (
    ExposureAwareObservationEstimator
)

import src.ahds.module5_defense.disturbance_memory as memory_module

importlib.reload(memory_module)

ExposureAwareDisturbanceMemory = (
    memory_module.ExposureAwareDisturbanceMemory
)

print("=" * 70)
print("AHDS — M5-DEF-02A-03C")
print("Persistent Disturbance Evidence Validation")
print("=" * 70)

torch.manual_seed(42)

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

attacked_scene, _ = apply_optical_attack(
    scene,
    attack_config,
    time_step=5
)

base_config = SensorConfig(
    exposure_ms=10.0,
    gain=1.0
)

estimator = ExposureAwareObservationEstimator()

memory = ExposureAwareDisturbanceMemory()

def capture(optical_scene, exposure=10.0, gain=1.0):

    config = replace(
        base_config,
        exposure_ms=exposure,
        gain=gain
    )

    result = simulate_sensor(
        optical_scene,
        config,
        add_noise=False
    )

    permitted = {
        key: result[key]
        for key in (
            "normalized_output",
            "saturated",
            "adc_clipped"
        )
    }

    observation = estimator.estimate(
        permitted,
        config
    )

    return observation


reference = capture(scene)

measurements = []

def evaluate(name, observation, expected):

    evidence = memory.assess(observation)

    measurements.append({

        "scenario": name,

        "exposure_ms":
            observation["exposure_ms"],

        "gain":
            observation["gain"],

        "expected_status": expected,

        "observed_status":
            evidence["status"],

        "global_scale":
            evidence["global_scale"],

        "common_valid_pct":
            100 * evidence["common_valid_fraction"],

        "changed_samples_pct":
            (
                None
                if evidence["changed_fraction"] is None
                else 100 * evidence["changed_fraction"]
            ),

        "mean_positive_residual":
            evidence["mean_positive_residual"],

        "censored_samples_pct":
            100 * evidence["censored_fraction"],

        "reason":
            evidence["reason"],

        "passed":
            evidence["status"] == expected
    })

    return evidence


# ------------------------------------------------------------
# 1. Missing reference
# ------------------------------------------------------------

evaluate(
    "uninitialized",
    reference,
    "UNKNOWN"
)

# Controlled experiment: the initial clean observation
# serves as a trusted startup reference.

memory.initialize_reference(reference)

reference_snapshot = memory.reference.clone()

mask_snapshot = memory.reference_mask.clone()

# ------------------------------------------------------------
# 2. Benign observations
# ------------------------------------------------------------

evaluate(
    "clean_reference",
    reference,
    "CLEAR"
)

evaluate(
    "clean_reduced_exposure",
    capture(scene, exposure=5.0),
    "CLEAR"
)

evaluate(
    "clean_increased_gain",
    capture(scene, gain=2.0),
    "CLEAR"
)

evaluate(
    "uniform_brightening",
    capture(scene * 1.3),
    "CLEAR"
)

evaluate(
    "uniform_dimming",
    capture(scene * 0.7),
    "CLEAR"
)

# ------------------------------------------------------------
# 3. Persistent optical disturbance
# ------------------------------------------------------------

high_evidence = evaluate(
    "disturbance_10ms",
    capture(attacked_scene),
    "PRESENT"
)

low_evidence = evaluate(
    "persistent_disturbance_5ms",
    capture(attacked_scene, exposure=5.0),
    "PRESENT"
)

evaluate(
    "repeated_disturbance_5ms",
    capture(attacked_scene, exposure=5.0),
    "PRESENT"
)

# ------------------------------------------------------------
# 4. Environmental recovery
# ------------------------------------------------------------

evaluate(
    "disturbance_removed_5ms",
    capture(scene, exposure=5.0),
    "CLEAR"
)

# ------------------------------------------------------------
# 5. Insufficient optical information
# ------------------------------------------------------------

evaluate(
    "extremely_low_signal",
    capture(scene, exposure=0.1),
    "UNKNOWN"
)

# ------------------------------------------------------------
# 6. Reference preservation
# ------------------------------------------------------------

reference_unchanged = torch.equal(
    memory.reference,
    reference_snapshot
)

mask_unchanged = torch.equal(
    memory.reference_mask,
    mask_snapshot
)

duplicate_reference_rejected = False

try:

    memory.initialize_reference(reference)

except RuntimeError:

    duplicate_reference_rejected = True

# ------------------------------------------------------------
# 7. Reset behavior
# ------------------------------------------------------------

memory.reset()

evaluate(
    "after_reference_reset",
    reference,
    "UNKNOWN"
)

# ------------------------------------------------------------
# 8. Numerical results
# ------------------------------------------------------------

memory_results = pd.DataFrame(measurements)

print("\n[DISTURBANCE MEMORY RESULTS]")

display(memory_results.round(6))

memory_checks = {

    "all_twelve_scenarios_completed":
        len(memory_results) == 12,

    "all_expected_statuses_matched":
        bool(memory_results["passed"].all()),

    "reference_preserved":
        reference_unchanged,

    "reference_mask_preserved":
        mask_unchanged,

    "duplicate_initialization_rejected":
        duplicate_reference_rejected,

    "disturbance_detected_before_reconfiguration":
        high_evidence["status"] == "PRESENT",

    "disturbance_detected_after_reconfiguration":
        low_evidence["status"] == "PRESENT",

    "censoring_removed_after_reconfiguration":
        low_evidence["censored_fraction"] == 0.0,

    "reference_reset_successful":
        memory.reference is None
}

print("\n[VALIDATION CHECKS]")

for name, passed in memory_checks.items():

    print(
        f"[{'PASS' if passed else 'FAIL'}] {name}"
    )

# ------------------------------------------------------------
# 9. Archive results, including any failures
# ------------------------------------------------------------

OUT = ROOT / "results/module5/dadrc_reproduction/disturbance_memory"

OUT.mkdir(parents=True, exist_ok=True)

memory_results.to_csv(
    OUT / "disturbance_memory_validation.csv",
    index=False
)

report = {

    "task": "M5-DEF-02A-03C",

    "scenarios": len(memory_results),

    "passed_scenarios": int(
        memory_results["passed"].sum()
    ),

    "checks": {
        k: bool(v)
        for k, v in memory_checks.items()
    },

    "all_checks_passed": all(
        memory_checks.values()
    ),

    "memory_config": asdict(memory.config),

    "estimator_config": asdict(estimator.config),

    "sensor_config": asdict(base_config),

    "attack_config": asdict(attack_config),

    "noise_enabled": False,

    "random_seed": 42,

    "reference_policy": "frozen_trusted_startup",

    "limitations": [
        "Static scene.",
        "Trusted clean startup reference.",
        "Noise-free observations.",
        "Single localized laser disturbance.",
        "Heuristic thresholds."
    ]
}

(OUT / "validation_report.json").write_text(
    json.dumps(report, indent=2)
)

print("\n" + "=" * 70)
print("DISTURBANCE MEMORY VALIDATION SUMMARY")
print("=" * 70)

print(
    "Scenarios passed:",
    f"{report['passed_scenarios']}/12"
)

print(
    "Validation checks:",
    f"{sum(bool(v) for v in memory_checks.values())}"
    f"/{len(memory_checks)}"
)

print(
    "Disturbance evidence at 10 ms:",
    high_evidence["status"]
)

print(
    "Disturbance evidence at 5 ms:",
    low_evidence["status"]
)

print(
    "Censored samples at 5 ms:",
    low_evidence["censored_fraction"]
)

print("Results saved:", OUT)

assert all(memory_checks.values()), (
    "Disturbance memory validation failed. "
    "Inspect saved results before proceeding."
)

print("\nM5-DEF-02A-03C: PASSED")