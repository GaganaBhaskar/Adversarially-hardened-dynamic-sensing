
# ============================================================
# AHDS — M5-DEF-02A-02B
# Exposure-Aware Physical Observation Validation
# ============================================================

from pathlib import Path
from dataclasses import replace
import importlib
import sys
import os
import torch
import pandas as pd
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

import src.ahds.module5_defense.exposure_aware_observation as obs_module

importlib.reload(obs_module)

ExposureAwareObservationEstimator = (
    obs_module.ExposureAwareObservationEstimator
)

print("=" * 70)
print("AHDS — M5-DEF-02A-02B")
print("Exposure-Aware Physical Observation Validation")
print("=" * 70)

torch.manual_seed(42)

# ------------------------------------------------------------
# 1. Prepare controlled optical environments
# ------------------------------------------------------------

scene = generate_clean_scene(
    256,
    256,
    device="cpu"
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

scenarios = {

    "clean_10ms": (
        scene,
        base_config
    ),

    "clean_5ms": (
        scene,
        replace(
            base_config,
            exposure_ms=5.0
        )
    ),

    "clean_gain2": (
        scene,
        replace(
            base_config,
            gain=2.0
        )
    ),

    "disturbed_10ms": (
        attacked_scene,
        base_config
    ),

    "disturbed_5ms": (
        attacked_scene,
        replace(
            base_config,
            exposure_ms=5.0
        )
    )
}

# ------------------------------------------------------------
# 2. Estimate optical intensity
# ------------------------------------------------------------

observations = {}
measurements = []

for name, (optical_scene, config) in scenarios.items():

    sensor_result = simulate_sensor(
        optical_scene,
        config,
        add_noise=False
    )

    # Explicitly restrict the estimator to permitted
    # sensor observations. No optical ground truth,
    # expected electrons or raw pre-clipping electrons.

    observable_result = {

        key: sensor_result[key]

        for key in (
            "normalized_output",
            "saturated",
            "adc_clipped"
        )
    }

    observation = estimator.estimate(
        observable_result,
        config
    )

    observations[name] = observation

    valid = observation["valid_mask"]

    assert valid.any(), (
        f"No valid observations: {name}"
    )

    # Ground truth is used ONLY for evaluation.

    error = (
        observation["optical_estimate"][valid]
        - optical_scene[valid]
    ).abs()

    measurements.append({

        "scenario": name,

        "exposure_ms":
            config.exposure_ms,

        "gain":
            config.gain,

        "valid_samples_pct":
            100 * observation["valid_fraction"],

        "clipped_samples_pct":
            100 * observation["clipped_fraction"],

        "censored_samples_pct":
            100 * observation["censored_fraction"],

        "low_signal_pct":
            100 * observation["low_signal_fraction"],

        "mean_absolute_error":
            error.mean().item(),

        "maximum_absolute_error":
            error.max().item(),

        "quantization_bound":
            observation["quantization_error_bound"]
    })

results = pd.DataFrame(measurements)

print("\n[PHYSICAL RECONSTRUCTION RESULTS]")

display(results.round(8))

# ------------------------------------------------------------
# 3. Compare equivalent optical observations
# ------------------------------------------------------------

def compare_observations(first, second):

    a = observations[first]
    b = observations[second]

    common_valid = (
        a["valid_mask"]
        & b["valid_mask"]
    )

    assert common_valid.any()

    difference = (
        a["optical_estimate"][common_valid]
        - b["optical_estimate"][common_valid]
    ).abs()

    bound = (
        a["quantization_error_bound"]
        + b["quantization_error_bound"]
    )

    return {

        "comparison":
            f"{first} vs {second}",

        "common_valid_pct":
            common_valid.float().mean().item() * 100,

        "mean_difference":
            difference.mean().item(),

        "maximum_difference":
            difference.max().item(),

        "combined_quantization_bound":
            bound,

        "passed":
            difference.max().item() <= bound + 1e-5
    }

comparisons = [

    compare_observations(
        "clean_10ms",
        "clean_5ms"
    ),

    compare_observations(
        "clean_10ms",
        "clean_gain2"
    ),

    compare_observations(
        "disturbed_10ms",
        "disturbed_5ms"
    )
]

comparison_df = pd.DataFrame(comparisons)

print("\n[EXPOSURE AND GAIN INVARIANCE]")

display(comparison_df.round(8))

# ------------------------------------------------------------
# 4. Validate recovery of previously clipped samples
# ------------------------------------------------------------

high_exposure = observations["disturbed_10ms"]
low_exposure = observations["disturbed_5ms"]

newly_observable = (
    high_exposure["censored_mask"]
    & low_exposure["valid_mask"]
)

newly_observable_count = (
    newly_observable.sum().item()
)

print("\n[RECOVERED OPTICAL OBSERVATIONS]")

print(
    "Previously censored samples now observable:",
    newly_observable_count
)

print(
    "Censored samples at 10 ms:",
    high_exposure["censored_mask"].sum().item()
)

print(
    "Censored samples at 5 ms:",
    low_exposure["censored_mask"].sum().item()
)

# ------------------------------------------------------------
# 5. Formal validation checks
# ------------------------------------------------------------

checks = {

    "five_scenarios_completed":
        len(observations) == 5,

    "clean_exposure_invariance":
        comparisons[0]["passed"],

    "clean_gain_invariance":
        comparisons[1]["passed"],

    "disturbed_exposure_invariance":
        comparisons[2]["passed"],

    "all_errors_within_quantization_bounds":
        bool((
            results["maximum_absolute_error"]
            <= results["quantization_bound"] + 1e-5
        ).all()),

    "clipped_samples_excluded":
        all(
            not (
                obs["valid_mask"]
                & obs["physical_clipping_mask"]
            ).any().item()
            for obs in observations.values()
        ),

    "censored_samples_excluded":
        all(
            not (
                obs["valid_mask"]
                & obs["censored_mask"]
            ).any().item()
            for obs in observations.values()
        ),

    "low_signal_samples_excluded":
        all(
            not (
                obs["valid_mask"]
                & obs["low_signal_mask"]
            ).any().item()
            for obs in observations.values()
        ),

    "invalid_samples_zero_filled":
        all(
            (
                obs["optical_estimate"][
                    ~obs["valid_mask"]
                ] == 0
            ).all().item()
            for obs in observations.values()
        ),

    "lower_exposure_reduces_censoring":
        low_exposure["censored_fraction"]
        < high_exposure["censored_fraction"],

    "previously_censored_samples_recovered":
        newly_observable_count > 0,

    "shorter_exposure_increases_uncertainty":
        observations["clean_5ms"][
            "quantization_error_bound"
        ]
        >
        observations["clean_10ms"][
            "quantization_error_bound"
        ],

    "finite_reconstructions":
        all(
            torch.isfinite(
                obs["optical_estimate"]
            ).all().item()
            for obs in observations.values()
        )
}

print("\n[VALIDATION CHECKS]")

for name, passed in checks.items():

    status = "PASS" if passed else "FAIL"

    print(f"[{status}] {name}")

passed_count = sum(
    bool(value) for value in checks.values()
)

print("\n" + "=" * 70)
print("PHYSICAL OBSERVATION VALIDATION SUMMARY")
print("=" * 70)

print(
    f"Validation checks: "
    f"{passed_count}/{len(checks)}"
)

assert all(checks.values()), (
    "Physical observation validation failed."
)

print("Exposure invariance: VERIFIED")
print("Gain invariance: VERIFIED")
print("Clipping exclusion: VERIFIED")
print("Quantization bounds: VERIFIED")
print("Previously clipped observations recovered: VERIFIED")

print("\nM5-DEF-02A-02B: PASSED")