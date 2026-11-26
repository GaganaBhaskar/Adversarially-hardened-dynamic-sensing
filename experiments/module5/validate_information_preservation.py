
"""
AHDS — M5-VAL-01A-03
Controlled information-preservation metric validation.

Evaluation-only ground truth.
No defensive controllers are executed.
"""

from pathlib import Path
from dataclasses import replace, asdict

import sys
import math
import json
import hashlib

import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

from src.ahds.module1_environment.optical_environment import (
    AttackConfig,
    generate_clean_scene,
    apply_optical_attack
)

from src.ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor
)

from src.ahds.module7_evaluation.information_preservation import (
    InformationPreservationMetrics
)

TASK = "M5-VAL-01A-03"

OUT = (
    ROOT /
    "results/module5/"
    "information_preservation_validation"
)

torch.manual_seed(42)

evaluator = InformationPreservationMetrics()

base = SensorConfig(
    exposure_ms=10.0,
    gain=1.0
)

clean = generate_clean_scene(
    height=256,
    width=256,
    device="cpu"
)

attack_config = AttackConfig(
    attack_type="laser",
    center_x=0.5,
    center_y=0.5,
    radius=0.12,
    intensity=15.0
)

disturbed, attack_map = apply_optical_attack(
    clean,
    attack_config,
    time_step=0
)

rows = []
checks = []
bound_rows = []
scenario_results = {}


def capture(scene, config):

    return simulate_sensor(
        scene,
        config,
        add_noise=False
    )


def close(a, b, atol=1e-8, rtol=1e-6):

    return math.isclose(
        float(a),
        float(b),
        abs_tol=atol,
        rel_tol=rtol
    )


def check(name, condition, detail=None):

    passed = bool(condition)

    checks.append({
        "check": name,
        "passed": passed,
        "detail": detail
    })

    print(
        "[PASS]" if passed else "[FAIL]",
        name
    )


def run_scenario(
    scenario_id,
    name,
    reference_scene,
    observed_scene,
    applied_config,
    reference_config=None,
    region=None
):

    if reference_config is None:

        reference_config = base

    reference = capture(
        reference_scene,
        reference_config
    )

    counterfactual = capture(
        reference_scene,
        applied_config
    )

    observed = capture(
        observed_scene,
        applied_config
    )

    result = evaluator.evaluate(

        reference_result=reference,

        counterfactual_result=counterfactual,

        observed_result=observed,

        reference_config=reference_config,

        applied_config=applied_config,

        clean_optical_scene=reference_scene,

        region_mask=region
    )

    assert len(result) == 42

    json.dumps(result, allow_nan=False)

    row = {

        "scenario_id": scenario_id,

        "scenario": name,

        "exposure_ms": applied_config.exposure_ms,

        "gain": applied_config.gain,

        **result
    }

    rows.append(row)

    scenario_results[scenario_id] = result

    return result, reference, counterfactual, observed


# ------------------------------------------------------------
# 3. Independent per-sample quantization-bound verification
# ------------------------------------------------------------

def verify_quantization_bound(
    name,
    scene,
    result,
    config
):

    adc_max = (2 ** config.adc_bits) - 1

    scale = (

        config.exposure_ms
        * config.gain
        * config.electrons_per_unit_ms
        / config.full_well_electrons
    )

    reconstructed = (

        result["adc_output"].double()
        / adc_max
        / scale
    )

    valid = ~(

        result["saturated"]
        | result["adc_clipped"]
    )

    errors = (

        reconstructed
        - scene.double()

    ).abs()[valid]

    half_step = 0.5 / (adc_max * scale)

    if errors.numel() == 0:

        max_error = None
        mean_error = None
        violations = None
        passed = False

    else:

        max_error = float(errors.max())

        mean_error = float(errors.mean())

        # Allow small float32 simulation roundoff.
        tolerance = 1e-5

        violations = int(
            (errors > half_step + tolerance)
            .sum()
            .item()
        )

        passed = violations == 0

    bound_rows.append({

        "scenario": name,

        "valid_samples": int(valid.sum()),

        "maximum_absolute_error": max_error,

        "mean_absolute_error": mean_error,

        "ideal_half_step": half_step,

        "roundoff_tolerance": 1e-5,

        "bound_violations": violations,

        "passed": passed
    })

    check(
        name + "_per_sample_quantization_bound",
        passed
    )


# ------------------------------------------------------------
# 4. V01 — Clean reference
# ------------------------------------------------------------

print("\n[V01 — CLEAN REFERENCE]")

v01, r01, c01, y01 = run_scenario(
    "V01",
    "clean_reference",
    clean,
    clean,
    base
)

check(
    "V01_zero_configuration_distortion",
    v01["configuration_mae"] == 0
)

check(
    "V01_zero_disturbance_distortion",
    v01["disturbance_mae"] == 0
)

check(
    "V01_zero_reference_distortion",
    v01["reference_mae"] == 0
)

check(
    "V01_zero_gradient_distortion",
    v01["gradient_mae"] == 0
)

check(
    "V01_full_valid_coverage",
    v01["valid_observation_coverage"] == 1
)

check(
    "V01_reference_contrast_identity",
    v01["contrast_retention"] == 1
)

verify_quantization_bound(
    "V01",
    clean,
    y01,
    base
)


# ------------------------------------------------------------
# 5. V02–V04 — Exposure reduction
# ------------------------------------------------------------

print("\n[V02–V04 — EXPOSURE REDUCTION]")

for scenario_id, exposure in (

    ("V02", 5.0),

    ("V03", 1.0),

    ("V04", 0.1)
):

    config = replace(
        base,
        exposure_ms=exposure
    )

    m, r, c, y = run_scenario(

        scenario_id,

        f"clean_exposure_{exposure}",

        clean,

        clean,

        config
    )

    check(
        scenario_id + "_zero_disturbance_error",
        m["disturbance_mae"] == 0
    )

    check(
        scenario_id + "_configuration_error_positive",
        m["configuration_mae"] > 0
    )

    check(
        scenario_id + "_exposure_utilization",
        close(
            m["exposure_utilization"],
            exposure / 10.0
        )
    )

    check(
        scenario_id + "_quantization_step_scaling",
        close(
            m["relative_quantization_step"],
            10.0 / exposure
        )
    )

    check(
        scenario_id + "_no_saturation_or_clipping",
        m["censored_ratio"] == 0
    )

    verify_quantization_bound(
        scenario_id,
        clean,
        y,
        config
    )

v02 = scenario_results["V02"]
v03 = scenario_results["V03"]
v04 = scenario_results["V04"]

check(
    "exposure_reduction_increases_configuration_error",

    v01["configuration_mae"]
    < v02["configuration_mae"]
    < v03["configuration_mae"]
    < v04["configuration_mae"]
)

check(
    "clean_ADC_level_count_decreases",

    v01["clean_adc_levels"]
    > v02["clean_adc_levels"]
    > v03["clean_adc_levels"]
    > v04["clean_adc_levels"]
)

check(
    "minimum_exposure_has_zero_ADC_samples",
    v04["clean_zero_adc_ratio"] > 0
)

check(
    "minimum_exposure_nonzero_coverage_degrades",
    v04["nonzero_valid_coverage"] < 1
)


# ------------------------------------------------------------
# 6. V05 — Noise-free gain compensation
# ------------------------------------------------------------

print("\n[V05 — CLEAN GAIN COMPENSATION]")

gain_config = replace(
    base,
    exposure_ms=5.0,
    gain=2.0
)

v05, r05, c05, y05 = run_scenario(

    "V05",

    "clean_gain_compensation",

    clean,

    clean,

    gain_config
)

check(
    "V05_exact_ADC_identity",

    torch.equal(
        r05["adc_output"],
        y05["adc_output"]
    )
)

check(
    "V05_zero_configuration_error",
    v05["configuration_mae"] == 0
)

check(
    "V05_optical_scale_identity",
    v05["optical_scale_utilization"] == 1
)

check(
    "V05_quantization_step_identity",
    v05["relative_quantization_step"] == 1
)

check(
    "V05_exposure_and_gain_distinguished",

    v05["exposure_utilization"] == 0.5

    and v05["gain_utilization"] == 2.0
)

verify_quantization_bound(
    "V05",
    clean,
    y05,
    gain_config
)


# ------------------------------------------------------------
# 7. V06–V08 — Optical disturbance
# ------------------------------------------------------------

print("\n[V06–V08 — OPTICAL DISTURBANCE]")

for scenario_id, name, config in (

    ("V06", "disturbed_reference", base),

    (
        "V07",
        "disturbed_reduced_exposure",
        replace(base, exposure_ms=5.0)
    ),

    (
        "V08",
        "disturbed_gain_compensation",
        gain_config
    )
):

    m, r, c, y = run_scenario(

        scenario_id,

        name,

        clean,

        disturbed,

        config
    )

    check(
        scenario_id + "_positive_disturbance_error",
        m["disturbance_mae"] > 0
    )

    check(
        scenario_id + "_sensor_saturation_agreement",

        close(
            m["saturation_ratio"],
            y["saturation_ratio"]
        )
    )

    check(
        scenario_id + "_sensor_ADC_clipping_agreement",

        close(
            m["adc_clipping_ratio"],
            y["adc_clip_ratio"]
        )
    )

    # Validate reconstruction against the actual
    # disturbed radiance, not the clean target.
    verify_quantization_bound(
        scenario_id,
        disturbed,
        y,
        config
    )

v06 = scenario_results["V06"]
v07 = scenario_results["V07"]
v08 = scenario_results["V08"]

check(
    "V06_physical_saturation_detected",
    v06["saturation_ratio"] > 0
)

check(
    "V06_ADC_clipping_detected",
    v06["adc_clipping_ratio"] > 0
)

check(
    "V07_saturation_suppressed",
    v07["saturation_ratio"] == 0
)

check(
    "V07_ADC_clipping_suppressed",
    v07["adc_clipping_ratio"] == 0
)

check(
    "V07_full_uncensored_coverage",
    v07["valid_observation_coverage"] == 1
)

check(
    "V08_no_full_well_saturation",
    v08["saturation_ratio"] == 0
)

check(
    "V08_ADC_clipping_reintroduced",
    v08["adc_clipping_ratio"] > 0
)

check(
    "V08_clipping_matches_reference",

    close(
        v08["adc_clipping_ratio"],
        v06["adc_clipping_ratio"]
    )
)


# ------------------------------------------------------------
# 8. V09 — Fully saturated observation
# ------------------------------------------------------------

print("\n[V09 — FULL SATURATION]")

saturated_scene = torch.full_like(
    clean,
    100.0
)

v09, _, _, _ = run_scenario(

    "V09",

    "fully_saturated",

    clean,

    saturated_scene,

    base
)

check(
    "V09_full_well_saturation_complete",
    v09["saturation_ratio"] == 1
)

check(
    "V09_zero_valid_coverage",
    v09["valid_observation_coverage"] == 0
)

check(
    "V09_zero_valid_sample_count",
    v09["valid_observation_count"] == 0
)

check(
    "V09_reconstruction_error_undefined",
    v09["radiance_reconstruction_mae"] is None
)

check(
    "V09_common_valid_error_undefined",
    v09["common_valid_radiance_distortion_mae"] is None
)

check(
    "V09_raw_distortion_remains_defined",
    v09["reference_mae"] > 0
)


# ------------------------------------------------------------
# 9. V10 — Empty evaluation region
# ------------------------------------------------------------

print("\n[V10 — EMPTY REGION]")

empty_region = torch.zeros(
    clean.shape[-2:],
    dtype=torch.bool
)

v10, _, _, _ = run_scenario(

    "V10",

    "empty_region",

    clean,

    clean,

    base,

    region=empty_region
)

check(
    "V10_zero_region_samples",
    v10["region_sample_count"] == 0
)

check(
    "V10_zero_region_fraction",
    v10["region_fraction"] == 0
)

check(
    "V10_regional_averages_undefined",

    all(

        v10[key] is None

        for key in (

            "saturation_ratio",

            "valid_observation_coverage",

            "configuration_mae",

            "disturbance_mae",

            "reference_mae",

            "radiance_reconstruction_mae",

            "gradient_mae",

            "contrast_retention"
        )
    )
)

check(
    "V10_ADC_levels_zero",
    v10["observed_adc_levels"] == 0
)

check(
    "V10_global_configuration_remains_defined",
    v10["exposure_utilization"] == 1
)


# ------------------------------------------------------------
# 10. V11 — Constant reference
# ------------------------------------------------------------

print("\n[V11 — CONSTANT REFERENCE]")

constant = torch.full(
    (3, 8, 8),
    0.5,
    dtype=torch.float32
)

v11, _, _, _ = run_scenario(

    "V11",

    "constant_reference",

    constant,

    constant,

    base
)

check(
    "V11_zero_reference_contrast",

    close(
        v11["reference_contrast"],
        0
    )
)

check(
    "V11_contrast_retention_undefined",
    v11["contrast_retention"] is None
)

check(
    "V11_zero_gradient_error",
    v11["gradient_mae"] == 0
)

check(
    "V11_single_ADC_level",
    v11["reference_adc_levels"] == 1
)


# ------------------------------------------------------------
# 11. V12 — Analytically known gradient
# ------------------------------------------------------------

print("\n[V12 — KNOWN SPATIAL GRADIENT]")

# With a 4-bit ADC and unit optical scale,
# these values represent exact ADC levels.

analytic_config = replace(

    base,

    exposure_ms=100.0,

    gain=1.0,

    adc_bits=4
)

gradient_reference = torch.zeros(
    (3, 2, 2),
    dtype=torch.float32
)

gradient_observed = torch.tensor(

    [

        [0.0, 0.2],

        [0.0, 0.2]
    ],

    dtype=torch.float32

).unsqueeze(0).repeat(3, 1, 1)

v12, _, _, _ = run_scenario(

    "V12",

    "analytic_gradient",

    gradient_reference,

    gradient_observed,

    analytic_config,

    reference_config=analytic_config
)

# Per channel:
# Horizontal edges: 0.2, 0.2
# Vertical edges:   0.0, 0.0
# Mean edge error:  0.1

check(
    "V12_analytic_gradient_MAE",

    close(
        v12["gradient_mae"],
        0.1
    )
)

check(
    "V12_analytic_disturbance_MAE",

    close(
        v12["disturbance_mae"],
        0.1
    )
)

check(
    "V12_zero_configuration_gradient",
    v12["configuration_gradient_mae"] == 0
)


# ------------------------------------------------------------
# 12. V13 — Analytically known radiance perturbation
# ------------------------------------------------------------

print("\n[V13 — KNOWN OPTICAL PERTURBATION]")

optical_reference = torch.full(
    (3, 4, 4),
    0.2,
    dtype=torch.float32
)

optical_observed = torch.full_like(
    optical_reference,
    0.4
)

v13, _, _, _ = run_scenario(

    "V13",

    "analytic_optical_perturbation",

    optical_reference,

    optical_observed,

    analytic_config,

    reference_config=analytic_config
)

check(
    "V13_known_radiance_reconstruction_error",

    close(
        v13["radiance_reconstruction_mae"],
        0.2
    )
)

check(
    "V13_known_common_valid_radiance_error",

    close(
        v13["common_valid_radiance_distortion_mae"],
        0.2
    )
)

check(
    "V13_uniform_perturbation_preserves_gradients",
    v13["gradient_mae"] == 0
)

check(
    "V13_full_valid_coverage",
    v13["valid_observation_coverage"] == 1
)


# ------------------------------------------------------------
# 13. V14 — Single-pixel region
# ------------------------------------------------------------

print("\n[V14 — SINGLE-PIXEL REGION]")

single_pixel = torch.zeros(
    clean.shape[-2:],
    dtype=torch.bool
)

single_pixel[0, 0] = True

v14, _, _, _ = run_scenario(

    "V14",

    "single_pixel_region",

    clean,

    disturbed,

    base,

    region=single_pixel
)

check(
    "V14_three_RGB_samples",
    v14["region_sample_count"] == 3
)

check(
    "V14_gradient_undefined_without_edges",
    v14["gradient_mae"] is None
)

check(
    "V14_reference_MAE_defined",
    v14["reference_mae"] is not None
)


# ------------------------------------------------------------
# 14. V15 — Complementary spatial regions
# ------------------------------------------------------------

print("\n[V15 — COMPLEMENTARY REGIONS]")

height, width = clean.shape[-2:]

left = torch.zeros(
    (height, width),
    dtype=torch.bool
)

left[:, :width // 2] = True

right = ~left

v15a, _, _, _ = run_scenario(

    "V15A",

    "left_region",

    clean,

    disturbed,

    base,

    region=left
)

v15b, _, _, _ = run_scenario(

    "V15B",

    "right_region",

    clean,

    disturbed,

    base,

    region=right.unsqueeze(0)
)

n_left = v15a["region_sample_count"]

n_right = v15b["region_sample_count"]

n_total = v06["region_sample_count"]

check(
    "V15_complementary_sample_accounting",
    n_left + n_right == n_total
)

for key in (

    "saturation_ratio",

    "adc_clipping_ratio",

    "censored_ratio",

    "valid_observation_coverage",

    "configuration_mae",

    "disturbance_mae",

    "reference_mae",

    "observed_mean_output"
):

    combined = (

        v15a[key] * n_left

        + v15b[key] * n_right

    ) / n_total

    check(

        "V15_weighted_aggregation_" + key,

        close(combined, v06[key])
    )

# Reconstruction error must be weighted by
# valid sample count, not total region size.

valid_left = v15a["valid_observation_count"]

valid_right = v15b["valid_observation_count"]

assert valid_left + valid_right > 0

combined_radiance_error = (

    v15a["radiance_reconstruction_mae"]
    * valid_left

    +

    v15b["radiance_reconstruction_mae"]
    * valid_right

) / (valid_left + valid_right)

check(
    "V15_valid_sample_weighted_radiance_error",

    close(

        combined_radiance_error,

        v06["radiance_reconstruction_mae"]
    )
)


# ------------------------------------------------------------
# 15. Cross-scenario mathematical invariants
# ------------------------------------------------------------

print("\n[CROSS-SCENARIO INVARIANTS]")

for scenario_id, m in scenario_results.items():

    if m["region_sample_count"] == 0:

        continue

    check(

        scenario_id + "_MAE_triangle_inequality",

        m["reference_mae"]

        <= (

            m["configuration_mae"]

            + m["disturbance_mae"]

            + 1e-10
        )
    )

    check(

        scenario_id + "_coverage_complement",

        close(

            m["censored_ratio"]

            + m["valid_observation_coverage"],

            1.0
        )
    )

    check(

        scenario_id + "_coverage_ordering",

        0

        <= m["nonzero_valid_coverage"]

        <= m["valid_observation_coverage"]

        <= 1
    )

    check(

        scenario_id + "_common_valid_ordering",

        0

        <= m["common_valid_coverage"]

        <= min(

            m["valid_observation_coverage"],

            m["clean_valid_coverage"]
        )
    )

    check(

        scenario_id + "_censoring_union_bounds",

        max(

            m["saturation_ratio"],

            m["adc_clipping_ratio"]

        ) <= m["censored_ratio"] <= min(

            1.0,

            m["saturation_ratio"]

            + m["adc_clipping_ratio"]
        )
    )

    check(

        scenario_id + "_valid_sample_accounting",

        close(

            m["valid_observation_count"]

            / m["region_sample_count"],

            m["valid_observation_coverage"]
        )
    )


# ------------------------------------------------------------
# 16. Display experimental results
# ------------------------------------------------------------

results = pd.DataFrame(rows)

check_df = pd.DataFrame(checks)

bounds = pd.DataFrame(bound_rows)

print("\n[CONTROLLED SCENARIO RESULTS]")

columns = [

    "scenario_id",

    "exposure_ms",

    "gain",

    "saturation_ratio",

    "adc_clipping_ratio",

    "valid_observation_coverage",

    "configuration_mae",

    "disturbance_mae",

    "reference_mae",

    "radiance_reconstruction_mae",

    "clean_adc_levels",

    "gradient_mae",

    "contrast_retention"
]

print(

    results[columns].to_string(

        index=False,

        float_format=lambda x: f"{x:.8f}"
    )
)

print("\n[PER-SAMPLE QUANTIZATION VERIFICATION]")

print(

    bounds.to_string(

        index=False,

        float_format=lambda x: f"{x:.10f}"
    )
)

passed = sum(
    item["passed"] for item in checks
)

failed = len(checks) - passed

assert len(results) == 16

assert len(scenario_results) == 16

assert len(bounds) == 8

assert results["scenario_id"].is_unique

report = {

    "task": TASK,

    "scenario_groups": 15,

    "scenario_evaluations": len(results),

    "scalar_metric_fields": 42,

    "checks_executed": len(checks),

    "checks_passed": passed,

    "checks_failed": failed,

    "quantization_bound_scenarios": len(bounds),

    "quantization_bound_checks_passed": int(
        bounds["passed"].sum()
    ),

    "noise_enabled": False,

    "scene_seed": 42,

    "reference_config": asdict(base),

    "attack_config": asdict(attack_config),

    "analytic_config": asdict(analytic_config),

    "metric_source_sha256": hashlib.sha256(

        (
            ROOT /
            "src/ahds/module7_evaluation/"
            "information_preservation.py"
        ).read_bytes()

    ).hexdigest(),

    "controller_executed": False,

    "ground_truth_evaluation_only": True,

    "all_checks_passed": failed == 0,

    "limitations": [

        "Deterministic noise-free validation.",

        "One generated scene and one optical disturbance.",

        "Analytic fixtures validate selected metric properties.",

        "No comparative controller performance evaluated.",

        "No downstream semantic perception evaluated."
    ]
}


# ------------------------------------------------------------
# 17. Archive controlled validation evidence
# ------------------------------------------------------------

OUT.mkdir(
    parents=True,
    exist_ok=True
)

results.to_csv(

    OUT / "controlled_metric_results.csv",

    index=False
)

check_df.to_csv(

    OUT / "validation_checks.csv",

    index=False
)

bounds.to_csv(

    OUT / "quantization_bound_verification.csv",

    index=False
)

(OUT / "validation_report.json").write_text(

    json.dumps(

        report,

        indent=2,

        allow_nan=False
    ),

    encoding="utf-8"
)

print("\n[ARCHIVED EVIDENCE]")

for filename in (

    "controlled_metric_results.csv",

    "validation_checks.csv",

    "quantization_bound_verification.csv",

    "validation_report.json"
):

    print("[SAVED]", OUT / filename)

print("\n" + "=" * 70)

print("CONTROLLED METRIC VALIDATION SUMMARY")

print("=" * 70)

print("Scenario groups: 15")

print("Scenario evaluations:", len(results))

print("Scalar metric fields: 42")

print("Validation checks:", len(checks))

print("Checks passed:", passed)

print("Checks failed:", failed)

print(

    "Quantization-bound scenarios:",

    int(bounds["passed"].sum()),

    "/", len(bounds)
)

print("Controller experiments executed: NONE")

assert failed == 0, (
    "Controlled metric validation failed. "
    "Inspect validation_checks.csv."
)

print("\nCONTROLLED METRIC VALIDATION: PASSED")
