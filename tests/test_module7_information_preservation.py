
"""
AHDS information-preservation metric regression tests.

Tests use deterministic synthetic observations and the
committed physics-informed sensor simulator.

Evaluation ground truth remains isolated from controllers.
"""

from dataclasses import replace
import json

import pytest
import torch

from src.ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor
)

from src.ahds.module7_evaluation.information_preservation import (
    InformationMetricConfig,
    InformationPreservationMetrics
)


@pytest.fixture
def evaluator():

    return InformationPreservationMetrics()


@pytest.fixture
def config():

    return SensorConfig(
        exposure_ms=10.0,
        gain=1.0
    )


@pytest.fixture
def scene():

    return torch.linspace(
        0.05,
        0.95,
        3 * 8 * 8
    ).reshape(3, 8, 8)


def capture(scene, config):

    return simulate_sensor(
        scene,
        config,
        add_noise=False
    )


def evaluate(
    evaluator,
    scene,
    config,
    applied=None,
    observed_scene=None,
    region=None
):

    if applied is None:

        applied = config

    if observed_scene is None:

        observed_scene = scene

    reference = capture(scene, config)

    counterfactual = capture(scene, applied)

    observed = capture(observed_scene, applied)

    return evaluator.evaluate(

        reference_result=reference,

        counterfactual_result=counterfactual,

        observed_result=observed,

        reference_config=config,

        applied_config=applied,

        clean_optical_scene=scene,

        region_mask=region
    )


def evaluate_modified(
    evaluator,
    scene,
    config,
    modifications
):

    reference = capture(scene, config)

    observed = dict(reference)

    observed.update(modifications)

    return evaluator.evaluate(

        reference_result=reference,

        counterfactual_result=reference,

        observed_result=observed,

        reference_config=config,

        applied_config=config,

        clean_optical_scene=scene
    )


# ============================================================
# A. Configuration validation
# ============================================================

@pytest.mark.parametrize(
    "epsilon",
    [0.0, -1.0, float("nan"),
     float("inf"), -float("inf")]
)
def test_invalid_epsilon(epsilon):

    with pytest.raises(ValueError):

        InformationMetricConfig(
            epsilon=epsilon
        )


def test_valid_custom_epsilon():

    config = InformationMetricConfig(
        epsilon=1e-9
    )

    assert config.epsilon == 1e-9


# ============================================================
# B. Identity and numerical contracts
# ============================================================

def test_identical_observations(
    evaluator, scene, config
):

    m = evaluate(
        evaluator, scene, config
    )

    for key in (

        "configuration_mae",
        "disturbance_mae",
        "reference_mae",
        "gradient_mae",
        "configuration_gradient_mae",
        "disturbance_gradient_mae",
        "common_valid_radiance_distortion_mae"
    ):

        assert m[key] == 0.0


def test_identity_ratios(
    evaluator, scene, config
):

    m = evaluate(
        evaluator, scene, config
    )

    for key in (

        "valid_observation_coverage",
        "common_valid_coverage",
        "nonzero_valid_coverage",
        "exposure_utilization",
        "gain_utilization",
        "optical_scale_utilization",
        "relative_quantization_step",
        "contrast_retention",
        "clean_contrast_retention",
        "clean_adc_level_retention"
    ):

        assert m[key] == pytest.approx(1.0)


def test_output_contract(
    evaluator, scene, config
):

    m = evaluate(
        evaluator, scene, config
    )

    assert len(m) == 42

    assert m["region_sample_count"] == 192

    assert m["region_fraction"] == 1.0

    assert m["valid_observation_count"] == 192

    assert m["common_valid_count"] == 192

    assert all(

        isinstance(v, (float, int))
        or v is None

        for v in m.values()
    )

    json.dumps(m, allow_nan=False)


@pytest.mark.parametrize(
    "exposure",
    [5.0, 1.0, 0.1]
)
def test_clean_exposure_reduction(
    evaluator, scene, config, exposure
):

    applied = replace(
        config,
        exposure_ms=exposure
    )

    m = evaluate(
        evaluator,
        scene,
        config,
        applied=applied
    )

    assert m["configuration_mae"] > 0

    assert m["disturbance_mae"] == 0

    assert m["reference_mae"] == pytest.approx(
        m["configuration_mae"]
    )

    assert m["exposure_utilization"] == (
        pytest.approx(exposure / 10)
    )

    assert m["relative_quantization_step"] == (
        pytest.approx(10 / exposure)
    )


def test_gain_compensation_identity(
    evaluator, scene, config
):

    applied = replace(
        config,
        exposure_ms=5.0,
        gain=2.0
    )

    m = evaluate(
        evaluator,
        scene,
        config,
        applied=applied
    )

    assert m["configuration_mae"] == 0

    assert m["exposure_utilization"] == 0.5

    assert m["gain_utilization"] == 2.0

    assert m["optical_scale_utilization"] == 1

    assert m["relative_quantization_step"] == 1


@pytest.mark.parametrize(
    "exposure,gain",
    [(10.0, 1.0),
     (5.0, 1.0),
     (1.0, 1.0),
     (0.1, 1.0),
     (5.0, 2.0)]
)
def test_quantization_formula(
    evaluator, scene, config, exposure, gain
):

    applied = replace(
        config,
        exposure_ms=exposure,
        gain=gain
    )

    m = evaluate(
        evaluator,
        scene,
        config,
        applied=applied
    )

    expected = (

        applied.full_well_electrons

        /

        (
            (2 ** applied.adc_bits - 1)
            * exposure
            * gain
            * applied.electrons_per_unit_ms
        )
    )

    assert m["radiance_quantization_step"] == (
        pytest.approx(expected)
    )

    assert m["ideal_quantization_half_step"] == (
        pytest.approx(expected / 2)
    )

    assert (
        m["radiance_reconstruction_mae"]
        <= expected / 2 + 1e-5
    )


# ============================================================
# C. Physical protection and coverage
# ============================================================

def test_full_saturation(
    evaluator, scene, config
):

    m = evaluate(
        evaluator,
        scene,
        config,
        observed_scene=torch.full_like(
            scene, 100.0
        )
    )

    assert m["saturation_ratio"] == 1

    assert m["adc_clipping_ratio"] == 1

    assert m["censored_ratio"] == 1

    assert m["valid_observation_coverage"] == 0

    assert m["valid_observation_count"] == 0

    assert m["radiance_reconstruction_mae"] is None

    assert (
        m["common_valid_radiance_distortion_mae"]
        is None
    )

    assert m["reference_mae"] > 0


def test_adc_clipping_without_full_well_saturation(
    evaluator, scene, config
):

    applied = replace(
        config,
        exposure_ms=5.0,
        gain=2.0
    )

    # Full-well threshold is 20 radiance units
    # at 5 ms; ADC clipping threshold is 10.

    m = evaluate(
        evaluator,
        scene,
        config,
        applied=applied,
        observed_scene=torch.full_like(
            scene, 15.0
        )
    )

    assert m["saturation_ratio"] == 0

    assert m["adc_clipping_ratio"] == 1

    assert m["censored_ratio"] == 1

    assert m["valid_observation_coverage"] == 0


def test_zero_signal_is_not_censored(
    evaluator, scene, config
):

    zero = torch.zeros_like(scene)

    m = evaluate(
        evaluator,
        zero,
        config
    )

    assert m["valid_observation_coverage"] == 1

    assert m["nonzero_valid_coverage"] == 0

    assert m["observed_zero_adc_ratio"] == 1

    assert m["radiance_reconstruction_mae"] == 0


def test_partial_censoring(
    evaluator, scene, config
):

    observed = scene.clone()

    observed[:, :4, :] = 100.0

    m = evaluate(
        evaluator,
        scene,
        config,
        observed_scene=observed
    )

    assert m["saturation_ratio"] == 0.5

    assert m["adc_clipping_ratio"] == 0.5

    assert m["censored_ratio"] == 0.5

    assert m["valid_observation_coverage"] == 0.5

    assert m["valid_observation_count"] == 96

    assert m["common_valid_count"] == 96

    assert m["radiance_reconstruction_mae"] is not None


def test_censoring_mask_union(
    evaluator, scene, config
):

    saturated = torch.zeros_like(
        scene, dtype=torch.bool
    )

    clipped = torch.zeros_like(
        scene, dtype=torch.bool
    )

    saturated[:, :4, :] = True

    clipped[:, :, :4] = True

    m = evaluate_modified(
        evaluator,
        scene,
        config,
        {
            "saturated": saturated,
            "adc_clipped": clipped
        }
    )

    # Synthetic mask fixture:
    # each mask covers 50%, overlap is 25%.

    assert m["saturation_ratio"] == 0.5

    assert m["adc_clipping_ratio"] == 0.5

    assert m["censored_ratio"] == 0.75

    assert m["valid_observation_coverage"] == 0.25


# ============================================================
# D. Analytical metric correctness
# ============================================================

def test_analytic_gradient(evaluator, config):

    analytic = replace(
        config,
        exposure_ms=100.0,
        adc_bits=4
    )

    reference = torch.zeros(3, 2, 2)

    observed = torch.tensor(
        [[0.0, 0.2],
         [0.0, 0.2]]
    ).unsqueeze(0).repeat(3, 1, 1)

    m = evaluate(
        evaluator,
        reference,
        analytic,
        observed_scene=observed
    )

    assert m["gradient_mae"] == pytest.approx(0.1)

    assert m["disturbance_mae"] == pytest.approx(0.1)

    assert m["configuration_gradient_mae"] == 0


def test_analytic_radiance_error(
    evaluator, config
):

    analytic = replace(
        config,
        exposure_ms=100.0,
        adc_bits=4
    )

    reference = torch.full(
        (3, 4, 4), 0.2
    )

    observed = torch.full_like(
        reference, 0.4
    )

    m = evaluate(
        evaluator,
        reference,
        analytic,
        observed_scene=observed
    )

    assert m["radiance_reconstruction_mae"] == (
        pytest.approx(0.2)
    )

    assert m["common_valid_radiance_distortion_mae"] == (
        pytest.approx(0.2)
    )

    assert m["gradient_mae"] == 0


def test_mae_triangle_inequality(
    evaluator, scene, config
):

    m = evaluate(
        evaluator,
        scene,
        config,
        applied=replace(
            config,
            exposure_ms=5.0
        ),
        observed_scene=scene + 2.0
    )

    assert m["reference_mae"] <= (

        m["configuration_mae"]

        + m["disturbance_mae"]

        + 1e-12
    )


# ============================================================
# E. Spatial regions and undefined values
# ============================================================

def test_empty_region(
    evaluator, scene, config
):

    region = torch.zeros(
        (8, 8),
        dtype=torch.bool
    )

    m = evaluate(
        evaluator,
        scene,
        config,
        region=region
    )

    assert m["region_sample_count"] == 0

    assert m["region_fraction"] == 0

    assert m["observed_adc_levels"] == 0

    for key in (

        "saturation_ratio",
        "valid_observation_coverage",
        "reference_mae",
        "radiance_reconstruction_mae",
        "gradient_mae",
        "contrast_retention"
    ):

        assert m[key] is None

    assert m["exposure_utilization"] == 1

    json.dumps(m, allow_nan=False)


def test_single_pixel_region(
    evaluator, scene, config
):

    region = torch.zeros(
        (8, 8),
        dtype=torch.bool
    )

    region[2, 2] = True

    m = evaluate(
        evaluator,
        scene,
        config,
        region=region
    )

    assert m["region_sample_count"] == 3

    assert m["gradient_mae"] is None

    assert m["reference_mae"] == 0


def test_disconnected_region_has_no_edges(
    evaluator, scene, config
):

    region = torch.zeros(
        (8, 8),
        dtype=torch.bool
    )

    region[0, 0] = True

    region[7, 7] = True

    m = evaluate(
        evaluator,
        scene,
        config,
        region=region
    )

    assert m["region_sample_count"] == 6

    assert m["gradient_mae"] is None


def test_constant_reference(
    evaluator, scene, config
):

    constant = torch.full_like(
        scene, 0.5
    )

    m = evaluate(
        evaluator,
        constant,
        config
    )

    assert m["reference_contrast"] == (
        pytest.approx(0.0)
    )

    assert m["contrast_retention"] is None

    assert m["clean_contrast_retention"] is None

    assert m["reference_adc_levels"] == 1


def test_region_shape_equivalence(
    evaluator, scene, config
):

    region = torch.zeros(
        (8, 8),
        dtype=torch.bool
    )

    region[:, :4] = True

    a = evaluate(
        evaluator,
        scene,
        config,
        region=region
    )

    b = evaluate(
        evaluator,
        scene,
        config,
        region=region.unsqueeze(0)
    )

    assert a == b


def test_full_region_equivalence(
    evaluator, scene, config
):

    region = torch.ones(
        (8, 8),
        dtype=torch.bool
    )

    a = evaluate(
        evaluator, scene, config
    )

    b = evaluate(
        evaluator,
        scene,
        config,
        region=region
    )

    assert a == b


def test_complementary_region_aggregation(
    evaluator, scene, config
):

    observed = scene.clone()

    observed[:, :2, :3] = 100.0

    region = torch.zeros(
        (8, 8),
        dtype=torch.bool
    )

    region[:, :3] = True

    full = evaluate(
        evaluator,
        scene,
        config,
        observed_scene=observed
    )

    a = evaluate(
        evaluator,
        scene,
        config,
        observed_scene=observed,
        region=region
    )

    b = evaluate(
        evaluator,
        scene,
        config,
        observed_scene=observed,
        region=~region
    )

    n1 = a["region_sample_count"]

    n2 = b["region_sample_count"]

    assert n1 + n2 == full["region_sample_count"]

    for key in (

        "saturation_ratio",
        "adc_clipping_ratio",
        "censored_ratio",
        "valid_observation_coverage",
        "reference_mae",
        "disturbance_mae"
    ):

        combined = (

            a[key] * n1

            + b[key] * n2

        ) / (n1 + n2)

        assert combined == pytest.approx(
            full[key]
        )

    v1 = a["valid_observation_count"]

    v2 = b["valid_observation_count"]

    combined_error = (

        a["radiance_reconstruction_mae"] * v1

        + b["radiance_reconstruction_mae"] * v2

    ) / (v1 + v2)

    assert combined_error == pytest.approx(
        full["radiance_reconstruction_mae"]
    )


# ============================================================
# F. Invalid normalized sensor observations
# ============================================================

@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), -0.1, 1.1]
)
def test_invalid_normalized_values(
    evaluator, scene, config, value
):

    image = torch.full_like(
        scene, value
    )

    with pytest.raises(ValueError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {"normalized_output": image}
        )


@pytest.mark.parametrize(
    "shape",
    [(8, 8), (1, 8, 8),
     (4, 8, 8), (3, 0, 8)]
)
def test_invalid_normalized_shape(
    evaluator, scene, config, shape
):

    image = torch.zeros(shape)

    with pytest.raises(ValueError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {"normalized_output": image}
        )


def test_nonfloating_normalized_output(
    evaluator, scene, config
):

    with pytest.raises(TypeError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {
                "normalized_output":
                    torch.zeros_like(
                        scene,
                        dtype=torch.int64
                    )
            }
        )


def test_nontensor_normalized_output(
    evaluator, scene, config
):

    with pytest.raises(TypeError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {"normalized_output": []}
        )


# ============================================================
# G. Invalid physical masks
# ============================================================

@pytest.mark.parametrize(
    "field",
    ["saturated", "adc_clipped"]
)
def test_nonboolean_sensor_mask(
    evaluator, scene, config, field
):

    with pytest.raises(TypeError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {field: torch.zeros_like(scene)}
        )


@pytest.mark.parametrize(
    "field",
    ["saturated", "adc_clipped"]
)
def test_invalid_sensor_mask_shape(
    evaluator, scene, config, field
):

    with pytest.raises(ValueError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {
                field: torch.zeros(
                    (8, 8),
                    dtype=torch.bool
                )
            }
        )


# ============================================================
# H. Invalid ADC observations
# ============================================================

def test_noninteger_adc(
    evaluator, scene, config
):

    with pytest.raises(TypeError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {"adc_output": torch.zeros_like(scene)}
        )


@pytest.mark.parametrize(
    "value",
    [-1, 4096]
)
def test_adc_out_of_range(
    evaluator, scene, config, value
):

    adc = torch.full(
        scene.shape,
        value,
        dtype=torch.int64
    )

    with pytest.raises(ValueError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {"adc_output": adc}
        )


def test_adc_shape_mismatch(
    evaluator, scene, config
):

    with pytest.raises(ValueError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {
                "adc_output": torch.zeros(
                    (8, 8),
                    dtype=torch.int64
                )
            }
        )


def test_adc_normalized_inconsistency(
    evaluator, scene, config
):

    with pytest.raises(ValueError):

        evaluate_modified(
            evaluator,
            scene,
            config,
            {
                "adc_output": torch.zeros_like(
                    scene,
                    dtype=torch.int64
                )
            }
        )


# ============================================================
# I. Invalid evaluation regions and optical ground truth
# ============================================================

@pytest.mark.parametrize(
    "shape",
    [(7, 8), (3, 8, 8), (1, 7, 8)]
)
def test_invalid_region_shape(
    evaluator, scene, config, shape
):

    region = torch.ones(
        shape,
        dtype=torch.bool
    )

    with pytest.raises(ValueError):

        evaluate(
            evaluator,
            scene,
            config,
            region=region
        )


def test_nonboolean_region(
    evaluator, scene, config
):

    with pytest.raises(TypeError):

        evaluate(
            evaluator,
            scene,
            config,
            region=torch.ones(8, 8)
        )


def test_nontensor_region(
    evaluator, scene, config
):

    with pytest.raises(TypeError):

        evaluate(
            evaluator,
            scene,
            config,
            region=[]
        )


@pytest.mark.parametrize(
    "value",
    [-0.1, float("nan"), float("inf")]
)
def test_invalid_optical_ground_truth(
    evaluator, scene, config, value
):

    reference = capture(scene, config)

    invalid = torch.full_like(
        scene, value
    )

    with pytest.raises(ValueError):

        evaluator.evaluate(
            reference_result=reference,
            counterfactual_result=reference,
            observed_result=reference,
            reference_config=config,
            applied_config=config,
            clean_optical_scene=invalid
        )


def test_ground_truth_shape_mismatch(
    evaluator, scene, config
):

    reference = capture(scene, config)

    with pytest.raises(ValueError):

        evaluator.evaluate(
            reference_result=reference,
            counterfactual_result=reference,
            observed_result=reference,
            reference_config=config,
            applied_config=config,
            clean_optical_scene=torch.zeros(3, 4, 4)
        )


# ============================================================
# J. Repeatability and input preservation
# ============================================================

def test_repeatable_metric_evaluation(
    evaluator, scene, config
):

    first = evaluate(
        evaluator, scene, config
    )

    second = evaluate(
        evaluator, scene, config
    )

    assert first == second


def test_evaluation_does_not_modify_inputs(
    evaluator, scene, config
):

    result = capture(scene, config)

    originals = {

        key: value.clone()

        for key, value in result.items()

        if isinstance(value, torch.Tensor)
    }

    original_scene = scene.clone()

    evaluator.evaluate(
        reference_result=result,
        counterfactual_result=result,
        observed_result=result,
        reference_config=config,
        applied_config=config,
        clean_optical_scene=scene
    )

    for key, original in originals.items():

        assert torch.equal(
            result[key], original
        )

    assert torch.equal(
        scene, original_scene
    )
