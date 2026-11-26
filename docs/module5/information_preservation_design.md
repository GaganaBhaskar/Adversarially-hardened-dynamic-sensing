# AHDS Information-Preservation Metric Design

## 1. Purpose and architectural placement

The Adversarially-Hardened Dynamic Sensing (AHDS) project
investigates closed-loop physical-layer self-defense for
adaptive smart sensors.

The information-preservation metric suite supports Module 5
defensive-controller comparison and is implemented as a
reusable Module 7 evaluation component.

It does not change the established seven-module architecture
or the controller decision interfaces.

Implementation:

`src/ahds/module7_evaluation/information_preservation.py`

The metric suite provides physical and structural proxies for
information preservation. It does not directly quantify
semantic information, mutual information, entropy, or
downstream perception accuracy.

## 2. Three-observation evaluation framework

For each evaluated operating point or frame:

- R: clean optical scene captured at reference sensor settings.
- C: clean optical scene captured at applied sensor settings.
- Y: actual optical scene captured at applied sensor settings.
- L: clean optical ground truth.

C is a counterfactual observation used to isolate the effect
of sensor reconfiguration.

R, C, L and ground-truth spatial regions are evaluation-only
information. They must not be supplied as privileged inputs
to defensive controllers during comparative experiments.

## 3. Distortion decomposition

For a selected nonempty region with N RGB samples:

$$
D_{\mathrm{config}}=\frac{1}{N}\sum_i |C_i-R_i|
$$

$$
D_{\mathrm{dist}}=\frac{1}{N}\sum_i |Y_i-C_i|
$$

$$
D_{\mathrm{ref}}=\frac{1}{N}\sum_i |Y_i-R_i|
$$

All three normalized-output MAEs include censored samples.

The triangle inequality gives:

$$
D_{\mathrm{ref}}\leq
D_{\mathrm{config}}+D_{\mathrm{dist}}
$$

The distortion components are not generally additive.

## 4. Physical censoring

Full-well saturation and ADC clipping are separate physical
events in the sensor model.

The sensor accumulates charge, applies full-well saturation,
applies gain, and digitizes the resulting signal.

ADC clipping can therefore occur without full-well saturation.

Let S be the saturation mask and A the ADC clipping mask.

The censored mask is:

$$
M_{\mathrm{censored}}=S\lor A
$$

The uncensored mask is its logical complement.

An ADC code of zero is not automatically classified as
censored. Zero-valued uncensored observations remain in
radiance reconstruction error calculations.

Coverage must be reported alongside errors calculated over
uncensored observations.

## 5. Radiance reconstruction

Define the optical conversion scale:

$$
s=\frac{TGk}{Q_{\mathrm{FW}}}
$$

where T is exposure duration, G is gain, k is the electron
conversion coefficient, and Q_FW is full-well capacity.

For uncensored normalized digitized output Y:

$$
\widehat L=\frac{Y}{s}
$$

The reconstructed radiance is compared with clean optical
ground truth over the appropriate valid sample population.

The common-valid radiance distortion metric compares Y and C
after reconstruction using their shared applied configuration.

Common validity between Y and C does not automatically imply
a common evaluation population across different strategies.

Cross-strategy matched-valid comparisons remain an additional
comparative evaluation requirement.

## 6. Quantization

For b ADC bits, the maximum ADC code is:

$$
A_{\max}=2^b-1
$$

The radiance-domain quantization step is:

$$
\Delta L=\frac{1}{A_{\max}s}
$$

The ideal half-step bound is:

$$
\frac{\Delta L}{2}
$$

This bound describes ideal nearest-code quantization under
noise-free, uncensored conditions, subject to numerical
precision.

It is not a universal bound on reconstruction error in the
presence of optical disturbances or sensor noise.

Reducing exposure increases the quantization step when other
sensor parameters remain fixed.

Increasing gain can compensate for output-scale reduction,
but may reintroduce ADC clipping.

Distinct occupied ADC codes are descriptive diagnostics,
not direct measurements of information content or effective
number of bits.

## 7. Structural fidelity

Gradient discrepancies use signed horizontal and vertical
finite differences of normalized RGB output.

Only edges with both endpoints inside the selected region
are included.

Contrast is the population standard deviation of pooled
selected RGB samples, not a channel-wise or perceptual
contrast measurement.

Contrast retention close to one does not establish preserved
scene structure. Optical contamination can increase contrast.

Brightness, gradient and contrast diagnostics must be
interpreted jointly with physical censoring and reconstruction
error.

## 8. Spatial region semantics

Evaluation supports a full-image region or a boolean spatial
region mask broadcast across RGB channels.

Region-dependent statistics use the selected sample population.

Undefined statistics are represented by None, serialized as
JSON null, rather than artificial zero values.

Empty regions and regions without valid finite-difference
edges require explicit handling.

Ratio denominators at or below the configured numerical
tolerance are treated as undefined where applicable.

Gaussian optical disturbance tails can extend throughout
the image. A geometric background region must not be
described as completely unaffected without verification.

No final disturbance-region threshold is established by this
documentation. Comparative region definitions must be
declared before comparative results are examined.

## 9. Complete metric dictionary

### Spatial region accounting

| Metric | Definition |
|---|---|
| `region_sample_count` | Number of selected RGB samples. |
| `region_fraction` | Selected sample count divided by total image sample count. |


### Physical protection

| Metric | Definition |
|---|---|
| `saturation_ratio` | Fraction of selected observed samples flagged as full-well saturated. |
| `adc_clipping_ratio` | Fraction of selected observed samples flagged as ADC clipped. |
| `censored_ratio` | Fraction flagged by the union of saturation and ADC clipping masks. |


### Observation coverage

| Metric | Definition |
|---|---|
| `valid_observation_coverage` | Fraction of selected observed samples that are uncensored. |
| `clean_valid_coverage` | Uncensored fraction of the clean counterfactual observation. |
| `reference_valid_coverage` | Uncensored fraction of the clean reference observation. |
| `common_valid_coverage` | Fraction uncensored in both observed and counterfactual observations. |
| `nonzero_valid_coverage` | Fraction of selected samples that are uncensored and have a nonzero observed ADC code. |
| `valid_observation_count` | Number of selected uncensored observed samples. |
| `common_valid_count` | Number of selected samples uncensored in both observed and counterfactual observations. |


### Counterfactual distortion

| Metric | Definition |
|---|---|
| `configuration_mae` | Mean absolute difference between counterfactual and reference normalized outputs. |
| `disturbance_mae` | Mean absolute difference between observed and counterfactual normalized outputs. |
| `reference_mae` | Mean absolute difference between observed and reference normalized outputs. |


### Radiance reconstruction

| Metric | Definition |
|---|---|
| `radiance_reconstruction_mae` | Observed reconstructed radiance error relative to clean optical ground truth, over uncensored observed samples. |
| `clean_radiance_reconstruction_mae` | Counterfactual reconstructed radiance error relative to clean ground truth, over uncensored counterfactual samples. |
| `reference_radiance_reconstruction_mae` | Reference reconstructed radiance error relative to clean ground truth, over uncensored reference samples. |
| `common_valid_radiance_distortion_mae` | Observed versus counterfactual reconstructed radiance error over their common uncensored samples. |


### Operating-point utilization

| Metric | Definition |
|---|---|
| `exposure_utilization` | Applied exposure divided by reference exposure. |
| `gain_utilization` | Applied gain divided by reference gain. |
| `optical_scale_utilization` | Applied radiance-to-normalized-output scale divided by reference scale. |


### Quantization diagnostics

| Metric | Definition |
|---|---|
| `radiance_quantization_step` | Applied ADC code spacing expressed in optical radiance units. |
| `ideal_quantization_half_step` | Half of the applied radiance quantization step. |
| `relative_quantization_step` | Applied radiance quantization step divided by reference step. |
| `reference_adc_levels` | Number of distinct reference ADC codes in the selected region. |
| `clean_adc_levels` | Number of distinct counterfactual ADC codes in the selected region. |
| `observed_adc_levels` | Number of distinct observed ADC codes in the selected region. |
| `clean_adc_level_retention` | Counterfactual distinct ADC code count divided by reference count. |
| `clean_zero_adc_ratio` | Fraction of selected counterfactual samples with ADC code zero. |
| `observed_zero_adc_ratio` | Fraction of selected observed samples with ADC code zero. |


### Structural fidelity

| Metric | Definition |
|---|---|
| `gradient_mae` | Mean absolute signed finite-difference discrepancy between observed and reference outputs. |
| `configuration_gradient_mae` | Mean absolute signed finite-difference discrepancy between counterfactual and reference outputs. |
| `disturbance_gradient_mae` | Mean absolute signed finite-difference discrepancy between observed and counterfactual outputs. |
| `reference_contrast` | Population standard deviation of selected reference RGB samples. |
| `clean_contrast` | Population standard deviation of selected counterfactual RGB samples. |
| `observed_contrast` | Population standard deviation of selected observed RGB samples. |
| `contrast_retention` | Observed contrast divided by reference contrast. |
| `clean_contrast_retention` | Counterfactual contrast divided by reference contrast. |
| `reference_mean_output` | Mean selected reference normalized output. |
| `clean_mean_output` | Mean selected counterfactual normalized output. |
| `observed_mean_output` | Mean selected observed normalized output. |


## 10. Interpretation and limitations

The metric suite supports multi-objective evaluation rather
than an arbitrary weighted aggregate score.

A strategy with lower saturation is not automatically better
if it sacrifices radiance precision, observation coverage,
structural fidelity or downstream task performance.

Nonzero valid coverage is not a validated signal-resolution
or signal-to-noise metric.

The current controlled validation is deterministic and
noise-free. Noise-aware validation requires an explicit
noise-generation and comparison protocol.

The simulated optical conversion coefficient is not a
hardware-calibrated radiometric measurement.

The current metric validation does not establish hardware
performance, controller superiority, real-time feasibility,
energy savings or semantic perception robustness.

Extreme numerical operating points outside the controlled
validation domain have not been comprehensively characterized.

These limitations must remain visible in subsequent AHDS
scientific reports.

