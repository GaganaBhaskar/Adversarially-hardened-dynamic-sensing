# AHDS — Five-Strategy Comparative Protocol

## Research identification

- Task: M5-VAL-01B-01B
- Protocol version: 1.0
- Parent research checkpoint: 453c68727e96ee373c1afa8b28d6413f74c8c52f
- Parent Git tree: 8838d787c2ae4c9a53b60f0aa638d5e9c3116003

## Objective

Evaluate physical protection, measurement preservation,
exposure efficiency, persistent disturbance behavior and
post-disturbance recovery across five defensive strategies.

The protocol does not assume DADRC superiority.

## Experimental strategies

### S0 — Fixed exposure

- Controller: None
- Interface: None
- Trusted startup reference: False

### S1 — Original exposure protection

- Controller: ExposureProtectionController
- Interface: ['current_exposure_ms', 'assessment']
- Trusted startup reference: False

```json
{
  "min_exposure_ms": 0.1,
  "max_exposure_ms": 30.0,
  "saturation_trigger": 0.005,
  "reduction_factor": 0.5,
  "recovery_factor": 1.1,
  "recovery_delay_frames": 3
}
```

### S2 — Probed exposure

- Controller: ProbedExposureController
- Interface: ['current_exposure_ms', 'assessment']
- Trusted startup reference: False

```json
{
  "min_exposure_ms": 0.1,
  "max_exposure_ms": 30.0,
  "saturation_trigger": 0.005,
  "reduction_factor": 0.5,
  "probe_factor": 1.05,
  "probe_interval_frames": 4,
  "failed_probe_cooldown": 8
}
```

### S3 — State-aware exposure

- Controller: StateAwareExposureController
- Interface: ['current_exposure_ms', 'assessment', 'state']
- Trusted startup reference: False

```json
{
  "min_exposure_ms": 0.1,
  "max_exposure_ms": 30.0,
  "saturation_trigger": 0.005,
  "reduction_factor": 0.5,
  "recovery_factor": 1.1
}
```

### S4 — DADRC

- Controller: DisturbanceAwareDefenseController
- Interface: ['sensor_result', 'sensor_config', 'assessment', 'tracker_state']
- Trusted startup reference: True

```json
{
  "min_exposure_ms": 0.1,
  "max_exposure_ms": 30.0,
  "saturation_trigger": 0.005,
  "reduction_factor": 0.5,
  "persistent_probe_factor": 1.05,
  "clear_probe_factor": 1.1,
  "recovery_factor": 1.2,
  "persistent_probe_interval": 4,
  "clear_confirmation_frames": 3,
  "failed_probe_cooldown": 8
}
```


S1 retains its original behavior, including exposure
increases above the nominal initial exposure.

S4 receives a separate trusted startup observation before
frame zero. This is an additional initialization requirement
and is not counted as a comparative frame.

S2 is the previously reconstructed behavioral baseline.

## Experimental environment

The initial benchmark uses one deterministic synthetic road
scene and a previously explored persistent optical disturbance.

All strategies start from the same sensor configuration.
Their subsequent applied exposure trajectories may differ.

```json
{
  "scene": "deterministic_synthetic_road",
  "height": 256,
  "width": 256,
  "channels": 3,
  "device": "cpu",
  "seed": 42,
  "scene_randomized": false,
  "sensor_noise_enabled": false,
  "total_frames": 60,
  "initial_sensor_config": {
    "exposure_ms": 10.0,
    "gain": 1.0,
    "electrons_per_unit_ms": 100.0,
    "full_well_electrons": 10000.0,
    "read_noise_electrons": 3.0,
    "adc_bits": 12
  },
  "gain_adaptation_enabled": false,
  "roi_adaptation_enabled": false,
  "module3_adaptation_enabled": false,
  "command_delay_frames": 1,
  "attack_config": {
    "attack_type": "laser",
    "center_x": 0.5,
    "center_y": 0.5,
    "radius": 0.12,
    "intensity": 15.0,
    "pulse_frequency": 2.0,
    "pulse_phase": 0.0
  },
  "canonical_attack_map_time_step": 0,
  "attack_map_policy": "Generate one canonical active map and reuse it for every active frame.",
  "phases": {
    "clean": {
      "start": 0,
      "stop_exclusive": 10
    },
    "disturbance": {
      "start": 10,
      "stop_exclusive": 30
    },
    "recovery": {
      "start": 30,
      "stop_exclusive": 60
    }
  }
}

```

## Execution order

For each strategy, initialize independent detector, tracker
and controller instances.

For each frame:

1. Apply the previous exposure command.
2. Select the actual optical scene.
3. Capture the actual sensor observation Y.
4. Compute the disturbance assessment.
5. Update the disturbance tracker.
6. Execute the strategy-specific controller.
7. Generate or retrieve evaluation-only R and C observations.
8. Evaluate all regional information-preservation metrics.
9. Record diagnostics and retain the next exposure command.

The controller receives no evaluation-only scene, mask,
phase label or information-preservation metric.

## Controller information boundary

```json
{
  "permitted_sensor_fields": [
    "normalized_output",
    "saturated",
    "adc_clipped"
  ],
  "permitted_operating_information": [
    "applied_sensor_config",
    "disturbance_assessment",
    "tracker_state",
    "controller_internal_state"
  ],
  "evaluation_only_information": [
    "clean_optical_scene",
    "actual_optical_scene",
    "attack_map",
    "attack_active",
    "phase_label",
    "reference_result_R",
    "counterfactual_result_C",
    "evaluation_region_masks",
    "matched_valid_masks",
    "information_preservation_metrics",
    "expected_electrons",
    "electrons"
  ],
  "S4_initialization": {
    "separate_startup_capture": true,
    "before_frame_zero": true,
    "trusted_clean_startup": true,
    "initial_exposure_ms": 10.0,
    "initial_gain": 1.0,
    "sensor_noise_enabled": false,
    "count_as_comparative_frame": false,
    "reuse_evaluation_reference_object": false
  },
  "ground_truth_feedback_allowed": false,
  "controller_state_shared_between_strategies": false
}

```

## Spatial evaluation

The canonical active disturbance map defines fixed regions
for every frame and every strategy.

The core, transition and background regions must form a
disjoint and complete partition of the spatial image.

The background is a low-disturbance region, not a guarantee
of zero optical interference.

```json
{
  "regions": {
    "full": {
      "definition": "all_pixels"
    },
    "core": {
      "minimum_relative_peak": 0.1,
      "minimum_inclusive": true
    },
    "transition": {
      "minimum_relative_peak": 0.01,
      "minimum_inclusive": true,
      "maximum_relative_peak": 0.1,
      "maximum_inclusive": false
    },
    "background": {
      "maximum_relative_peak": 0.01,
      "maximum_inclusive": false
    }
  },
  "mask_source": "canonical_active_attack_map",
  "fixed_across_frames": true,
  "fixed_across_strategies": true,
  "require_positive_finite_peak": true,
  "require_partition": true,
  "require_disjoint_subregions": true,
  "mask_shapes": [
    "H,W",
    "1,H,W"
  ],
  "sample_accounting": "RGB channel samples",
  "empty_region_policy": "Preserve undefined metrics as null.",
  "background_interpretation": "Low-disturbance region, not necessarily an unaffected region."
}

```

## Information-preservation evaluation

R: clean scene at nominal reference settings.

C: clean scene at the strategy's current applied settings.

Y: actual optical scene at the strategy's current settings.

Configuration MAE compares C with R.

Disturbance MAE compares Y with C.

Reference MAE compares Y with R.

These errors are not assumed to be additive.

All 42 validated metric fields will be recorded.

### Primary metrics

- saturation_ratio
- adc_clipping_ratio
- censored_ratio
- valid_observation_coverage
- configuration_mae
- disturbance_mae
- reference_mae
- radiance_reconstruction_mae
- radiance_quantization_step
- gradient_mae
- exposure_utilization


### Matched-valid comparisons

Individual conditional radiance errors must be accompanied
by valid sample coverage.

For each pair of strategies, compare reconstruction errors
on the intersection of their uncensored RGB samples.

Additionally evaluate all five strategies on the intersection
of all five valid-sample masks.

Zero ADC samples remain included when uncensored.

Empty intersections produce undefined errors, not zero errors.

Report sample counts and coverage for every intersection.

Phase summaries report defined and undefined frame counts.
Conditional radiance MAE is also aggregated using valid
sample counts as weights.

No arbitrary weighted composite performance score is used.

## Recovery specification

A recovery frame qualifies when:

- Applied exposure is within 5 percent of nominal.
- Full-frame censored ratio is below 0.005.
- Full-frame reference MAE is at most 0.002.

Three consecutive qualifying frames are required.

The first qualifying frame defines the restoration frame.
The third frame defines the confirmation frame.

Report whether the restored condition remains sustained
through the end of the observation window.

Recovery delay is restoration frame minus frame 30.

If no qualifying sequence exists, report an undefined delay
and the not-restored-within-observation-window status.

The thresholds are operational evaluation conventions,
not hardware-calibrated quality requirements.

## Protection response specification

Record the first exposure reduction command and the first
frame applying that reduction.

Record first saturation clearance and first censoring
clearance during the disturbance interval.

Report subsequent threshold violations and distinguish
strategies already below the threshold at disturbance onset.

Report applied exposure at disturbance onset, exposure
overshoot, exposure total variation and probe diagnostics.

## Fairness requirements

- F01: Identical clean optical scene.
- F02: Identical active disturbance map and schedule.
- F03: Identical initial sensor configuration.
- F04: Independent controller and detector state.
- F05: Identical sensor simulator and noise policy.
- F06: Identical one-frame command application delay.
- F07: Identical information-preservation evaluator.
- F08: Identical fixed evaluation regions.
- F09: No evaluation ground truth enters controllers.
- F10: No post-result controller tuning.
- F11: Explicit disclosure of S4 startup reference.
- F12: Undefined metrics remain null.
- F13: Identical experimental phase boundaries.
- F14: Matched-valid cross-strategy comparisons.
- F15: Preserve all historical research artifacts.
- F16: Preserve original S1 nominal exposure overshoot.
- F17: Use common observed recovery criteria.
- F18: Do not use internal controller labels as ground truth.
- F19: Record actual applied and next commanded exposure.
- F20: Retain unfavorable and negative results.

## Scientific hypotheses

- H01: Adaptive defensive strategies reduce physical censoring relative to fixed exposure.
- H02: DADRC retains persistent disturbance evidence after saturation disappears.
- H03: DADRC improves the protection-preservation trade-off relative to defensive baselines.
- H04: DADRC restores nominal sensing conditions after disturbance removal.
- H05: Lower saturation does not necessarily imply better radiance reconstruction or structural fidelity.

## Expected experimental accounting

- strategies: 5
- frames_per_strategy: 60
- strategy_frame_records: 300
- regions_per_frame: 4
- regional_metric_records: 1200
- metric_fields_per_regional_record: 42
- metric_field_entries: 50400
- unique_strategy_pairs: 10
- pairwise_matched_region_records: 2400
- all_strategy_matched_region_records: 240

## Scientific limitations

- Single deterministic synthetic scene.
- Single previously explored optical disturbance scenario.
- Noise-free controlled experiment.
- No independent generalization or statistical inference.
- No calibrated physical laser power.
- Simplified linear RGB sensor model.
- No dynamic gain or ROI adaptation.
- Module 3 integration is deferred.
- S4 requires an additional trusted startup observation.
- S2 is the previously reconstructed behavioral baseline.
- Spatial thresholds are evaluation conventions.
- Recovery thresholds are proposed operational conventions.
- Measurement fidelity is not semantic information preservation.
- Conditional radiance errors depend on valid sample coverage.
- All-strategy intersections may exclude difficult samples.
- Independent metric experiment reproduction remains unverified.
- Hardware and edge performance are not established.

## Interpretation and change control

This protocol is established before the five-strategy
comparative experiment is executed.

The scenario was previously explored and is not an
independent generalization benchmark.

Any necessary protocol amendment must be documented before
the affected comparative execution.

If an amendment follows inspection of comparative results,
the affected analysis must be identified as exploratory.

No unfavorable results may be silently discarded.

## Source provenance

- environment: `src/ahds/module1_environment/optical_environment.py`
  - SHA256: `7d6eb60892cddf1ee571ff600b966587478b9ec64186dabebc49e9224cc6054c`
- sensor: `src/ahds/module2_sensor/sensor_simulator.py`
  - SHA256: `a01ad9fed04e4526ec234a96ea20360b782d3d394fb14fe3f2ce99004e0700be`
- assessor: `src/ahds/module4_anomaly_detection/disturbance_assessor.py`
  - SHA256: `a7e519481195136cfef32c35d82afaab1aa115024f62dd029bcc4e7431c37b29`
- tracker: `src/ahds/module4_anomaly_detection/disturbance_state_tracker.py`
  - SHA256: `5de9d2e4b1c07fc97de97510d1da34abcf126c8829cc4743162390205f0a163e`
- S1: `src/ahds/module5_defense/exposure_protection.py`
  - SHA256: `a45f7e1aa97d8e5eaf2598dbbc781a4b87221eb44196a60148d36548201d4a24`
- S2: `src/ahds/module5_defense/probed_exposure.py`
  - SHA256: `8d87f15b2a28ee7d467eed60b34b748cd555e7f0d09ea28067b07149f9430a80`
- S3: `src/ahds/module5_defense/state_aware_exposure.py`
  - SHA256: `3c144e7e1087b1bbd63539da3a228f380449b2f5ce0d41d1d5df93ac5b30dbab`
- S4: `src/ahds/module5_defense/disturbance_aware_defense.py`
  - SHA256: `c5f8d2203502dc8cc2175396c5335d4ced3dcaca04d7c08b8d8d99dcd0e2fd2b`
- metrics: `src/ahds/module7_evaluation/information_preservation.py`
  - SHA256: `c89ec27ac109a7552fd42675677c7ecf49e6c8809f14bd429568003ba87f5bfb`

## Current status

Protocol formalization and static consistency validation
are complete upon successful execution of this task.

Runtime harness validation and comparative experiments
remain pending.

No controller superiority, semantic perception advantage
or hardware robustness has been established.
