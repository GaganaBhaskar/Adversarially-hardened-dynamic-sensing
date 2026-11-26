# AHDS — M5-VAL-01B-04C

## Scientific Interpretation

Scope: one controlled deterministic synthetic scenario.

## Principal finding

DADRC maintains physical disturbance awareness after exposure reduction removes observable saturation and ADC clipping symptoms.

## Quantitative comparative findings

- censoring reduction vs S0 percent: 95.000000000
- censoring reduction vs S3 percent: 40.092165899
- reference mae reduction vs S0 percent: 35.337397245
- gradient mae reduction vs S0 percent: 27.059016159
- valid coverage gain vs S0 percentage points: 3.768920898
- quantization step increase vs S0 percent: 78.610995419
- conditional radiance mae increase vs S0 percent: 46.991140563
- matched radiance mae increase vs S0 percent: 0.008356562

## Matched-valid comparison

- Common valid coverage: 95.632324%
- Maximum absolute matched-valid MAE difference: 0.000103803165

## Hypothesis assessment

### H01

Status: supported_in_controlled_scenario

Adaptive defensive exposure reduces physical censoring relative to fixed exposure.

- S4 reduces mean censoring relative to S0.
- S4 clears censoring after one frame.
- No subsequent S4 threshold violations.

### H02

Status: supported_in_controlled_scenario

DADRC maintains disturbance awareness after observable saturation and ADC clipping disappear.

- 20 of 20 active frames recognized.
- 19 of 19 uncensored active frames recognized.
- No unknown evidence states.

### H03

Status: partially_supported_in_controlled_scenario

Protection improves valid coverage, reference distortion and structural distortion relative to fixed exposure, but introduces competing radiometric and quantization costs.

- Higher valid observation coverage.
- Lower reference and gradient MAE than S0.
- Coarser radiance quantization than S0.
- Higher conditional radiance MAE than S0.
- Nearly identical matched-valid errors.
- S2 and S4 tie on primary disturbance metrics.
- S1 has slightly lower gradient MAE than S4.

### H04

Status: supported_in_controlled_scenario

DADRC restores nominal sensing after disturbance removal under the frozen observed recovery criteria.

- Restoration at frame 35.
- Confirmation at frame 37.
- Five-frame recovery delay.
- Restoration sustained through frame 59.

### H05

Status: supported_in_controlled_scenario

Lower saturation does not guarantee improvement across all information metrics.

- Censoring decreases while quantization coarsens.
- Conditional radiance errors increase.
- Common-valid sample comparison removes differences in valid sample populations.
- No universal information superiority inferred.

## Supported claims

- Persistent disturbance awareness was demonstrated after physical censoring disappeared.
- DADRC combines the observed disturbance protection of S2 with the observed recovery performance of S3.
- DADRC increases valid observation coverage relative to fixed exposure in this scenario.
- Matched-valid radiance errors are numerically close across strategies on identical valid samples.
- Physical protection and information preservation are distinct, competing sensing objectives.

## Unsupported claims

- Universal DADRC superiority.
- Generalized perfect disturbance detection.
- Robustness across unseen optical scenarios.
- Robustness under realistic sensor noise.
- Improved downstream semantic perception.
- Verified embedded latency, energy or bandwidth gains.
- Novelty relative to all published sensing methods.
- Statistically significant population-level effects.

## Limitations

- Single deterministic synthetic static scene.
- Single controlled optical disturbance configuration.
- Sensor noise disabled.
- Fixed gain and ROI.
- Module 3 dynamic adaptation disabled.
- Trusted startup reference available.
- No calibrated optical power or real camera.
- No downstream perception evaluation.
- No embedded hardware profiling.
- Temporally dependent frames are not independent trials.
- S2 is a reconstructed behavioral comparator.
- Independent full numerical reproduction pending.

## Visualization review

All eleven figure screenshots were reviewed. No blocking scientific plotting errors were identified. Cosmetic publication refinements are deferred.

- F06: Annotation overlaps trajectories.
- F07: Coverage annotation overlaps bar tops.
- F09: Threshold label placement could improve.
- F10: Repeated shading could be consolidated.
- F11: Some strategy labels are closely spaced.

## Conclusion

The experiment supports the proposed physical-layer mechanism in the tested scenario. Generalized superiority is not established. Multi-scenario robustness, controller integration, downstream perception and embedded evaluation remain necessary.
