# AHDS Information-Preservation Validation Report

## 1. Research checkpoint

Parent Git checkpoint:

`837d98d0e42630c6736b472cea39833ff6a579a1`

Development phase: M5-VAL-01A.

The information-preservation implementation is an evaluation
component supporting defensive-controller comparison.

No new defensive controller or sensor-model modification
was required by this metric development phase.

## 2. Validation evidence

| Validation category | Result |
|---|---:|
| Implemented scalar metric fields | 42 |
| Controlled scenario evaluations | 16 |
| Controlled scientific checks passed | 180 |
| Controlled scientific checks failed | 0 |
| Per-sample quantization-bound checks passed | 8 |
| Targeted regression tests passed | 63 |
| Previous regression tests retained | 116 |
| Complete regression tests passed | 179 |
| Complete regression failures | 0 |
| Complete regression errors | 0 |
| Complete regression skipped tests | 0 |

The quantization-bound checks are part of the controlled
validation evidence. The 63 targeted regression cases are
also included in the 179-case complete regression suite.

These counts must not be added together and presented as
independent experimental trials.

## 3. Regression distribution

| Test category | Passed |
|---|---:|
| module1 | 9 |
| module2 | 10 |
| module3 | 16 |
| module4 | 10 |
| module5_baseline | 24 |
| module5_dadrc | 47 |
| module7_information_preservation | 63 |

## 4. Scientific interpretation

The controlled validation and regression evidence support
the implemented metric definitions and their behavior under
the tested operating conditions.

The tested properties include:

- Identical-observation distortion and ratio identities.
- Exposure reduction and gain compensation behavior.
- Radiance quantization relationships.
- Full-well saturation and independent ADC clipping.
- Censoring-mask union and observation coverage.
- Inclusion of uncensored zero ADC samples.
- Analytic radiance and gradient calculations.
- Distortion triangle inequality.
- Empty, single-pixel and disconnected region behavior.
- Invalid input rejection.
- Repeatable evaluation and input preservation.

These results establish metric implementation correctness
within the tested conditions, not comparative superiority
of the proposed defensive controller.

## 5. Regression evidence recovery

A repeated complete-regression execution attempt encountered
an existing-file protection assertion.

The previous complete regression evidence was preserved.

M5-VAL-01A-04C-R01 inspected all five existing evidence files
and found a previously successful 179-case regression run.

M5-VAL-01A-04C-R02 independently cross-checked evidence
hashes, source provenance, test identities, individual test
outcomes, XML accounting, terminal output and JSON reports.

The recovered evidence was accepted without rerunning the
regression suite.

The acceptance audit was read-only and did not create a
separate persistent audit certificate.

## 6. Evidence locations

Controlled scientific validation:

`results/module5/information_preservation_validation/`

Regression evidence:

`results/module5/information_preservation_regression/`

Metric implementation:

`src/ahds/module7_evaluation/information_preservation.py`

Controlled experiment:

`experiments/module5/validate_information_preservation.py`

Regression tests:

`tests/test_module7_information_preservation.py`

## 7. Source provenance

- metric_source_sha256: `c89ec27ac109a7552fd42675677c7ecf49e6c8809f14bd429568003ba87f5bfb`

- test_source_sha256: `5ef1fb924d799e4d153e465ef27a910f033fa330118957906e600b83f86a6a3f`

- controlled_experiment_sha256: `9b651c68acf745b1be0a09f9bd1e10ae6dbdb4a64e09383d33c227260800dc7b`


## 8. Remaining scientific work

The next major milestone is M5-VAL-01B, the five-strategy
closed-loop comparative evaluation.

The strategies are fixed exposure, exposure protection,
probed exposure, state-aware exposure and DADRC.

The comparison must use consistent initial conditions,
optical disturbance schedules, sensor configurations and
evaluation definitions.

The following remain pending:

- Predeclared spatial-region definitions.
- Matched-valid-region comparisons across strategies.
- Noise-aware robustness evaluation.
- Multi-scenario disturbance and recovery experiments.
- Dynamic sensing and defensive-controller integration.
- Downstream perception evaluation.
- Edge performance and resource profiling.

No comparative controller ranking is established by the
current metric-validation evidence.

## 9. Archival status

At the documentation checkpoint, 82 protected files have
been verified unchanged.

The regression phase is complete. Reproducibility
documentation, archival manifest creation, Git staging,
local commit, remote archival and final archival audit
remain pending.

