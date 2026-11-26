# DADRC Validation and Experimental Reproduction

Documentation task: M5-DEF-02C-03

Parent checkpoint: `59b4ffb28c7c392d20d31836314ac6dae0985941`

## Formal software validation

| Test category | Passed | Expected |
|---|---:|---:|
| Targeted DADRC behavioral tests | 47 | 47 |
| Complete AHDS regression suite | 116 | 116 |

The complete regression suite includes the targeted DADRC
tests. These counts must not be added together.

## Physical observation validation

- Five controlled observation scenarios.
- Thirteen validation checks passed.
- Exposure and gain invariance verified under the
  implemented experimental criteria.
- Previously censored optical samples recovered: 7800.
- Censoring removed by reducing exposure from 10 ms to
  5 ms in the tested disturbance scenario.

Evidence directory:
`results/module5/exposure_aware_observation/`

## Persistent disturbance memory validation

- Twelve scenarios evaluated.
- All twelve expected evidence statuses matched.
- Nine validation checks passed.
- Persistent disturbance evidence remained PRESENT
  after exposure reduction eliminated censoring.
- Uniform brightness changes were classified CLEAR
  in the tested scenarios.
- Insufficient observation coverage produced UNKNOWN.

Evidence directory:
`results/module5/disturbance_memory/`

## DADRC closed-loop smoke experiment

| Metric | Observed value |
|---|---:|
| Total frames | 60 |
| Disturbance frames | 20 |
| Mean disturbance-period saturation | 0.198364258% |
| Minimum exposure | 5.0 ms |
| Final applied exposure | 10.0 ms |
| Final commanded exposure | 10.0 ms |
| Persistent unsaturated disturbance frames | 19 |
| Unsaturated frames with PRESENT evidence | 19 |
| Protective actions | 1 |
| Probe actions | 5 |
| Failed probes | 0 |
| Recovery actions | 3 |
| Restoration frame | 36 |
| Restoration delay | 6 frames |

All fifteen smoke-test validation checks passed.

The archived timeline demonstrates that the Module 4 tracker
can return to NORMAL while the simulated disturbance
remains active.

The DADRC optical memory retains PRESENT evidence and
the controller maintains protective behavior.

Evidence directory:
`results/module5/dadrc_smoke_test/`

## Independent experimental reproduction

| Verification category | Result |
|---|---:|
| Original notebook cells preserved | 4/4 |
| Standalone experiments executed | 3/3 |
| CSV values matched | 1472/1472 |
| JSON values matched | 166/166 |
| Total scientific values matched | 1638/1638 |
| Maximum absolute error | 0.0 |
| Scientific mismatches | 0 |
| Equivalent experimental artifacts | 10/10 |
| Byte-identical experimental artifacts | 9/10 |
| Manifest entries verified | 4/4 |
| Figure pixel equality | True |

Numerical tolerances:

- Absolute: 1e-08
- Relative: 1e-06

All compared scientific values matched with zero observed
numerical error.

The non-byte-identical artifact is the observation manifest.
Its differences are explained by the reproduction output
directory redirection.

Evidence directory:
`results/module5/dadrc_reproduction/`

## Interpretation and limitations

These results establish software validation and
reproducibility for the archived configurations.

They do not establish:

- Robustness across diverse optical disturbances.
- Robustness under sensor noise or changing scenes.
- Performance with contaminated startup references.
- Comparative superiority over existing controllers.
- Downstream perception improvement.
- Real sensor or embedded-device performance.
- Energy or bandwidth savings.

The smoke experiment contains no failed probes and no
UNKNOWN-evidence frames. Conditional assertions involving
these events are not independently exercised by that
experiment.

Targeted behavioral tests provide additional coverage.

The archived mean saturation metric is not equivalent
to preserved useful scene information.

Information-preservation metrics and controlled comparative
evaluation are required before making broader performance
claims.
