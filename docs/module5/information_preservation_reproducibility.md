# AHDS Information-Preservation Reproducibility Guide

## 1. Scope

This guide accompanies the M5-VAL-01A information-preservation
metric development phase.

The metric evaluator belongs to Module 7 and supports
Module 5 defensive-controller comparison.

Parent research checkpoint:

`837d98d0e42630c6736b472cea39833ff6a579a1`

This checkpoint contains the previous DADRC archival package.
The information-preservation artifacts are prepared for a
subsequent commit.

## 2. Reproduction targets

| Evidence category | Expected result |
|---|---:|
| Scalar metric fields | 42 |
| Controlled scenario evaluations | 16 |
| Controlled scientific checks | 180 passed |
| Quantization-bound checks | 8 passed |
| Targeted metric regression | 63 passed |
| Previous AHDS regression tests | 116 passed |
| Complete AHDS regression | 179 passed |

The targeted metric tests are a subset of the complete
regression suite.

These counts describe software and controlled scientific
validation, not independent physical experiments.

## 3. Required source files

- `src/ahds/module7_evaluation/information_preservation.py`
- `experiments/module5/validate_information_preservation.py`
- `tests/test_module7_information_preservation.py`

The previously archived AHDS environment, sensor simulator,
dynamic sensing, anomaly detection and defensive-controller
sources must remain available.

## 4. Software environment

The current archival environment is recorded in:

`results/module5/information_preservation_regression/reproducibility_environment.json`

Current Python version: 3.13.18.

Current package versions:

| Package | Version |
|---|---|
| torch | 2.14.1+cpu |
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| pytest | 8.4.2 |

This snapshot records the environment available during
archival preparation. It is not a complete environment
lockfile or independent proof of historical environment
equivalence.

Cross-platform and cross-version numerical reproducibility
remain unverified.

## 5. Safe reproduction workflow

Perform reproduction only after the information-preservation
archive has been committed.

Use a disposable checkout or a separate clone of the final
archival commit.

Do not rerun the controlled experiment in the primary
research checkout containing the accepted historical evidence.

From the disposable repository root, verify the source
files and inspect the experiment's output paths and execution
interface before running it.

The controlled experiment entry file is:

```bash
experiments/module5/validate_information_preservation.py
```

Static source inspection found the following:

- Python syntax verification: passed.
- Main guard locations: [].
- Argparse import detected: False.

The presence or absence of argparse does not establish
the complete command-line interface.

If the experiment supports an output-directory argument,
select a new reproduction directory. Otherwise, use the
disposable checkout and follow the script's existing output
configuration.

The candidate direct execution command, subject to inspection
of its execution interface and output paths, is:

```bash
python experiments/module5/validate_information_preservation.py
```

If the script refuses to overwrite archived evidence,
preserve the original artifacts outside the reproduction
output directory in the disposable checkout before retrying.

Never remove overwrite protection from the primary research
checkout merely to make a reproduction run succeed.

This archival preparation task did not independently execute
the controlled experiment.

## 6. Targeted metric regression

From the disposable repository root:

```bash
python -m pytest -v tests/test_module7_information_preservation.py
```

Expected result:

```text
63 passed
```

Optional structured test evidence:

```bash
python -m pytest -v tests/test_module7_information_preservation.py \
    --junitxml=/tmp/ahds_metric_reproduction.xml
```

## 7. Complete AHDS regression

From the disposable repository root:

```bash
python -m pytest --collect-only -q tests
```

Expected collection:

```text
179 tests collected
```

Execute the complete suite:

```bash
python -m pytest -v tests \
    --junitxml=/tmp/ahds_complete_reproduction.xml
```

Expected result:

```text
179 passed
```

The expected counts apply to this archival checkpoint.
Future development may legitimately introduce more tests.

## 8. Accepted evidence locations

Controlled validation:

`results/module5/information_preservation_validation/`

Targeted and complete regression:

`results/module5/information_preservation_regression/`

Scientific documentation:

`docs/module5/information_preservation_design.md`

`docs/module5/information_preservation_validation_report.md`

The original complete regression execution was recovered
and accepted through read-only evidence inspection and
cross-verification.

No regression rerun was necessary for that acceptance.

## 9. Reproduction comparison requirements

A future independent reproduction should compare:

- Scenario identities and operating-point configurations.
- Metric field names and missing-value positions.
- Controlled scientific check outcomes.
- Numerical values under explicitly declared tolerances.
- Test node identities and pass/fail outcomes.
- Artifact schemas and expected output inventories.

Execution times, timestamps, environment metadata and
output-directory paths may legitimately differ.

Byte-identical artifacts should not be required where
legitimate execution metadata differs.

The current archival task does not establish independent
numerical reproduction of the controlled experiment.

## 10. Scientific safeguards

Reference observations, clean counterfactual observations,
clean optical ground truth and ground-truth region masks
are evaluation-only instruments.

They must not become privileged controller inputs during
comparative evaluation.

No final comparative region thresholds, arbitrary aggregate
scoring weights or controller rankings are established here.

Noise-aware robustness, downstream perception, hardware
calibration and edge profiling remain future work.

## 11. Archival integrity

The information-preservation archival manifest identifies
the files intended for the next Git commit.

Each manifested file is recorded with its relative path,
size, category and SHA-256 hash.

The manifest excludes itself to avoid circular hashing.

The previous DADRC archival package is protected separately
through its existing manifest and Git checkpoint.

A matching SHA-256 digest establishes byte-level identity
with the recorded artifact, not independent scientific
correctness.

Git staging, commit, remote push and final archival audit
remain pending at the time this guide is generated.

