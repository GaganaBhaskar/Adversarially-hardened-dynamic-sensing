# DADRC Reproducibility Guide

Documentation task: M5-DEF-02C-03

Parent checkpoint: `59b4ffb28c7c392d20d31836314ac6dae0985941`

## Recorded execution environment

Python:

`3.13.18 (main, Oct 26 2026, 15:28:15) [GCC 13.3.0]`

Platform:

`Linux-6.6.159+-x86_64-with-glibc2.39`

| Package | Recorded version |
|---|---|
| torch | 2.14.1+cpu |
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| pytest | 8.4.2 |

The archived results were reproduced using independent
Python processes in the recorded environment.

Cross-platform reproducibility has not been established.

## Source organization

Original notebook cells:

`experiments/module5/original_notebook_cells/`

These files preserve the exact original experiment sources.

Do not execute the original notebook archival copies
directly. They retain historical output paths.

Standalone executable experiments:

- `experiments/module5/validate_exposure_aware_observation.py`
- `experiments/module5/validate_disturbance_memory.py`
- `experiments/module5/validate_dadrc_closed_loop.py`

Conversion evidence:

- `experiments/module5/standalone_conversion_report.json`
- `experiments/module5/conversion_diffs/`

The conversion preserves the scientific calculations
while adapting repository paths, output paths, display
operations and plotting behavior.

## Reproduction procedure

Use a separate disposable checkout or working copy.

Install dependencies compatible with the recorded
environment and execute from the repository root.

First run the formal regression suite:

```bash
python -m pytest tests/ -q
```

Then execute the standalone experiments:

```bash
python experiments/module5/validate_exposure_aware_observation.py
python experiments/module5/validate_disturbance_memory.py
python experiments/module5/validate_dadrc_closed_loop.py
```

The standalone scripts write to:

`results/module5/dadrc_reproduction/`

Before rerunning, preserve or move any existing reproduction
output directories in the disposable working copy.

The standalone scripts themselves do not guarantee
protection against overwriting existing reproduction files.

Do not overwrite the permanent archived evidence.

## Expected results

- Complete regression suite: 116 passed at this checkpoint.
- Observation validation: 13 checks passed.
- Disturbance memory validation: 9 checks passed.
- DADRC smoke test: 15 checks passed.
- Scientific values reproduced: 1638/1638.
- Scientific mismatches: 0.
- Equivalent artifacts: 10/10.
- Byte-identical artifacts: 9/10.

Exact results may depend on software versions,
hardware and numerical execution behavior.

## Archived verification evidence

The reproduction directory contains:

- Independent execution logs.
- Recorded software environment.
- Independent execution report.
- Numerical comparison tables.
- Scientific mismatch report.
- Figure verification report.
- Manifest semantic verification.
- Final reproduction certificate.
- Reproduction evidence integrity manifest.

Final certificate:

`results/module5/dadrc_reproduction/final_verification/dadrc_reproduction_certificate.json`

The certificate is an internal research record,
not an external certification.

## Integrity verification

The Module 5 archival manifest records repository-relative
paths, file sizes and SHA256 hashes.

The archival manifest excludes itself to avoid
a recursive self-hash.

## Subsequent research tasks

1. Define information-preservation metrics.
2. Compare five controller strategies under matched conditions.
3. Evaluate multi-scenario robustness and recovery.
4. Integrate defensive reconfiguration with Module 3.
5. Implement downstream perception in Module 6.
6. Perform end-to-end evaluation and edge profiling in Module 7.
