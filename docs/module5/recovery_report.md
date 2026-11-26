# AHDS Module 5 Recovery Report

## Recovery objective

Restore the defensive controller implementations and
reproduce the historical four-strategy closed-loop
baseline experiment without modifying historical results.

## Controller provenance

| Controller | Provenance |
|---|---|
| Exposure protection | Recovered original source |
| State-aware protection | Recovered original source |
| Probe-based protection | Reconstructed source |

The original probe-based controller source could not
be retrieved. Its implementation was reconstructed
using documented behavioral requirements.

## Behavioral validation

- Module 5 dedicated unit tests: 24 passed.
- Complete AHDS regression suite: 69 passed.
- Original historical artifacts preserved: 5/5.

## Historical baseline reproduction

- Strategies: 4.
- Frames per strategy: 30.
- Original measurements: 120.
- Reproduced measurements: 120.
- Non-key fields per frame: 12.
- Recorded values compared: 1440.
- Strategy-field checks: 48/48 passed.
- Mismatched values: 0.
- Absolute tolerance: 1e-8.
- Relative tolerance: 1e-6.

## Reproduction conditions

- Scene resolution: 256 x 256.
- Random seed: 42.
- Initial exposure: 10 ms.
- Fixed gain: 1.0.
- Laser intensity: 15.0 simulator units.
- Laser radius: 0.12 normalized units.
- Disturbance interval: frames 5 through 19.
- Sensor noise: disabled.
- Exposure command delay: one frame.

## Reproduction command

From the repository root:

```bash
PYTHONPATH="$PWD:$PWD/src" python experiments/module5/reproduce_closed_loop_baseline.py
```

Use the reproduction script rather than the preserved
original experiment, which writes to the historical
results directory.

The current experiment scripts assume the original
Google Colab repository path.

## Research limitations

Successful reproduction establishes behavioral
equivalence for this particular historical experiment
within the declared numerical tolerances.

It does not establish source-code identity or
equivalence under all possible optical disturbances.

The historical experiment uses one deterministic
laser disturbance, fixed gain, no sensor noise,
and no downstream perception evaluation.

The background brightness metric may include
laser spatial tails and does not independently
establish useful information preservation.

No calibrated physical laser power, hardware
energy, or hardware bandwidth claims are made.

## Next development phase

M5-DEF-02A: Disturbance-Aware Defensive
Reconfiguration Controller implementation.
