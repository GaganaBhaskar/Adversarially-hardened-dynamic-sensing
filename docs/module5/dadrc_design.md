# Disturbance-Aware Defensive Reconfiguration Controller

## Project

Adversarially-Hardened Dynamic Sensing (AHDS)

Module 5: Defensive Reconfiguration Controller

Documentation task: M5-DEF-02C-03

Parent checkpoint: `59b4ffb28c7c392d20d31836314ac6dae0985941`

## Research motivation

A physical optical disturbance can remain present after
sensor reconfiguration removes visible saturation.

A controller that interprets the disappearance of
saturation as evidence of disturbance removal may recover
prematurely.

DADRC investigates a physical-evidence-driven alternative
that distinguishes sensor safety from disturbance presence.

## Implemented components

### Exposure-aware physical observation

Source:
`src/ahds/module5_defense/exposure_aware_observation.py`

The observation estimator reconstructs exposure-normalized
optical measurements using sensor observations and
configuration information.

Clipped, censored and insufficient-signal measurements
are handled through observation validity information.

The estimator is evaluated for exposure and gain invariance
and quantization-aware reconstruction.

### Exposure-aware disturbance memory

Source:
`src/ahds/module5_defense/disturbance_memory.py`

The disturbance memory maintains a trusted optical reference.

Current valid optical observations are compared against
the reference with compensation for global brightness changes.

The evidence statuses are:

- CLEAR: insufficient positive residual evidence of disturbance.
- PRESENT: positive optical residual evidence of disturbance.
- UNKNOWN: reference or valid observation coverage is insufficient.

CLEAR is an estimator decision, not proof that no physical
disturbance exists.

### Defensive reconfiguration controller

Source:
`src/ahds/module5_defense/disturbance_aware_defense.py`

The controller combines physical observation information,
disturbance memory and Module 4 assessment information.

Its implemented behavior includes:

- Exposure reduction following unsafe observations.
- Protection during persistent disturbance evidence.
- Conservative exposure probing.
- Rollback following unsuccessful probes.
- Cooldown following probe failure.
- Conservative handling of UNKNOWN evidence.
- Clear-evidence confirmation before recovery.
- Recovery bounded by the pre-disturbance reference exposure.
- Restoration confirmation after observing the reference exposure.

## Default controller configuration

| Parameter | Default |
|---|---:|
| `min_exposure_ms` | 0.1 |
| `max_exposure_ms` | 30.0 |
| `saturation_trigger` | 0.005 |
| `reduction_factor` | 0.5 |
| `persistent_probe_factor` | 1.05 |
| `clear_probe_factor` | 1.1 |
| `recovery_factor` | 1.2 |
| `persistent_probe_interval` | 4 |
| `clear_confirmation_frames` | 3 |
| `failed_probe_cooldown` | 8 |

## Closed-loop timing

The sensor is simulated using the currently applied exposure.

The controller processes the resulting observation and
produces a command for the subsequent frame.

The disturbance ground truth is used for evaluation,
not as a controller input in the archived smoke experiment.

## Relationship to the AHDS architecture

The fixed seven-module architecture is retained:

1. Scene and Optical Environment.
2. Physics-Informed Sensor Simulator.
3. Dynamic Sensing Controller.
4. Physical-Layer Anomaly Detection.
5. Defensive Reconfiguration Controller.
6. Downstream Perception.
7. Evaluation and Edge Profiling.

The current DADRC smoke experiment validates a Module 5
feedback subsystem using Modules 1, 2 and 4.

Full arbitration and integration with Module 3 remain pending.

## Current research scope

The implementation provides evidence of persistent
disturbance awareness during exposure reconfiguration
under controlled simulation conditions.

The current experiments do not establish general robustness,
comparative superiority, literature-level novelty or
real-hardware effectiveness.
