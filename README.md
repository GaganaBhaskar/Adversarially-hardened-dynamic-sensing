
# AHDS: Adversarially-Hardened Dynamic Sensing

### Closed-Loop Physical-Layer Self-Defense for Adaptive Smart Sensors

AHDS is a research prototype investigating how adaptive sensing
systems can detect adversarial optical interference and autonomously
reconfigure sensor operation to preserve reliable downstream perception.

## Research Objective

Can an embedded defensive intelligence mechanism identify
physically implausible sensor measurements and autonomously
reconfigure a dynamic sensing system to preserve reliable perception
under adversarial optical interference?

## System Architecture

AHDS consists of seven interconnected modules:

1. Scene and Optical Environment
2. Physics-Informed Sensor Simulator
3. Dynamic Sensing Controller
4. Physical-Layer Anomaly Detection
5. Defensive Reconfiguration Controller
6. Downstream Perception
7. Evaluation and Edge Profiling

Modules 3, 4 and 5 form the central closed-loop defensive system.

## Implementation Progress

| Module | Description | Status |
|---|---|---|
| 1 | Scene and Optical Environment | Initial implementation; 4 tests passed |
| 2 | Physics-Informed Sensor Simulator | Planned |
| 3 | Dynamic Sensing Controller | Planned |
| 4 | Physical-Layer Anomaly Detection | Planned |
| 5 | Defensive Reconfiguration Controller | Planned |
| 6 | Downstream Perception | Planned |
| 7 | Evaluation and Edge Profiling | Planned |

## Module 1 Capabilities

- Synthetic road-scene generation
- Localized Gaussian optical interference
- Laser-like illumination
- Temporally modulated optical pulses
- Distributed glare simulation

## Technology Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Pytest

## Research Limitations

The initial scene generator is synthetic.
Its normalized illumination values are not calibrated physical radiance.

Optical interference is modeled as additional incident illumination.
Sensor response, noise, saturation and ADC behavior are reserved
for Module 2.

## Project Status

Research prototype under active development.
Experimental claims will be reported only after validation.


## Module 1.2 — Optical Scenario Extension

Implemented capabilities:

- Daylight, dusk, night and low-light conditions
- Legitimate localized bright-light simulation
- Multiple simultaneous optical attacks
- Temporal optical pulse sequences
- Automated validation: 9 passing tests

The current optical model is synthetic and uncalibrated.
Sensor-level effects will be implemented in Module 2.


## Module 2 — Physics-Informed Sensor Simulator

Implemented in:
`src/ahds/module2_sensor/sensor_simulator.py`

Capabilities:
- Exposure-dependent photoelectron generation
- Poisson shot noise and Gaussian read noise
- Full-well saturation and ADC clipping
- Configurable analog gain and ADC bit depth
- Deterministic experiments using random seeds

Validation: 10 Module 2 tests passed.
Total project tests: 19 passed.

The model is a simplified, uncalibrated RGB sensor simulation.


## Module 3 — Dynamic Sensing Controller

Implemented components:

- `exposure_controller.py`
- `joint_controller.py`
- `roi_allocator.py`

Capabilities:
- Adaptive exposure regulation
- Joint exposure and gain adjustment
- ADC clipping feedback
- ROI-prioritized sampling
- Sparse background readout

Validation:
- 16 Module 3 tests passed
- 35 total project tests passed

The current implementation is a conventional dynamic-sensing
baseline without adversarial anomaly detection or defense.
