
# AHDS Research Log

## Milestone 1: Initial Environment and Module 1

Date: 2026-11-23

### Completed

- Initialized the AHDS repository structure.
- Configured the Python and PyTorch environment.
- Implemented synthetic road-scene generation.
- Implemented Gaussian optical interference.
- Implemented laser, pulse and glare attack modes.
- Added initial automated tests.

### Validation

Test command:

pytest -v tests/test_module1.py

Result:

4 passed in 2.17 seconds.

### Limitations

- Scene generation is currently synthetic.
- Scene radiance is normalized and not physically calibrated.
- Optical interference has not been validated against real optical measurements.
- Temporal attack behavior requires additional experiments.

### Next Milestone

Extend Module 1 with illumination conditions, temporal
attack sequences, reproducible experiments and hard-negative
environmental lighting conditions.

## Module 1.2 — Optical Scenario Extension
Date: 2026-11-23

### Implemented
- Daylight, dusk, night and low-light illumination presets
- Legitimate localized bright-light generation
- Multiple simultaneous optical interference sources
- Temporal optical pulse sequences
- Additional automated tests

### Validation
- 9 tests passed in 1.75 seconds
- Temporal pulse periodicity verified
- Multiple optical attack locations visualized

### Limitations
- Synthetic, uncalibrated optical environment
- Gaussian approximation of localized illumination
- No camera sensor saturation or noise simulation yet
- No validation against real optical attack measurements

### Next milestone
Module 2: Physics-Informed Sensor Simulator


## Module 2 — Physics-Informed Sensor Simulator
Date: 2026-11-23

### Implemented
- Exposure-dependent photoelectron generation
- Poisson photon shot noise
- Gaussian electronic read noise
- Full-well saturation
- Simplified analog gain
- ADC quantization (configurable bit depth)
- Reproducible noise generation using random seeds

### Experimental observations
- Clean scene: 0% saturation at 10 ms exposure
- Attacked scene: approximately 2.75% saturation at 10 ms
- Attacked scene: approximately 10.17% saturation at 30 ms
- Simulated noise closely matched theoretical noise
- Same random seed reproduced identical measurements

### Validation
- 19 total automated tests passed
- 10 new Module 2 tests passed

### Limitations
- Optical inputs are not physically calibrated
- Simplified linear RGB sensor model
- No Bayer color filter array or spectral response
- No rolling shutter or optical point-spread model
- No real hardware measurements yet

### Next milestone
Module 3 — Dynamic Sensing Controller


## Module 3 — Dynamic Sensing Controller
Date: 2026-11-23

### Implemented
- Conventional adaptive exposure controller
- Joint exposure and gain controller
- Brightness-target regulation
- ADC clipping feedback
- ROI-prioritized sampling
- Configurable sparse background readout

### Experimental results
- Target brightness: 0.35
- Joint controller converged to approximately 0.35
- Final clean-scene gain: approximately 2.68x
- Final clean-scene exposure: 30 ms
- Clean-scene ADC clipping: 0%
- Sudden illumination transition peak clipping: 30.18%
- Recovery after bright transition: 6 frame intervals
- ROI sample retention: 18.42%
- ROI sample-count reduction: 81.58%

### Validation
- 16 Module 3 tests passed
- 35 total project tests passed

### Limitations
- Conventional reactive control, not adversarial defense
- Global exposure and gain only
- ROI allocation is simulated
- Sample-count reduction is not measured bandwidth saving
- No physical sensor hardware integration
- No automatic object-aware ROI selection

### Next milestone
Module 4 — Physical-Layer Anomaly Detection
