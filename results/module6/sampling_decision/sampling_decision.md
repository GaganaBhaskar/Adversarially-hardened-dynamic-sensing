# Module 6 — Final Spatial Sampling Decision

## Decision

Retain R0, the existing rectangular ROI with sparse
background sampling.

Reject G1 and G2. Neither candidate improved any
of the 192 matched observations.

## Experimental evidence

- Unique validation scenes: 32
- Acquisition configurations: 3
- Noise conditions: 2
- Sensor acquisitions: 192
- Perception evaluations: 768
- Equal candidate sample budget: 26,214 pixels
- Ideal sample-count reduction: 60.0006%

## Mean paired foreground mIoU change

| Candidate | Change relative to R0 |
|---|---:|
| G1 | -13.5373 percentage points |
| G2 | -35.6005 percentage points |

## Interpretation

Intuitive spatial prioritization did not improve
semantic information preservation under the
evaluated frozen perception model.

The result is conditional on the synthetic scene
distribution, acquisition conditions and model
training distribution. The model was trained using
full-frame and rectangular ROI observations, so
distribution shift may explain part of the loss.

No universal superiority of rectangular sampling
is claimed.

## Research disposition

Preserve both negative candidate results.
Retain the existing validated ROI controller.
Do not retrain the perception model or tune further
geometries in the current development phase.

The central AHDS physical-layer defensive mechanism
remains unchanged.

The sample-count reduction is theoretical.
Physical bandwidth, energy and real-world sensing
performance remain to be validated.
