# Dataset Card — AHDS Procedural Perception v1

## Dataset description

A synthetic road-scene benchmark for evaluating downstream
semantic perception in adaptive optical sensing systems.

The dataset contains 1120 procedurally generated scenes at
256 x 256 spatial resolution.

## Semantic classes

0. Background
1. Lane markings and road boundaries
2. Vehicle
3. Pedestrian

## Input representation

Four channels:
- Red sensor readout
- Green sensor readout
- Blue sensor readout
- Binary spatial sampling mask

## Ground truth

Visible-surface semantic labels are generated directly from
scene geometry before sensor capture.

Noise, exposure, gain and missing sensor samples do not alter
the semantic ground truth.

## Acquisition variation

The dataset includes deterministic per-scene variation in
exposure, gain, sensor noise and ROI-priority sampling.

No optical attack augmentation is used in this clean dataset.

## Intended use

- Lightweight semantic segmentation research.
- Evaluation of perception under adaptive sensing.
- Controlled comparison of static, unprotected dynamic
  and defensively reconfigured sensing systems.
- Research reproducibility and edge-oriented evaluation.

## Known limitations

- Synthetic road geometry and object appearance.
- Severe foreground/background class imbalance.
- Uncalibrated optical radiance.
- Fixed scene-prior ROI rather than learned ROI localization.
- Deterministic acquisition augmentation per scene.
- No demonstrated real-world autonomous-driving generalization.
- No demonstrated physical camera or hardware bandwidth savings.
- Byte-level regeneration may vary across software environments.

## Ethical and safety considerations

No real people, personal information or externally collected
driving images are included.

This dataset is intended for controlled research and is not
validated for safety-critical deployment.

## Evaluation policy

Foreground mean Intersection over Union and individual class
metrics should be reported alongside overall pixel accuracy.

Training and checkpoint selection must remain independent
of the test split.

All sensing strategies must be compared using the same frozen
perception model.

## Attribution

Original procedural dataset developed within the AHDS
research repository.

No external dataset license is asserted.
Distribution follows the applicable project licensing terms.
