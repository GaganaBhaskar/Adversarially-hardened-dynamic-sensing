# AHDS Procedural Perception Dataset v1

This directory contains the reproducibility registry for the
original synthetic dataset developed for Adversarially-Hardened
Dynamic Sensing (AHDS).

## Dataset origin

The dataset is generated procedurally within the AHDS repository.
No external image dataset is used.

## Dataset composition

| Split | Scenes | Scene seeds |
|---|---:|---|
| Training | 800 | 100000–100799 |
| Validation | 160 | 200000–200159 |
| Test | 160 | 300000–300159 |
| Total | 1120 | Disjoint seed ranges |

## Registry files

- dataset_card.md: intended use and limitations.
- generation_config.json: generation and software provenance.
- train_index.csv: individual training sample records.
- validation_index.csv: individual validation sample records.
- test_index.csv: individual test sample records.
- class_distribution.csv: semantic class statistics.
- sample_visualization.png: validated training examples.
- dataset_manifest.json: registry artifact inventory.
- checksums.sha256: SHA256 file integrity records.

## Regenerate a sample

From the repository root, make the src directory importable:

```python
import sys
sys.path.insert(0, "src")

from ahds.module6_perception.procedural_dataset import (
    ProceduralPerceptionDataset,
)

dataset = ProceduralPerceptionDataset("train")

scene, target, metadata = dataset.optical_sample(0)
sample = dataset[0]

model_input = sample["input"]
semantic_target = sample["target"]

print(model_input.shape)
print(semantic_target.shape)
```

The generator creates optical radiance, semantic ground truth,
sensor observations and sampling masks on demand.

The individual sample indexes record the configuration and
content hashes of the registered observations.

Exact byte-level reproducibility may depend on the recorded
software versions and computational environment.

## Data separation

Training, validation and test scene seeds are disjoint.
Test samples must not be used for training, hyperparameter
selection or checkpoint selection.

## Storage policy

Raw image tensors are not duplicated in GitHub.
The source code, generation configuration, individual indexes
and content hashes provide the regeneration record.

The clean procedural registry is separate from subsequent
closed-loop optical disturbance evaluation records.
