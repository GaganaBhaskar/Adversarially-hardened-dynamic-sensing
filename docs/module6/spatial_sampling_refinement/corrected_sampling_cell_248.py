
# =====================================================================
# AHDS — M6-DEV-01D-04B-01
# Fixed-Budget Spatial Sampling Experiment
# =====================================================================

from pathlib import Path
from collections import defaultdict
import hashlib
import subprocess
import json
import time
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

START = time.perf_counter()

ROOT = Path(
    "/content/adversarially-hardened-dynamic-sensing"
)

STATE = AHDS_M6_ABLATION_STATE
PROTECTED = dict(STATE["protected_files"])

OUTPUT = ROOT / "results/module6/spatial_sampling_refinement"

CHECKPOINT = "a9258ba4a307d6a6ad4ff23788a392c16260ed48"

MODEL_REL = "results/module6/model_training/best_model.pt"
MODEL_PATH = ROOT / MODEL_REL

CLASS_NAMES = (
    "background",
    "lane_markings",
    "vehicle",
    "pedestrian",
)

print("=" * 70)
print("AHDS — M6-DEV-01D-04B-01")
print("Fixed-Budget Spatial Sampling Experiment")
print("=" * 70)


# =====================================================================
# 1. PREREQUISITES
# =====================================================================

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protected():

    head = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    assert head == CHECKPOINT

    for name, digest in PROTECTED.items():
        assert sha256(ROOT / name) == digest, name


assert STATE["task_passed"]
assert len(PROTECTED) == 180
assert not OUTPUT.exists()

verify_protected()

assert sha256(MODEL_PATH) == PROTECTED[MODEL_REL]

print("\n[PREREQUISITES]")
print("Protected inventory: 180/180")
print("Research checkpoint: VERIFIED")
print("Frozen model checkpoint: VERIFIED")
print("Output overwrite protection: PASS")


# =====================================================================
# 2. VALIDATED MODULES
# =====================================================================

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ahds.module2_sensor.sensor_simulator import (
    SensorConfig,
    simulate_sensor,
)

from ahds.module3_dynamic_sensing.roi_allocator import (
    allocate_readout,
)

from ahds.integration.roi_closed_loop import (
    AdaptiveROIConfig,
)

from ahds.module6_perception.procedural_dataset import (
    ProceduralPerceptionDataset,
)

from ahds.module6_perception.lightweight_segmentation import (
    LightweightSegmentationNet,
)

from ahds.module6_perception.segmentation_metrics import (
    SegmentationMetrics,
)


# =====================================================================
# 3. FROZEN MODEL
# =====================================================================

DEVICE = torch.device("cpu")

MODEL = LightweightSegmentationNet(
    num_classes=4
).to(DEVICE)

checkpoint_data = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True,
)

MODEL.load_state_dict(
    checkpoint_data["model_state_dict"],
    strict=True,
)

MODEL.eval()
MODEL.requires_grad_(False)

assert checkpoint_data["epoch"] == 38

assert sum(
    p.numel() for p in MODEL.parameters()
) == 10700


def tensor_hash(tensor):

    array = tensor.detach().cpu().contiguous().numpy()

    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))

    return digest.hexdigest()


def model_hashes():

    return {
        name: tensor_hash(value)
        for name, value in MODEL.state_dict().items()
    }


INITIAL_MODEL_HASHES = model_hashes()


# =====================================================================
# 4. FIXED-BUDGET SAMPLING MASKS
# =====================================================================

HEIGHT = WIDTH = 256

ROI_CONFIG = AdaptiveROIConfig()
ROI = ROI_CONFIG.resolve(HEIGHT, WIDTH)

dummy = torch.zeros(3, HEIGHT, WIDTH)

baseline = allocate_readout(
    dummy,
    ROI,
    background_stride=ROI_CONFIG.background_stride,
)

RECTANGULAR_MASK = baseline["sampling_mask"].bool()

BUDGET = int(RECTANGULAR_MASK.sum())

SPARSE = torch.zeros(
    HEIGHT,
    WIDTH,
    dtype=torch.bool,
)

stride = ROI_CONFIG.background_stride
SPARSE[::stride, ::stride] = True

assert (RECTANGULAR_MASK[SPARSE]).all()

Y, X = np.mgrid[0:HEIGHT, 0:WIDTH]

XN = (X + 0.5) / WIDTH
YN = (Y + 0.5) / HEIGHT

# Candidate G1:
# Fixed road-corridor trapezoid.
#
# No scene metadata, labels or model predictions
# are used to construct the sampling mask.

road_width = (
    0.14
    + 0.26 * np.clip(
        (YN - 0.40) / 0.60,
        0,
        1,
    )
)

corridor_score = (
    1.0
    - np.abs(XN - 0.5) / road_width
)

corridor_score = np.where(
    YN >= 0.40,
    corridor_score,
    -10.0 - (0.40 - YN),
)

# Candidate G2:
# Road corridor with additional preference
# towards lateral lane boundaries.

normalized_distance = (
    np.abs(XN - 0.5) / road_width
)

boundary_score = (
    np.exp(
        -((normalized_distance - 0.85) / 0.30) ** 2
    )
    + 0.15 * np.exp(
        -(normalized_distance / 0.35) ** 2
    )
)

boundary_score = np.where(
    YN >= 0.40,
    boundary_score,
    -10.0 - (0.40 - YN),
)


def exact_budget_mask(score):

    sparse_flat = SPARSE.numpy().reshape(-1)

    available = np.flatnonzero(~sparse_flat)

    additional = BUDGET - int(sparse_flat.sum())

    assert additional > 0

    scores = score.reshape(-1)[available]

    # Stable sorting provides deterministic
    # tie-breaking by flattened pixel index.

    order = np.argsort(
        -scores,
        kind="stable",
    )

    selected = available[order[:additional]]

    mask = sparse_flat.copy()
    mask[selected] = True

    mask = torch.from_numpy(
        mask.reshape(HEIGHT, WIDTH)
    )

    assert int(mask.sum()) == BUDGET
    assert mask[SPARSE].all()

    return mask


MASKS = {
    "R0": RECTANGULAR_MASK,
    "G1": exact_budget_mask(corridor_score),
    "G2": exact_budget_mask(boundary_score),
}


# Recovery-specific geometry identity guard.
for _sb_name in ("R0", "G1", "G2"):
    _sb_actual = MASKS[_sb_name].detach().cpu().numpy()
    _sb_expected = AHDS_M6_SAMPLING_GEOMETRY_STATE["masks"][_sb_name]
    assert np.array_equal(_sb_actual, _sb_expected), _sb_name
print("Validated sampling mask identities: 3/3 EXACT MATCH")


assert not torch.equal(MASKS["R0"], MASKS["G1"])
assert not torch.equal(MASKS["R0"], MASKS["G2"])
assert not torch.equal(MASKS["G1"], MASKS["G2"])

FULL_MASK = torch.ones(
    HEIGHT,
    WIDTH,
    dtype=torch.bool,
)

ALL_MASKS = {
    "F0": FULL_MASK,
    **MASKS,
}

print("\n[SAMPLING MASKS]")

for name, mask in ALL_MASKS.items():

    print(
        f"{name} | Samples: {int(mask.sum())}"
        f" | Ratio: "
        f"{100 * mask.float().mean().item():.4f}%"
    )

print("Candidate budgets: EXACTLY MATCHED")
print("Ground-truth access for sampling: NONE")


# =====================================================================
# 5. VALIDATION DEVELOPMENT PROTOCOL
# =====================================================================

DATASET = ProceduralPerceptionDataset("validation")

REGISTRY = pd.read_csv(
    ROOT / "datasets/ahds_procedural_v1/validation_index.csv"
)

# 32 validation scenes not used in the preceding
# controlled acquisition ablation.
#
# They were available during model selection,
# so they are development data, not an independent
# confirmatory test set.

SCENE_INDICES = tuple(range(1, 160, 5))

assert len(SCENE_INDICES) == 32

assert not set(SCENE_INDICES).intersection(
    range(0, 160, 5)
)

CONFIGURATIONS = {
    "K1": (10.0, 1.0),
    "K3": (30.0, 3.0),
    "K5": (2.0, 3.0),
}

NOISE_CONDITIONS = (False, True)

EXPECTED = (
    len(SCENE_INDICES)
    * len(CONFIGURATIONS)
    * len(NOISE_CONDITIONS)
    * len(ALL_MASKS)
)

assert EXPECTED == 768

PROTOCOL = {
    "task_id": "M6-DEV-01D-04B-01",
    "research_checkpoint": CHECKPOINT,
    "model_checkpoint_sha256": sha256(MODEL_PATH),
    "dataset_split": "validation",
    "scene_indices": list(SCENE_INDICES),
    "configurations": CONFIGURATIONS,
    "noise_conditions": list(NOISE_CONDITIONS),
    "optical_condition": "clean",
    "sampling_modes": {
        "F0": "full_frame",
        "R0": "existing_rectangular_roi",
        "G1": "fixed_road_corridor",
        "G2": "lane_boundary_weighted_corridor",
    },
    "candidate_sample_budget": BUDGET,
    "background_stride": stride,
    "mask_hashes": {
        name: tensor_hash(mask)
        for name, mask in ALL_MASKS.items()
    },
    "noise_seed_rule": "scene_seed + 3000000",
    "expected_evaluations": EXPECTED,
    "primary_metric": "pooled_foreground_miou",
    "model_training": False,
    "controller_updates": False,
    "test_split_used": False,
}

OUTPUT.mkdir(parents=True, exist_ok=False)

(OUTPUT / "sampling_protocol.json").write_text(
    json.dumps(PROTOCOL, indent=2) + "\n"
)

print("\n[EXPERIMENT PROTOCOL]")
print("Validation scenes: 32")
print("Acquisition configurations: 3")
print("Noise conditions: 2")
print("Sampling modes: 4")
print("Sensor acquisitions: 192")
print("Perception evaluations:", EXPECTED)
print("Held-out test scenes used: 0")


# =====================================================================
# 6. METRIC UTILITIES
# =====================================================================

def confusion_matrix(prediction, target):

    encoded = (
        target.reshape(-1).long() * 4
        + prediction.reshape(-1).long()
    )

    return torch.bincount(
        encoded,
        minlength=16,
    ).reshape(4, 4).cpu().numpy()


def summarize(matrix):

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    tp = np.diag(matrix)

    union = (
        matrix.sum(axis=0)
        + matrix.sum(axis=1)
        - tp
    )

    iou = [
        float(tp[i] / union[i])
        if union[i] > 0 else None
        for i in range(4)
    ]

    return {
        "foreground_miou": float(np.mean([
            v for v in iou[1:]
            if v is not None
        ])),
        "pixel_accuracy": float(
            tp.sum() / matrix.sum()
        ),
        "class_iou": dict(
            zip(CLASS_NAMES, iou)
        ),
    }


# =====================================================================
# 7. MATCHED EXPERIMENT EXECUTION
# =====================================================================

RECORDS = []

POOLED = defaultdict(
    lambda: np.zeros((4, 4), dtype=np.int64)
)

ACQUISITIONS = 0
REGISTRY_CHECKS = 0
SAMPLING_CHECKS = 0
METRIC_CHECKS = 0

EXAMPLES = {}

print("\n[SPATIAL SAMPLING EXECUTION]")

with torch.inference_mode():

    for number, index in enumerate(
        SCENE_INDICES,
        start=1,
    ):

        scene, target, metadata = (
            DATASET.optical_sample(index)
        )

        seed = DATASET.scene_seed(index)
        row = REGISTRY.iloc[index]

        assert int(row.scene_seed) == seed

        assert tensor_hash(scene) == (
            row.optical_scene_sha256
        )

        assert tensor_hash(target) == (
            row.semantic_target_sha256
        )

        REGISTRY_CHECKS += 2

        for config_id, (
            exposure,
            gain,
        ) in CONFIGURATIONS.items():

            for noise in NOISE_CONDITIONS:

                sensor = simulate_sensor(
                    scene,
                    config=SensorConfig(
                        exposure_ms=exposure,
                        gain=gain,
                    ),
                    add_noise=noise,
                    seed=seed + 3000000,
                )

                ACQUISITIONS += 1

                rgb = sensor["normalized_output"]

                inputs = []

                for mode, mask in ALL_MASKS.items():

                    sampled = rgb * mask.unsqueeze(0)

                    assert torch.equal(
                        sampled[:, mask],
                        rgb[:, mask],
                    )

                    assert (
                        sampled[:, ~mask] == 0
                    ).all()

                    expected_budget = (
                        HEIGHT * WIDTH
                        if mode == "F0"
                        else BUDGET
                    )

                    assert int(mask.sum()) == (
                        expected_budget
                    )

                    SAMPLING_CHECKS += 3

                    inputs.append(
                        torch.cat([
                            sampled,
                            mask.float().unsqueeze(0),
                        ], dim=0)
                    )

                batch = torch.stack(inputs).to(DEVICE)

                logits = MODEL(batch)

                assert logits.shape == (
                    4, 4, HEIGHT, WIDTH
                )

                assert torch.isfinite(logits).all()

                predictions = logits.argmax(
                    dim=1
                ).cpu()

                for mode, prediction in zip(
                    ALL_MASKS,
                    predictions,
                ):

                    matrix = confusion_matrix(
                        prediction,
                        target,
                    )

                    metrics = summarize(matrix)

                    if METRIC_CHECKS < 4:

                        reference = SegmentationMetrics(
                            num_classes=4
                        )

                        reference.update(
                            prediction,
                            target,
                        )

                        result = reference.compute()

                        assert abs(
                            result["foreground_miou"]
                            - metrics["foreground_miou"]
                        ) < 1e-7

                        METRIC_CHECKS += 1

                    POOLED[
                        (config_id, noise, mode)
                    ] += matrix

                    record = {
                        "scene_index": index,
                        "scene_seed": seed,
                        "configuration": config_id,
                        "exposure_ms": exposure,
                        "gain": gain,
                        "noise_enabled": noise,
                        "sampling_mode": mode,
                        "sample_count": int(
                            ALL_MASKS[mode].sum()
                        ),
                        "sampling_ratio": float(
                            ALL_MASKS[mode].float().mean()
                        ),
                        "foreground_miou": (
                            metrics["foreground_miou"]
                        ),
                        "pixel_accuracy": (
                            metrics["pixel_accuracy"]
                        ),
                    }

                    for name in CLASS_NAMES:

                        record[f"iou_{name}"] = (
                            metrics["class_iou"][name]
                        )

                    RECORDS.append(record)

                if (
                    number in (1, 11, 21, 32)
                    and config_id == "K1"
                    and not noise
                ):

                    EXAMPLES[seed] = {
                        "rgb": rgb.clone(),
                        "target": target.clone(),
                        "predictions": {
                            mode: prediction.clone()
                            for mode, prediction in zip(
                                ALL_MASKS,
                                predictions,
                            )
                        },
                    }

        if number % 4 == 0:

            print(
                f"  Scenes: {number}/32"
                f" | Evaluations: "
                f"{len(RECORDS)}/{EXPECTED}",
                flush=True,
            )


# =====================================================================
# 8. EXECUTION INTEGRITY
# =====================================================================

assert len(RECORDS) == EXPECTED
assert ACQUISITIONS == 192
assert REGISTRY_CHECKS == 64
assert SAMPLING_CHECKS == 2304
assert METRIC_CHECKS == 4

assert model_hashes() == INITIAL_MODEL_HASHES
assert not MODEL.training

verify_protected()

DF = pd.DataFrame(RECORDS)

assert not DF.duplicated([
    "scene_seed",
    "configuration",
    "noise_enabled",
    "sampling_mode",
]).any()

DF.to_csv(
    OUTPUT / "sampling_records.csv",
    index=False,
)

print("\n[EXECUTION INTEGRITY]")
print("Perception evaluations: 768/768")
print("Registry checks: 64/64")
print("Sampling checks: 2304/2304")
print("Metric equivalence checks: 4/4")
print("Frozen model state: UNCHANGED")


# =====================================================================
# 9. CONFIGURATION PERFORMANCE
# =====================================================================

SUMMARY_ROWS = []

for key, matrix in sorted(POOLED.items()):

    config_id, noise, mode = key

    assert matrix.sum() == (
        32 * HEIGHT * WIDTH
    )

    metrics = summarize(matrix)

    row = {
        "configuration": config_id,
        "noise_enabled": noise,
        "sampling_mode": mode,
        "scenes": 32,
        "foreground_miou": metrics["foreground_miou"],
        "pixel_accuracy": metrics["pixel_accuracy"],
        "sampling_ratio": float(
            ALL_MASKS[mode].float().mean()
        ),
    }

    for name in CLASS_NAMES:

        row[f"iou_{name}"] = (
            metrics["class_iou"][name]
        )

    SUMMARY_ROWS.append(row)

SUMMARY = pd.DataFrame(SUMMARY_ROWS)

assert len(SUMMARY) == 24

SUMMARY.to_csv(
    OUTPUT / "sampling_summary.csv",
    index=False,
)

print("\n[CONFIGURATION PERFORMANCE]")

for config_id in CONFIGURATIONS:

    for noise in NOISE_CONDITIONS:

        group = SUMMARY[
            (SUMMARY.configuration == config_id)
            & (SUMMARY.noise_enabled == noise)
        ].set_index("sampling_mode")

        print(
            f"\n{config_id} | Noise: {noise}"
        )

        for mode in ALL_MASKS:

            value = group.loc[
                mode, "foreground_miou"
            ]

            delta = 100 * (
                value
                - group.loc["R0", "foreground_miou"]
            )

            print(
                f"  {mode}"
                f" | FG mIoU: {100 * value:.4f}%"
                f" | vs R0: {delta:+.4f} pp"
            )


# =====================================================================
# 10. PAIRED SCENE COMPARISON
# =====================================================================

PAIRED = DF.pivot(
    index=[
        "scene_seed",
        "configuration",
        "noise_enabled",
    ],
    columns="sampling_mode",
    values="foreground_miou",
).reset_index()

PAIRED.columns.name = None

assert len(PAIRED) == 192

for mode in ("G1", "G2"):

    PAIRED[f"{mode}_minus_R0_pp"] = (
        100 * (PAIRED[mode] - PAIRED["R0"])
    )

PAIRED.to_csv(
    OUTPUT / "paired_sampling_comparison.csv",
    index=False,
)

CANDIDATE_ROWS = []

print("\n[CANDIDATE COMPARISON]")

for mode in ("G1", "G2"):

    delta = PAIRED[f"{mode}_minus_R0_pp"]

    row = {
        "candidate": mode,
        "mean_paired_improvement_pp": float(
            delta.mean()
        ),
        "median_paired_improvement_pp": float(
            delta.median()
        ),
        "minimum_improvement_pp": float(
            delta.min()
        ),
        "maximum_improvement_pp": float(
            delta.max()
        ),
        "better": int((delta > 1e-6).sum()),
        "equal": int((delta.abs() <= 1e-6).sum()),
        "worse": int((delta < -1e-6).sum()),
    }

    CANDIDATE_ROWS.append(row)

    print(
        f"{mode}"
        f" | Mean: {delta.mean():+.4f} pp"
        f" | Median: {delta.median():+.4f} pp"
        f" | Better/equal/worse: "
        f"{row['better']}/"
        f"{row['equal']}/"
        f"{row['worse']}"
    )

CANDIDATES = pd.DataFrame(CANDIDATE_ROWS)

CANDIDATES.to_csv(
    OUTPUT / "candidate_comparison.csv",
    index=False,
)


# =====================================================================
# 11. CLASS-WISE TRADE-OFFS
# =====================================================================

CLASS_ROWS = []

print("\n[CLASS-WISE TRADE-OFFS]")

for mode in ("G1", "G2"):

    for name in CLASS_NAMES[1:]:

        reference = SUMMARY[
            SUMMARY.sampling_mode == "R0"
        ].set_index([
            "configuration",
            "noise_enabled",
        ])[f"iou_{name}"]

        candidate = SUMMARY[
            SUMMARY.sampling_mode == mode
        ].set_index([
            "configuration",
            "noise_enabled",
        ])[f"iou_{name}"]

        delta = 100 * (candidate - reference)

        CLASS_ROWS.append({
            "candidate": mode,
            "class_name": name,
            "mean_group_iou_change_pp": float(
                delta.mean()
            ),
            "worst_group_iou_change_pp": float(
                delta.min()
            ),
        })

        print(
            f"{mode} | {name:15s}"
            f" | Mean: {delta.mean():+.4f} pp"
            f" | Worst: {delta.min():+.4f} pp"
        )

CLASS_COMPARISON = pd.DataFrame(CLASS_ROWS)

CLASS_COMPARISON.to_csv(
    OUTPUT / "class_sampling_comparison.csv",
    index=False,
)


# =====================================================================
# 12. RESEARCH FIGURES
# =====================================================================

FIGURES = OUTPUT / "figures"
FIGURES.mkdir(exist_ok=False)


def save_figure(fig, name):

    fig.tight_layout()

    fig.savefig(
        FIGURES / f"{name}.png",
        dpi=180,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURES / f"{name}.pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# M6-F21 — Exact-budget sampling masks

fig, axes = plt.subplots(
    1, 3,
    figsize=(12, 4),
)

for ax, (mode, mask) in zip(
    axes,
    MASKS.items(),
):

    ax.imshow(
        mask.numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    ax.set_title(
        f"{mode} — {int(mask.sum())} samples"
    )

    ax.axis("off")

fig.suptitle(
    "M6-F21 — Exact-budget sampling geometries"
)

save_figure(
    fig,
    "M6_F21_sampling_geometries",
)


# M6-F22 — Sampling performance

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
    sharey=True,
)

x = np.arange(len(CONFIGURATIONS))

for ax, noise in zip(
    axes,
    NOISE_CONDITIONS,
):

    for i, mode in enumerate(ALL_MASKS):

        values = []

        for config_id in CONFIGURATIONS:

            row = SUMMARY[
                (SUMMARY.configuration == config_id)
                & (SUMMARY.noise_enabled == noise)
                & (SUMMARY.sampling_mode == mode)
            ].iloc[0]

            values.append(
                100 * row.foreground_miou
            )

        ax.bar(
            x + (i - 1.5) * 0.20,
            values,
            width=0.20,
            label=mode,
        )

    ax.set_xticks(
        x,
        list(CONFIGURATIONS),
    )

    ax.set_title(
        "Sensor noise enabled"
        if noise else "Noise-free"
    )

    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.2)

axes[0].set_ylabel("Foreground mIoU (%)")
axes[0].legend()

fig.suptitle(
    "M6-F22 — Fixed-budget perception comparison"
)

save_figure(
    fig,
    "M6_F22_sampling_performance",
)


# M6-F23 — Paired improvements

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.boxplot(
    [
        PAIRED["G1_minus_R0_pp"],
        PAIRED["G2_minus_R0_pp"],
    ],
    tick_labels=["G1", "G2"],
    showmeans=True,
)

ax.axhline(
    0,
    linestyle="--",
    color="black",
)

ax.set_ylabel(
    "Candidate minus rectangular ROI mIoU (pp)"
)

ax.set_title(
    "M6-F23 — Paired spatial sampling effects"
)

ax.grid(axis="y", alpha=0.2)

save_figure(
    fig,
    "M6_F23_paired_sampling_effects",
)


# M6-F24 — Fixed qualitative examples

from matplotlib.colors import ListedColormap

cmap = ListedColormap([
    "#383838",
    "#facc15",
    "#2563eb",
    "#ef4444",
])

fig, axes = plt.subplots(
    len(EXAMPLES),
    6,
    figsize=(15, 10),
)

for r, (seed, example) in enumerate(
    EXAMPLES.items()
):

    rgb = example["rgb"].permute(
        1, 2, 0
    ).numpy()

    axes[r, 0].imshow(
        np.clip(rgb, 0, 1)
    )

    axes[r, 1].imshow(
        example["target"].numpy(),
        cmap=cmap,
        vmin=0,
        vmax=3,
    )

    for c, mode in enumerate(
        ALL_MASKS,
        start=2,
    ):

        axes[r, c].imshow(
            example["predictions"][mode].numpy(),
            cmap=cmap,
            vmin=0,
            vmax=3,
        )

    for ax in axes[r]:
        ax.axis("off")

    axes[r, 0].set_title(
        f"Scene {seed}"
    )

for c, name in enumerate(
    ["Sensor", "Ground truth", "F0", "R0", "G1", "G2"]
):

    if c > 0:
        axes[0, c].set_title(name)

fig.suptitle(
    "M6-F24 — Fixed validation predictions"
)

save_figure(
    fig,
    "M6_F24_sampling_predictions",
)


# =====================================================================
# 13. REPORT AND MANIFEST
# =====================================================================

def json_records(frame):

    return json.loads(
        frame.to_json(orient="records")
    )


REPORT = {
    "task_id": "M6-DEV-01D-04B-01",
    "task_passed": True,
    "research_checkpoint": CHECKPOINT,
    "validation_scenes": 32,
    "test_scenes_used": 0,
    "sensor_acquisitions": ACQUISITIONS,
    "perception_evaluations": len(DF),
    "sampling_checks": SAMPLING_CHECKS,
    "candidate_sample_budget": BUDGET,
    "candidate_sampling_ratio": (
        BUDGET / (HEIGHT * WIDTH)
    ),
    "configuration_summary": json_records(
        SUMMARY
    ),
    "candidate_comparison": json_records(
        CANDIDATES
    ),
    "class_comparison": json_records(
        CLASS_COMPARISON
    ),
    "candidate_selected": None,
    "model_state_unchanged": True,
    "model_training_performed": False,
    "controller_modified": False,
    "dataset_modified": False,
    "historical_modifications": False,
    "limitations": [
        "Synthetic validation development scenes.",
        "Fixed camera and road geometry.",
        "Candidate geometry informed by prior findings.",
        "Repeated observations of 32 unique scenes.",
        "No optical disturbance in this experiment.",
        "Post-ADC spatial sample selection.",
        "No physical bandwidth or energy measurement.",
        "No independent confirmatory test.",
        "No real-world validation.",
    ],
    "execution_seconds": (
        time.perf_counter() - START
    ),
}

REPORT_PATH = OUTPUT / "sampling_report.json"

REPORT_PATH.write_text(
    json.dumps(
        REPORT,
        indent=2,
        allow_nan=False,
    ) + "\n"
)

verify_protected()

assert model_hashes() == INITIAL_MODEL_HASHES

INVENTORY = {
    str(path.relative_to(ROOT)): {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    for path in sorted(OUTPUT.rglob("*"))
    if path.is_file()
}

MANIFEST_PATH = OUTPUT / "sampling_manifest.json"

MANIFEST_PATH.write_text(
    json.dumps({
        "task_id": "M6-DEV-01D-04B-01",
        "research_checkpoint": CHECKPOINT,
        "files": INVENTORY,
    }, indent=2) + "\n"
)

NEW_PROTECTED = {}

for name, metadata in INVENTORY.items():

    path = ROOT / name

    assert sha256(path) == metadata["sha256"]

    assert path.stat().st_size == metadata["bytes"]

    NEW_PROTECTED[name] = metadata["sha256"]

NEW_PROTECTED[
    str(MANIFEST_PATH.relative_to(ROOT))
] = sha256(MANIFEST_PATH)

assert len(NEW_PROTECTED) == 16

AHDS_M6_SAMPLING_REFINEMENT_STATE = {
    **REPORT,
    "report_path": str(
        REPORT_PATH.relative_to(ROOT)
    ),
    "manifest_path": str(
        MANIFEST_PATH.relative_to(ROOT)
    ),
    "manifest_sha256": sha256(MANIFEST_PATH),
    "protected_files": {
        **PROTECTED,
        **NEW_PROTECTED,
    },
    "next_task": (
        "M6-DEV-01D-04B-02 — "
        "Candidate Selection and Refinement Validation"
    ),
}


# =====================================================================
# 14. FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("FIXED-BUDGET SPATIAL SAMPLING SUMMARY")
print("=" * 70)

print("Validation scenes: 32")
print("Sensor acquisitions:", ACQUISITIONS)
print("Perception evaluations:", len(DF))
print("Candidate sample budget:", BUDGET)

print(
    "Ideal sample-count reduction:",
    f"{100 * (1 - BUDGET / (HEIGHT * WIDTH)):.4f}%"
)

print("\n[CANDIDATE RESULTS]")

print(
    CANDIDATES.to_string(
        index=False,
        float_format=lambda v: f"{v:.4f}",
    )
)

print("\n[ARTIFACT INTEGRITY]")
print("Historical files: 180/180")
print("New artifacts:", len(NEW_PROTECTED))
print("Frozen model state: VERIFIED")
print("Model training: NONE")
print("Controller modifications: NONE")
print("Dataset modifications: NONE")
print("Candidate selection: PENDING")
print("Artifact verification: PASS")
print("Git staging: NONE")

print(
    "Protected inventory:",
    len(
        AHDS_M6_SAMPLING_REFINEMENT_STATE[
            "protected_files"
        ]
    ),
)

print("\n[ARTIFACT IDENTITIES]")

for name, digest in NEW_PROTECTED.items():

    print(name)
    print("SHA256:", digest)

print(
    "\nExecution time:",
    f"{(time.perf_counter() - START) / 60:.2f} minutes",
)

print(
    "\nM6-DEV-01D-04B-01: "
    "SPATIAL SAMPLING EXPERIMENT PASSED"
)

print(
    "Next: M6-DEV-01D-04B-02 — "
    "Candidate Selection and Refinement Validation"
)
