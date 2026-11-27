# =====================================================================
# AHDS — M6-DEV-01B-02
# Reproducible Perception Model Training
# =====================================================================

from pathlib import Path
import hashlib
import subprocess
import importlib
import random
import json
import csv
import time
import sys
import os
import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

START_TIME = time.perf_counter()

ROOT = Path(
    "/content/adversarially-hardened-dynamic-sensing"
).resolve()

CHECKPOINT = "a9258ba4a307d6a6ad4ff23788a392c16260ed48"

STATE = AHDS_M6_DATASET_REGISTRY_STATE
MODEL_STATE = AHDS_M6_MODEL_STATE

PROTECTED = dict(STATE["protected_files"])

SEED = 42
MAX_EPOCHS = 40
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 8
NUM_CLASSES = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT = ROOT / "results/module6/model_training"

print("=" * 70)
print("AHDS — M6-DEV-01B-02")
print("Reproducible Perception Model Training")
print("=" * 70)


# =====================================================================
# 1. PREREQUISITE INTEGRITY
# =====================================================================

def sha256(path):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def tensor_sha256(tensor):

    array = (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
    )

    digest = hashlib.sha256()

    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))

    return digest.hexdigest()


def git(*args):

    return subprocess.check_output(
        ["git", "-C", str(ROOT), *args],
        text=True,
    ).strip()


def verify_protected():

    assert git("rev-parse", "HEAD") == CHECKPOINT

    assert not git("diff", "HEAD", "--name-status")

    assert not git("diff", "--cached", "--name-status")

    for name, digest in PROTECTED.items():

        assert sha256(ROOT / name) == digest, name


assert STATE["task_passed"]
assert MODEL_STATE["task_passed"]

assert STATE["research_checkpoint"] == CHECKPOINT
assert MODEL_STATE["research_checkpoint"] == CHECKPOINT

assert len(PROTECTED) == 105

verify_protected()

assert not OUTPUT.exists(), (
    f"Training output already exists: {OUTPUT}"
)

REGISTRY = ROOT / STATE["registry_path"]

assert sha256(
    REGISTRY / "dataset_manifest.json"
) == (
    '68f82a8a72a582440abf7874c950396cbcb822fb76399559c6adc6ed9c541df1'
)

print("\n[PREREQUISITES]")
print("Research checkpoint: VERIFIED")
print("Protected files:", len(PROTECTED))
print("Dataset registry: VERIFIED")
print("Model architecture: VERIFIED")
print("Output overwrite protection: PASS")


# =====================================================================
# 2. REPRODUCIBILITY CONFIGURATION
# =====================================================================

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.use_deterministic_algorithms(True)

if DEVICE.type == "cpu":

    torch.set_num_threads(
        min(4, os.cpu_count() or 1)
    )

print("\n[TRAINING ENVIRONMENT]")
print("Device:", DEVICE)

if DEVICE.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

print("PyTorch:", torch.__version__)
print("NumPy:", np.__version__)
print("CPU threads:", torch.get_num_threads())
print("Random seed:", SEED)
print("Deterministic algorithms: ENABLED")


# =====================================================================
# 3. IMPORT EXISTING VALIDATED IMPLEMENTATIONS
# =====================================================================

if str(ROOT / "src") not in sys.path:

    sys.path.insert(0, str(ROOT / "src"))

importlib.invalidate_caches()

from ahds.module6_perception.procedural_dataset import (
    CLASS_NAMES,
    ProceduralPerceptionDataset,
)

from ahds.module6_perception.lightweight_segmentation import (
    LightweightSegmentationNet,
)

from ahds.module6_perception.segmentation_loss import (
    SegmentationLoss,
)

assert len(CLASS_NAMES) == NUM_CLASSES

MODEL = LightweightSegmentationNet(
    num_classes=NUM_CLASSES
).to(DEVICE)

PARAMETERS = sum(
    parameter.numel()
    for parameter in MODEL.parameters()
    if parameter.requires_grad
)

assert PARAMETERS == 10700

print("\n[MODEL]")
print("Trainable parameters:", PARAMETERS)
print("Existing architecture: REUSED")


# =====================================================================
# 4. TRAINING-ONLY CLASS WEIGHTS
# =====================================================================

with (
    REGISTRY / "class_distribution.csv"
).open() as file:

    distribution = list(csv.DictReader(file))

training_distribution = sorted(
    [
        row for row in distribution
        if row["split"] == "train"
    ],
    key=lambda row: int(row["class_id"]),
)

assert len(training_distribution) == NUM_CLASSES

CLASS_COUNTS = torch.tensor(
    [
        int(row["pixel_count"])
        for row in training_distribution
    ],
    dtype=torch.float64,
)

assert CLASS_COUNTS.tolist() == [
    50239614,
    1082348,
    815541,
    291297,
]

CLASS_WEIGHTS = CLASS_COUNTS.rsqrt()

CLASS_WEIGHTS /= CLASS_WEIGHTS.mean()

CLASS_WEIGHTS = CLASS_WEIGHTS.float()

EXPECTED_WEIGHTS = torch.tensor([
    0.138915,
    0.946434,
    1.090312,
    1.824339,
])

assert torch.allclose(
    CLASS_WEIGHTS,
    EXPECTED_WEIGHTS,
    atol=1e-5,
    rtol=0,
)

CRITERION = SegmentationLoss(
    class_weights=CLASS_WEIGHTS.to(DEVICE),
    ce_weight=0.5,
    dice_weight=0.5,
).to(DEVICE)

print("\n[TRAINING CLASS WEIGHTS]")

for name, weight in zip(
    CLASS_NAMES,
    CLASS_WEIGHTS.tolist(),
):

    print(f"{name:16s}: {weight:.6f}")


# =====================================================================
# 5. CACHE AND VERIFY TRAINING / VALIDATION DATA
# =====================================================================

# The held-out test dataset is deliberately not instantiated.

CACHE_HASH_CHECKS = 0


def cache_dataset(split):

    global CACHE_HASH_CHECKS

    dataset = ProceduralPerceptionDataset(split)

    with (
        REGISTRY / f"{split}_index.csv"
    ).open() as file:

        registry_rows = list(
            csv.DictReader(file)
        )

    assert len(dataset) == len(registry_rows)

    inputs = torch.empty(
        (len(dataset), 4, 256, 256),
        dtype=torch.float32,
    )

    # Compact target storage in RAM.
    # Converted to int64 inside each training batch.

    targets = torch.empty(
        (len(dataset), 256, 256),
        dtype=torch.uint8,
    )

    for index in range(len(dataset)):

        sample = dataset[index]
        row = registry_rows[index]

        x = sample["input"]
        y = sample["target"]

        assert sample["scene_seed"] == int(
            row["scene_seed"]
        )

        assert x.shape == (4, 256, 256)
        assert y.shape == (256, 256)

        assert torch.isfinite(x).all()

        assert y.min().item() >= 0
        assert y.max().item() < NUM_CLASSES

        assert tensor_sha256(x) == (
            row["model_input_sha256"]
        )

        assert tensor_sha256(y) == (
            row["semantic_target_sha256"]
        )

        CACHE_HASH_CHECKS += 2

        inputs[index].copy_(x)

        targets[index].copy_(
            y.to(torch.uint8)
        )

        if (
            (index + 1) % 100 == 0
            or index + 1 == len(dataset)
        ):

            print(
                f"  {split}: "
                f"{index + 1}/{len(dataset)} cached"
            )

    return TensorDataset(inputs, targets)


print("\n[DATASET CACHE]")

CACHE_START = time.perf_counter()

TRAIN_DATA = cache_dataset("train")
VAL_DATA = cache_dataset("validation")

assert len(TRAIN_DATA) == 800
assert len(VAL_DATA) == 160
assert CACHE_HASH_CHECKS == 1920

CACHE_BYTES = sum(
    tensor.numel() * tensor.element_size()
    for dataset in (TRAIN_DATA, VAL_DATA)
    for tensor in dataset.tensors
)

print(
    "Cache storage:",
    f"{CACHE_BYTES / 1024**3:.3f} GiB",
)

print("Registry hash checks: 1920/1920")
print("Test observations loaded: 0")

print(
    "Cache preparation time:",
    f"{time.perf_counter() - CACHE_START:.2f}s",
)


# =====================================================================
# 6. DATA LOADERS AND OPTIMIZER
# =====================================================================

SHUFFLE_GENERATOR = torch.Generator()

SHUFFLE_GENERATOR.manual_seed(SEED)

TRAIN_LOADER = DataLoader(
    TRAIN_DATA,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=SHUFFLE_GENERATOR,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=False,
)

VAL_LOADER = DataLoader(
    VAL_DATA,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=False,
)

OPTIMIZER = torch.optim.AdamW(
    MODEL.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(
    OPTIMIZER,
    mode="max",
    factor=0.5,
    patience=3,
    min_lr=1e-5,
)

print("\n[TRAINING PROTOCOL]")
print("Training scenes:", len(TRAIN_DATA))
print("Validation scenes:", len(VAL_DATA))
print("Batch size:", BATCH_SIZE)
print("Training batches:", len(TRAIN_LOADER))
print("Maximum epochs:", MAX_EPOCHS)
print("Optimizer: AdamW")
print("Initial learning rate:", LEARNING_RATE)
print("Weight decay:", WEIGHT_DECAY)
print("Early stopping patience:", PATIENCE)
print("Checkpoint metric: foreground mIoU")
print("Optical disturbance augmentation: DISABLED")


# =====================================================================
# 7. LOSS AND EVALUATION HELPERS
# =====================================================================

def extract_loss(result):

    # Supports the validated loss returning either a scalar
    # tensor or a scalar tensor with diagnostic components.

    if torch.is_tensor(result):

        loss = result

    elif isinstance(result, dict):

        if "total_loss" in result:

            loss = result["total_loss"]

        elif "loss" in result:

            loss = result["loss"]

        else:

            raise KeyError(
                f"Unknown loss keys: {list(result)}"
            )

    elif isinstance(result, (tuple, list)):

        loss = result[0]

    else:

        raise TypeError(
            f"Unsupported loss output: {type(result)}"
        )

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    return loss


def update_confusion(confusion, logits, target):

    prediction = logits.argmax(dim=1)

    encoded = (
        target.reshape(-1) * NUM_CLASSES
        + prediction.reshape(-1)
    )

    batch_confusion = torch.bincount(
        encoded,
        minlength=NUM_CLASSES**2,
    ).reshape(NUM_CLASSES, NUM_CLASSES)

    confusion += batch_confusion.cpu()


def summarize_confusion(confusion):

    matrix = confusion.double()

    true_positive = matrix.diag()

    actual = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)

    union = actual + predicted - true_positive

    iou = torch.where(
        union > 0,
        true_positive / union.clamp_min(1),
        torch.nan,
    )

    dice_denominator = actual + predicted

    dice = torch.where(
        dice_denominator > 0,
        2 * true_positive / dice_denominator.clamp_min(1),
        torch.nan,
    )

    def nullable(value):

        value = float(value)

        return value if np.isfinite(value) else None

    return {
        "pixel_accuracy": float(
            true_positive.sum() / matrix.sum()
        ),
        "mean_iou": float(torch.nanmean(iou)),
        "foreground_miou": float(
            torch.nanmean(iou[1:])
        ),
        "foreground_mean_dice": float(
            torch.nanmean(dice[1:])
        ),
        "class_iou": {
            name: nullable(iou[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "class_dice": {
            name: nullable(dice[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": confusion.tolist(),
    }


# Independent metric sanity checks.

perfect = torch.diag(
    torch.tensor([100, 20, 10, 5])
)

assert summarize_confusion(
    perfect
)["foreground_miou"] == 1.0

background_only = torch.tensor([
    [100, 0, 0, 0],
    [20, 0, 0, 0],
    [10, 0, 0, 0],
    [5, 0, 0, 0],
])

assert summarize_confusion(
    background_only
)["foreground_miou"] == 0.0

print("\n[METRIC SANITY CHECKS]")
print("Perfect prediction: PASS")
print("Background-only prediction: PASS")


# =====================================================================
# 8. PREFLIGHT FORWARD / LOSS INTERFACE
# =====================================================================

MODEL.eval()

with torch.no_grad():

    x = TRAIN_DATA.tensors[0][:2].to(DEVICE)

    y = TRAIN_DATA.tensors[1][:2].long().to(DEVICE)

    logits = MODEL(x)

    assert logits.shape == (2, 4, 256, 256)

    result = CRITERION(logits, y)

    loss = extract_loss(result)

print("\n[TRAINING PREFLIGHT]")
print("Model forward: PASS")
print("Loss interface: PASS")
print("Initial diagnostic loss:", float(loss))

del x, y, logits, result, loss


# =====================================================================
# 9. OUTPUT DIRECTORY AND FROZEN TRAINING PROTOCOL
# =====================================================================

OUTPUT.mkdir(parents=True, exist_ok=False)

PROTOCOL = {
    "task_id": "M6-DEV-01B-02",
    "research_checkpoint": CHECKPOINT,
    "dataset_version": "ahds_procedural_v1",
    "dataset_manifest_sha256": STATE["manifest_sha256"],
    "random_seed": SEED,
    "training_scenes": 800,
    "validation_scenes": 160,
    "test_scenes_used": 0,
    "maximum_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "optimizer": "AdamW",
    "initial_learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "scheduler": {
        "name": "ReduceLROnPlateau",
        "mode": "max",
        "factor": 0.5,
        "patience": 3,
        "minimum_lr": 1e-5,
    },
    "early_stopping_patience": PATIENCE,
    "checkpoint_metric": "validation_foreground_miou",
    "checkpoint_tie_policy": "retain_earliest",
    "class_weights": CLASS_WEIGHTS.tolist(),
    "cross_entropy_weight": 0.5,
    "foreground_dice_weight": 0.5,
    "model_parameters": PARAMETERS,
    "optical_disturbance_augmentation": False,
    "mixed_precision": False,
    "deterministic_algorithms": True,
    "device": str(DEVICE),
    "pytorch_version": str(torch.__version__),
    "numpy_version": np.__version__,
    "cpu_threads": torch.get_num_threads(),
    "cache_hash_checks": CACHE_HASH_CHECKS,
    "gpu_name": (
        torch.cuda.get_device_name(0)
        if DEVICE.type == "cuda"
        else None
    ),
}

(OUTPUT / "training_protocol.json").write_text(
    json.dumps(
        PROTOCOL,
        indent=2,
        allow_nan=False,
    ) + "\n"
)


# =====================================================================
# 10. TRAINING AND VALIDATION FUNCTIONS
# =====================================================================

OPTIMIZER_STEPS = 0


def run_epoch(loader, training):

    global OPTIMIZER_STEPS

    MODEL.train(training)

    total_loss = 0.0
    total_samples = 0

    confusion = torch.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=torch.int64,
    )

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:

        for batch_index, (x, y) in enumerate(loader):

            x = x.to(
                DEVICE,
                non_blocking=(DEVICE.type == "cuda"),
            )

            y = y.to(
                DEVICE,
                dtype=torch.long,
                non_blocking=(DEVICE.type == "cuda"),
            )

            if training:

                OPTIMIZER.zero_grad(
                    set_to_none=True
                )

            logits = MODEL(x)

            loss = extract_loss(
                CRITERION(logits, y)
            )

            if training:

                loss.backward()

                for parameter in MODEL.parameters():

                    if parameter.grad is not None:

                        assert torch.isfinite(
                            parameter.grad
                        ).all()

                OPTIMIZER.step()

                OPTIMIZER_STEPS += 1

            update_confusion(
                confusion,
                logits.detach(),
                y,
            )

            total_loss += (
                float(loss.detach().item())
                * x.shape[0]
            )

            total_samples += x.shape[0]

            if training and (
                (batch_index + 1) % 25 == 0
            ):

                print(
                    f"    Batch {batch_index + 1:3d}"
                    f"/{len(loader)}"
                    f" | Loss: "
                    f"{total_loss / total_samples:.5f}",
                    flush=True,
                )

    assert total_samples == len(loader.dataset)

    assert int(confusion.sum()) == (
        total_samples * 256 * 256
    )

    metrics = summarize_confusion(confusion)

    metrics["loss"] = (
        total_loss / total_samples
    )

    metrics["samples"] = total_samples

    return metrics


# =====================================================================
# 11. REPRODUCIBLE MODEL TRAINING
# =====================================================================

HISTORY = []

BEST_SCORE = -float("inf")
BEST_EPOCH = None
BEST_VALIDATION = None

EPOCHS_WITHOUT_IMPROVEMENT = 0

TRAINING_START = time.perf_counter()

print("\n" + "=" * 70)
print("MODEL TRAINING")
print("=" * 70)

for epoch in range(1, MAX_EPOCHS + 1):

    EPOCH_START = time.perf_counter()

    lr = OPTIMIZER.param_groups[0]["lr"]

    print(
        f"\nEpoch {epoch:02d}/{MAX_EPOCHS}"
        f" | LR: {lr:.7f}",
        flush=True,
    )

    train_metrics = run_epoch(
        TRAIN_LOADER,
        training=True,
    )

    val_metrics = run_epoch(
        VAL_LOADER,
        training=False,
    )

    score = val_metrics["foreground_miou"]

    assert np.isfinite(score)

    improved = score > BEST_SCORE

    if improved:

        BEST_SCORE = score
        BEST_EPOCH = epoch

        BEST_VALIDATION = copy.deepcopy(
            val_metrics
        )

        EPOCHS_WITHOUT_IMPROVEMENT = 0

        checkpoint = {
            "model_state_dict": {
                name: value.detach().cpu().clone()
                for name, value
                in MODEL.state_dict().items()
            },
            "epoch": epoch,
            "validation_foreground_miou": score,
            "validation_metrics": BEST_VALIDATION,
            "model_parameters": PARAMETERS,
            "dataset_version": "ahds_procedural_v1",
            "class_names": list(CLASS_NAMES),
            "random_seed": SEED,
        }

        temporary_checkpoint = (
            OUTPUT / "best_model.tmp"
        )

        torch.save(
            checkpoint,
            temporary_checkpoint,
        )

        os.replace(
            temporary_checkpoint,
            OUTPUT / "best_model.pt",
        )

    else:

        EPOCHS_WITHOUT_IMPROVEMENT += 1

    SCHEDULER.step(score)

    record = {
        "epoch": epoch,
        "learning_rate": lr,
        "training": train_metrics,
        "validation": val_metrics,
        "best_checkpoint_updated": improved,
        "epochs_without_improvement": (
            EPOCHS_WITHOUT_IMPROVEMENT
        ),
        "optimizer_steps": OPTIMIZER_STEPS,
        "execution_seconds": (
            time.perf_counter() - EPOCH_START
        ),
    }

    HISTORY.append(record)

    # Preserve progress after every completed epoch.

    history_temp = OUTPUT / "training_history.tmp"

    history_temp.write_text(
        json.dumps(
            HISTORY,
            indent=2,
            allow_nan=False,
        ) + "\n"
    )

    os.replace(
        history_temp,
        OUTPUT / "training_history.json",
    )

    print(
        f"  Train loss: "
        f"{train_metrics['loss']:.5f}"
        f" | Val loss: "
        f"{val_metrics['loss']:.5f}"
    )

    print(
        f"  Train foreground mIoU: "
        f"{100 * train_metrics['foreground_miou']:.2f}%"
    )

    print(
        f"  Val foreground mIoU: "
        f"{100 * score:.2f}%"
    )

    print(
        f"  Val pixel accuracy: "
        f"{100 * val_metrics['pixel_accuracy']:.2f}%"
    )

    for name in CLASS_NAMES:

        value = val_metrics["class_iou"][name]

        formatted = (
            f"{100 * value:.2f}%"
            if value is not None
            else "undefined"
        )

        print(
            f"    {name:16s} IoU: {formatted}"
        )

    print(
        "  Best checkpoint:",
        f"Epoch {BEST_EPOCH}",
        f"| {100 * BEST_SCORE:.2f}%",
        "| UPDATED" if improved else "",
    )

    print(
        "  Epoch duration:",
        f"{record['execution_seconds']:.2f}s",
        flush=True,
    )

    if EPOCHS_WITHOUT_IMPROVEMENT >= PATIENCE:

        print("\nEarly stopping activated.")

        break


TRAINING_SECONDS = (
    time.perf_counter() - TRAINING_START
)

assert BEST_EPOCH is not None

assert OPTIMIZER_STEPS == (
    len(HISTORY) * len(TRAIN_LOADER)
)


# =====================================================================
# 12. RESTORE AND VERIFY BEST CHECKPOINT
# =====================================================================

print("\n[BEST CHECKPOINT VERIFICATION]")

BEST_CHECKPOINT = torch.load(
    OUTPUT / "best_model.pt",
    map_location="cpu",
    weights_only=True,
)

MODEL.load_state_dict(
    BEST_CHECKPOINT["model_state_dict"],
    strict=True,
)

MODEL.to(DEVICE)
MODEL.eval()

RELOADED_VALIDATION = run_epoch(
    VAL_LOADER,
    training=False,
)

assert BEST_CHECKPOINT["epoch"] == BEST_EPOCH

assert abs(
    RELOADED_VALIDATION["foreground_miou"]
    - BEST_SCORE
) < 1e-7

assert (
    RELOADED_VALIDATION["confusion_matrix"]
    == BEST_VALIDATION["confusion_matrix"]
)

assert OPTIMIZER_STEPS == (
    len(HISTORY) * len(TRAIN_LOADER)
)

for parameter in MODEL.parameters():

    assert torch.isfinite(parameter).all()

print("Best checkpoint reload: PASS")
print("Validation confusion matrix equivalence: PASS")
print("Validation foreground mIoU equivalence: PASS")
print("Finite trained parameters: PASS")
print("Test dataset used: NO")


# =====================================================================
# 13. TRAINING HISTORY CSV
# =====================================================================

CSV_ROWS = []

for entry in HISTORY:

    row = {
        "epoch": entry["epoch"],
        "learning_rate": entry["learning_rate"],
        "training_loss": entry["training"]["loss"],
        "validation_loss": entry["validation"]["loss"],
        "training_foreground_miou": (
            entry["training"]["foreground_miou"]
        ),
        "validation_foreground_miou": (
            entry["validation"]["foreground_miou"]
        ),
        "validation_pixel_accuracy": (
            entry["validation"]["pixel_accuracy"]
        ),
        "validation_mean_iou": (
            entry["validation"]["mean_iou"]
        ),
        "validation_foreground_mean_dice": (
            entry["validation"]["foreground_mean_dice"]
        ),
        "best_checkpoint_updated": (
            entry["best_checkpoint_updated"]
        ),
        "optimizer_steps": entry["optimizer_steps"],
        "execution_seconds": entry["execution_seconds"],
    }

    for name in CLASS_NAMES:

        row[f"validation_iou_{name}"] = (
            entry["validation"]["class_iou"][name]
        )

    CSV_ROWS.append(row)

with (
    OUTPUT / "training_history.csv"
).open("w", newline="") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=list(CSV_ROWS[0]),
    )

    writer.writeheader()
    writer.writerows(CSV_ROWS)


# =====================================================================
# 14. TRAINING VISUALIZATIONS
# =====================================================================

EPOCHS = [
    entry["epoch"]
    for entry in HISTORY
]

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


# M6-F01: Loss convergence.

fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(
    EPOCHS,
    [r["training"]["loss"] for r in HISTORY],
    label="Training",
)

ax.plot(
    EPOCHS,
    [r["validation"]["loss"] for r in HISTORY],
    label="Validation",
)

ax.axvline(
    BEST_EPOCH,
    linestyle="--",
    label="Selected checkpoint",
)

ax.set_title("M6-F01 — Segmentation loss convergence")
ax.set_xlabel("Epoch")
ax.set_ylabel("Combined loss")
ax.grid(alpha=0.2)
ax.legend()

save_figure(fig, "M6_F01_loss_convergence")


# M6-F02: Foreground segmentation quality.

fig, ax = plt.subplots(figsize=(9, 4))

ax.plot(
    EPOCHS,
    [
        100 * r["training"]["foreground_miou"]
        for r in HISTORY
    ],
    label="Training",
)

ax.plot(
    EPOCHS,
    [
        100 * r["validation"]["foreground_miou"]
        for r in HISTORY
    ],
    label="Validation",
)

ax.axvline(
    BEST_EPOCH,
    linestyle="--",
    label="Selected checkpoint",
)

ax.set_title("M6-F02 — Foreground segmentation quality")
ax.set_xlabel("Epoch")
ax.set_ylabel("Foreground mIoU (%)")
ax.set_ylim(0, 100)
ax.grid(alpha=0.2)
ax.legend()

save_figure(fig, "M6_F02_foreground_miou")


# M6-F03: Per-class validation performance.

fig, ax = plt.subplots(figsize=(9, 4))

for name in CLASS_NAMES:

    ax.plot(
        EPOCHS,
        [
            100 * r["validation"]["class_iou"][name]
            if r["validation"]["class_iou"][name] is not None
            else np.nan
            for r in HISTORY
        ],
        label=name,
    )

ax.axvline(
    BEST_EPOCH,
    linestyle="--",
    color="gray",
)

ax.set_title("M6-F03 — Validation class IoU")
ax.set_xlabel("Epoch")
ax.set_ylabel("IoU (%)")
ax.set_ylim(0, 100)
ax.grid(alpha=0.2)
ax.legend()

save_figure(fig, "M6_F03_class_iou")


# M6-F04: Validation prediction examples.
# Fixed sample indices, not selected by prediction quality.

PALETTE = np.array([
    [60, 60, 60],
    [255, 205, 0],
    [37, 99, 235],
    [239, 68, 68],
], dtype=np.uint8)

EXAMPLE_INDICES = [0, 40, 80, 120]

fig, axes = plt.subplots(
    len(EXAMPLE_INDICES),
    4,
    figsize=(12, 12),
)

with torch.no_grad():

    for row_index, index in enumerate(EXAMPLE_INDICES):

        x, y = VAL_DATA[index]

        prediction = MODEL(
            x.unsqueeze(0).to(DEVICE)
        ).argmax(dim=1)[0].cpu()

        rgb = x[:3].permute(1, 2, 0).numpy()

        # Display-only normalization.
        # The model receives the original sensor values.

        display_scale = max(
            float(rgb.max()),
            1e-8,
        )

        rgb_display = np.clip(
            rgb / display_scale,
            0,
            1,
        )

        axes[row_index, 0].imshow(rgb_display)

        axes[row_index, 1].imshow(
            x[3],
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[row_index, 2].imshow(
            PALETTE[y.numpy()]
        )

        axes[row_index, 3].imshow(
            PALETTE[prediction.numpy()]
        )

        titles = [
            f"Readout {index}\n(display scaled)",
            "Sampling mask",
            "Ground truth",
            "Prediction",
        ]

        for column, title in enumerate(titles):

            axes[row_index, column].set_title(title)

            axes[row_index, column].axis("off")

fig.suptitle(
    "M6-F04 — Fixed validation prediction examples"
)

save_figure(fig, "M6_F04_validation_predictions")


# =====================================================================
# 15. RESEARCH REPORT
# =====================================================================

verify_protected()

REPORT = {
    "task_id": "M6-DEV-01B-02",
    "task_passed": True,
    "research_checkpoint": CHECKPOINT,
    "training_completed": True,
    "model_parameters": PARAMETERS,
    "training_scenes": 800,
    "validation_scenes": 160,
    "test_scenes_used": 0,
    "epochs_completed": len(HISTORY),
    "maximum_epochs": MAX_EPOCHS,
    "optimizer_steps": OPTIMIZER_STEPS,
    "early_stopping_triggered": (
        EPOCHS_WITHOUT_IMPROVEMENT >= PATIENCE
    ),
    "best_epoch": BEST_EPOCH,
    "best_validation_foreground_miou": BEST_SCORE,
    "best_validation_metrics": BEST_VALIDATION,
    "reloaded_validation_metrics": RELOADED_VALIDATION,
    "checkpoint_reload_verified": True,
    "checkpoint_path": str(
        (OUTPUT / "best_model.pt").relative_to(ROOT)
    ),
    "checkpoint_sha256": sha256(
        OUTPUT / "best_model.pt"
    ),
    "checkpoint_bytes": (
        OUTPUT / "best_model.pt"
    ).stat().st_size,
    "cache_hash_checks": CACHE_HASH_CHECKS,
    "training_seconds": TRAINING_SECONDS,
    "device": str(DEVICE),
    "dataset_modified": False,
    "architecture_modified": False,
    "controller_modified": False,
    "historical_modifications": False,
    "protected_files": PROTECTED,
    "limitations": [
        "Synthetic procedural dataset.",
        "Validation performance is not held-out test performance.",
        "No optical disturbance augmentation during training.",
        "No demonstrated real-world perception generalization.",
        "Deterministic settings do not guarantee cross-platform identity.",
        "Hardware latency is not measured by this training experiment.",
        "Successful execution does not imply adequate segmentation quality.",
    ],
}

(OUTPUT / "training_report.json").write_text(
    json.dumps(
        REPORT,
        indent=2,
        allow_nan=False,
    ) + "\n"
)


# =====================================================================
# 16. ARTIFACT MANIFEST
# =====================================================================

INVENTORY = {
    str(path.relative_to(ROOT)): {
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    for path in sorted(OUTPUT.rglob("*"))
    if path.is_file()
}

MANIFEST = {
    "task_id": "M6-DEV-01B-02",
    "research_checkpoint": CHECKPOINT,
    "best_epoch": BEST_EPOCH,
    "files": INVENTORY,
}

MANIFEST_PATH = OUTPUT / "training_manifest.json"

MANIFEST_PATH.write_text(
    json.dumps(
        MANIFEST,
        indent=2,
        allow_nan=False,
    ) + "\n"
)

for name, metadata in INVENTORY.items():

    path = ROOT / name

    assert sha256(path) == metadata["sha256"]

    assert path.stat().st_size == metadata["bytes"]

verify_protected()

NEW_PROTECTED = {
    name: metadata["sha256"]
    for name, metadata in INVENTORY.items()
}

NEW_PROTECTED[
    str(MANIFEST_PATH.relative_to(ROOT))
] = sha256(MANIFEST_PATH)


# =====================================================================
# 17. CONTINUATION STATE
# =====================================================================

AHDS_M6_TRAINING_STATE = {
    **REPORT,
    "report_path": str(
        (OUTPUT / "training_report.json").relative_to(ROOT)
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
        "M6-DEV-01C — Frozen Model and Held-out Evaluation"
    ),
}


# =====================================================================
# 18. FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
print("REPRODUCIBLE PERCEPTION TRAINING SUMMARY")
print("=" * 70)

print("Training device:", DEVICE)
print("Trainable parameters:", PARAMETERS)
print("Training scenes: 800")
print("Validation scenes: 160")
print("Test scenes used: 0")
print("Dataset hash checks: 1920/1920")
print("Completed epochs:", len(HISTORY))
print("Optimizer steps:", OPTIMIZER_STEPS)
print("Best epoch:", BEST_EPOCH)

print(
    "Best validation foreground mIoU:",
    f"{100 * BEST_SCORE:.4f}%",
)

print(
    "Best validation pixel accuracy:",
    f"{100 * BEST_VALIDATION['pixel_accuracy']:.4f}%",
)

print(
    "Best validation mean IoU:",
    f"{100 * BEST_VALIDATION['mean_iou']:.4f}%",
)

print(
    "Best validation foreground Dice:",
    f"{100 * BEST_VALIDATION['foreground_mean_dice']:.4f}%",
)

print("\n[BEST VALIDATION CLASS PERFORMANCE]")

for name in CLASS_NAMES:

    value = BEST_VALIDATION["class_iou"][name]

    formatted = (
        f"{100 * value:.4f}%"
        if value is not None
        else "undefined"
    )

    print(f"{name:16s} IoU: {formatted}")

print("\n[ARTIFACT INTEGRITY]")
print("Checkpoint reload: PASS")
print("Validation reproduction: PASS")
print("Protected historical files: 105/105")
print("New artifacts:", len(NEW_PROTECTED))
print("Artifact verification: PASS")
print("Dataset modifications: NONE")
print("Architecture modifications: NONE")
print("Controller modifications: NONE")
print("Historical modifications: NONE")
print("Git staging: NONE")

print(
    "Protected inventory:",
    len(AHDS_M6_TRAINING_STATE["protected_files"]),
)

print(
    "Training duration:",
    f"{TRAINING_SECONDS / 60:.2f} minutes",
)

print(
    "Total execution time:",
    f"{(time.perf_counter() - START_TIME) / 60:.2f} minutes",
)

print("\n[ARTIFACT IDENTITIES]")

for name, digest in NEW_PROTECTED.items():

    print(name)
    print("SHA256:", digest)

print("\nM6-DEV-01B-02: TRAINING EXECUTION PASSED")

print(
    "Next: M6-DEV-01C — "
    "Frozen Model and Held-out Evaluation"
)