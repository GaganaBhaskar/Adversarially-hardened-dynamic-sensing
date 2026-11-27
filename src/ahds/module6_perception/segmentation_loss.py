
"""
Weighted cross-entropy and foreground soft Dice loss.

Class weights are derived exclusively from training-set
pixel frequencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def derive_class_weights(pixel_counts):

    counts = torch.as_tensor(
        pixel_counts,
        dtype=torch.float32,
    )

    if counts.ndim != 1 or counts.numel() < 2:
        raise ValueError("Invalid class-count vector.")

    if not torch.isfinite(counts).all():
        raise ValueError("Class counts must be finite.")

    if (counts <= 0).any():
        raise ValueError("All classes must have positive counts.")

    frequencies = counts / counts.sum()

    # Square-root inverse frequency avoids the extreme
    # weighting produced by direct inverse frequency.

    weights = frequencies.rsqrt()

    return weights / weights.mean()


class SegmentationLoss(nn.Module):

    def __init__(
        self,
        class_weights,
        ce_weight=0.5,
        dice_weight=0.5,
        epsilon=1e-6,
    ):

        super().__init__()

        weights = torch.as_tensor(
            class_weights,
            dtype=torch.float32,
        )

        if weights.ndim != 1 or weights.numel() < 2:
            raise ValueError("Invalid class weights.")

        if not torch.isfinite(weights).all():
            raise ValueError("Nonfinite class weights.")

        if (weights <= 0).any():
            raise ValueError("Weights must be positive.")

        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss coefficients must be nonnegative.")

        if abs(ce_weight + dice_weight - 1.0) > 1e-6:
            raise ValueError("Loss coefficients must sum to one.")

        if epsilon <= 0:
            raise ValueError("Epsilon must be positive.")

        self.register_buffer("class_weights", weights)

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.epsilon = epsilon

    def forward(self, logits, target):

        if logits.ndim != 4:
            raise ValueError("Expected batched semantic logits.")

        if logits.shape[1] != self.class_weights.numel():
            raise ValueError("Class-count mismatch.")

        if target.shape != (
            logits.shape[0],
            logits.shape[2],
            logits.shape[3],
        ):
            raise ValueError("Target shape mismatch.")

        if target.dtype != torch.long:
            raise ValueError("Target must contain integer class IDs.")

        ce = F.cross_entropy(
            logits,
            target,
            weight=self.class_weights,
        )

        probabilities = torch.softmax(logits, dim=1)

        one_hot = F.one_hot(
            target,
            num_classes=logits.shape[1],
        ).permute(0, 3, 1, 2).to(probabilities.dtype)

        # Exclude background from Dice aggregation.

        probabilities = probabilities[:, 1:]
        one_hot = one_hot[:, 1:]

        dimensions = (0, 2, 3)

        intersection = (
            probabilities * one_hot
        ).sum(dim=dimensions)

        predicted_mass = probabilities.sum(dim=dimensions)

        target_mass = one_hot.sum(dim=dimensions)

        dice = (
            2 * intersection + self.epsilon
        ) / (
            predicted_mass + target_mass + self.epsilon
        )

        present = target_mass > 0

        if present.any():

            dice_loss = 1.0 - dice[present].mean()

        else:

            # Cross-entropy still penalizes false foreground
            # predictions in background-only batches.

            dice_loss = logits.sum() * 0.0

        total = (
            self.ce_weight * ce
            + self.dice_weight * dice_loss
        )

        return total, {
            "cross_entropy": ce.detach(),
            "foreground_dice_loss": dice_loss.detach(),
            "total_loss": total.detach(),
        }
