
"""
Confusion-matrix-based semantic segmentation metrics.

Rows represent ground truth.
Columns represent predictions.

Metrics are accumulated over pixels before aggregation.
Undefined class metrics are represented by None.
"""

import torch


class SegmentationMetrics:

    def __init__(self, num_classes=4):

        if num_classes < 2:
            raise ValueError("At least two classes required.")

        self.num_classes = num_classes

        self.reset()

    def reset(self):

        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes),
            dtype=torch.int64,
        )

    @torch.no_grad()
    def update(self, prediction, target):

        if prediction.ndim == target.ndim + 1:

            if prediction.shape[1] != self.num_classes:
                raise ValueError("Logit class-count mismatch.")

            prediction = prediction.argmax(dim=1)

        if prediction.shape != target.shape:
            raise ValueError("Prediction-target shape mismatch.")

        if torch.is_floating_point(prediction):
            raise ValueError("Expected integer predictions.")

        if torch.is_floating_point(target):
            raise ValueError("Expected integer targets.")

        prediction = prediction.detach().cpu().long().flatten()
        target = target.detach().cpu().long().flatten()

        if target.numel() == 0:
            raise ValueError("Empty evaluation input.")

        if (
            (target < 0).any()
            or (target >= self.num_classes).any()
            or (prediction < 0).any()
            or (prediction >= self.num_classes).any()
        ):
            raise ValueError("Invalid semantic class ID.")

        indices = (
            target * self.num_classes + prediction
        )

        matrix = torch.bincount(
            indices,
            minlength=self.num_classes ** 2,
        ).reshape(
            self.num_classes,
            self.num_classes,
        )

        self.confusion += matrix

    def compute(self):

        matrix = self.confusion.double()

        tp = matrix.diag()

        actual = matrix.sum(dim=1)
        predicted = matrix.sum(dim=0)

        fp = predicted - tp
        fn = actual - tp

        def safe_ratio(numerator, denominator):

            return [
                float(n / d) if float(d) > 0 else None
                for n, d in zip(numerator, denominator)
            ]

        def defined_mean(values):

            valid = [
                value
                for value in values
                if value is not None
            ]

            return (
                sum(valid) / len(valid)
                if valid else None
            )

        iou = safe_ratio(
            tp,
            tp + fp + fn,
        )

        dice = safe_ratio(
            2 * tp,
            2 * tp + fp + fn,
        )

        precision = safe_ratio(tp, predicted)
        recall = safe_ratio(tp, actual)

        total = float(matrix.sum())

        return {
            "pixel_accuracy": (
                float(tp.sum()) / total
                if total > 0 else None
            ),
            "mean_iou": defined_mean(iou),
            "foreground_miou": defined_mean(iou[1:]),
            "foreground_mean_dice": defined_mean(dice[1:]),
            "class_iou": iou,
            "class_dice": dice,
            "class_precision": precision,
            "class_recall": recall,
            "class_support": actual.long().tolist(),
            "confusion_matrix": self.confusion.tolist(),
            "evaluated_pixels": int(total),
        }
