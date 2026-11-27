
import pytest
import torch

from ahds.module6_perception.lightweight_segmentation import (
    LightweightSegmentationNet,
)

from ahds.module6_perception.segmentation_loss import (
    derive_class_weights,
    SegmentationLoss,
)

from ahds.module6_perception.segmentation_metrics import (
    SegmentationMetrics,
)


def test_parameter_budget():

    model = LightweightSegmentationNet()

    assert sum(
        p.numel() for p in model.parameters()
    ) < 50000


def test_output_shape():

    model = LightweightSegmentationNet().eval()

    with torch.no_grad():

        output = model(torch.rand(2, 4, 64, 64))

    assert output.shape == (2, 4, 64, 64)
    assert torch.isfinite(output).all()


def test_odd_resolution():

    model = LightweightSegmentationNet().eval()

    with torch.no_grad():

        output = model(torch.rand(1, 4, 65, 71))

    assert output.shape == (1, 4, 65, 71)


def test_invalid_input_channels():

    model = LightweightSegmentationNet()

    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 64, 64))


def test_sampling_mask_compatibility():

    model = LightweightSegmentationNet().eval()

    x = torch.rand(2, 4, 64, 64)

    x[0, 3] = 1

    x[1, 3] = 0
    x[1, 3, ::4, ::4] = 1
    x[1, :3] *= x[1, 3]

    with torch.no_grad():
        output = model(x)

    assert torch.isfinite(output).all()


def test_class_weight_order():

    weights = derive_class_weights(
        [9600, 200, 150, 50]
    )

    assert weights[3] > weights[2]
    assert weights[2] > weights[1]
    assert weights[1] > weights[0]

    assert torch.isclose(
        weights.mean(),
        torch.tensor(1.0),
    )


def test_invalid_class_counts():

    with pytest.raises(ValueError):
        derive_class_weights([100, 0, 20, 10])


def test_finite_loss_and_gradient():

    logits = torch.randn(
        2, 4, 32, 32,
        requires_grad=True,
    )

    target = torch.randint(
        0, 4, (2, 32, 32)
    )

    criterion = SegmentationLoss(torch.ones(4))

    loss, components = criterion(logits, target)

    assert torch.isfinite(loss)

    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_background_only_loss():

    logits = torch.randn(
        1, 4, 16, 16,
        requires_grad=True,
    )

    target = torch.zeros(
        1, 16, 16,
        dtype=torch.long,
    )

    criterion = SegmentationLoss(torch.ones(4))

    loss, components = criterion(logits, target)

    assert torch.isfinite(loss)
    assert components["foreground_dice_loss"] == 0

    loss.backward()

    assert torch.isfinite(logits.grad).all()


def test_perfect_metrics():

    target = torch.tensor([[[0, 1, 2, 3]]])

    metrics = SegmentationMetrics()

    metrics.update(target, target)

    result = metrics.compute()

    assert result["pixel_accuracy"] == 1.0
    assert result["foreground_miou"] == 1.0


def test_background_prediction_metrics():

    target = torch.tensor([[[0, 1, 2, 3]]])

    metrics = SegmentationMetrics()

    metrics.update(torch.zeros_like(target), target)

    result = metrics.compute()

    assert result["pixel_accuracy"] == 0.25
    assert result["foreground_miou"] == 0.0


def test_absent_class_handling():

    target = torch.zeros(
        1, 4, 4,
        dtype=torch.long,
    )

    metrics = SegmentationMetrics()

    metrics.update(target, target)

    result = metrics.compute()

    assert result["class_iou"] == [
        1.0, None, None, None
    ]

    assert result["foreground_miou"] is None


def test_metric_accumulation():

    target = torch.tensor([[[0, 1, 2, 3]]])

    prediction = torch.tensor([[[0, 1, 1, 3]]])

    full = SegmentationMetrics()
    full.update(prediction, target)

    partial = SegmentationMetrics()

    partial.update(
        prediction[:, :, :2],
        target[:, :, :2],
    )

    partial.update(
        prediction[:, :, 2:],
        target[:, :, 2:],
    )

    assert full.compute() == partial.compute()


def test_invalid_semantic_class():

    metrics = SegmentationMetrics()

    with pytest.raises(ValueError):

        metrics.update(
            torch.tensor([[[4]]]),
            torch.tensor([[[0]]]),
        )


def test_metric_reset():

    metrics = SegmentationMetrics()

    target = torch.tensor([[[0, 1, 2, 3]]])

    metrics.update(target, target)

    metrics.reset()

    result = metrics.compute()

    assert result["evaluated_pixels"] == 0
    assert result["pixel_accuracy"] is None


def test_end_to_end_gradient():

    model = LightweightSegmentationNet()

    criterion = SegmentationLoss(torch.ones(4))

    x = torch.rand(2, 4, 64, 64)

    target = torch.randint(
        0, 4, (2, 64, 64)
    )

    loss, _ = criterion(model(x), target)

    loss.backward()

    for parameter in model.parameters():

        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
