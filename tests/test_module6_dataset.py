
import pytest
import torch

from ahds.module6_perception.procedural_dataset import (
    NUM_CLASSES,
    SPLITS,
    render_scene,
    ProceduralPerceptionDataset,
)


def test_split_sizes():

    assert len(ProceduralPerceptionDataset("train")) == 800
    assert len(ProceduralPerceptionDataset("validation")) == 160
    assert len(ProceduralPerceptionDataset("test")) == 160


def test_disjoint_split_seeds():

    sets = [
        set(range(start, start + count))
        for start, count in SPLITS.values()
    ]

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j])


def test_scene_shape_and_dtype():

    scene, target, _ = render_scene(100000)

    assert scene.shape == (3, 256, 256)
    assert target.shape == (256, 256)
    assert scene.dtype == torch.float32
    assert target.dtype == torch.long


def test_valid_semantic_labels():

    _, target, _ = render_scene(100000)

    assert int(target.min()) >= 0
    assert int(target.max()) < NUM_CLASSES


def test_nonnegative_finite_radiance():

    scene, _, _ = render_scene(100000)

    assert torch.isfinite(scene).all()
    assert (scene >= 0).all()


def test_scene_reproducibility():

    a, ya, _ = render_scene(100000)
    b, yb, _ = render_scene(100000)

    assert torch.equal(a, b)
    assert torch.equal(ya, yb)


def test_geometry_variation():

    a, ya, _ = render_scene(100000)
    b, yb, _ = render_scene(100001)

    assert not torch.equal(a, b)
    assert not torch.equal(ya, yb)


def test_model_input_shape():

    sample = ProceduralPerceptionDataset("train")[0]

    assert sample["input"].shape == (4, 256, 256)


def test_missing_samples():

    dataset = ProceduralPerceptionDataset("train")

    for index in range(16):

        x = dataset[index]["input"]
        mask = x[3].bool()

        assert (x[:3, ~mask] == 0).all()


def test_ground_truth_independent_of_sensor():

    dataset = ProceduralPerceptionDataset("train")

    _, target, _ = dataset.optical_sample(0)

    assert torch.equal(
        target,
        dataset[0]["target"],
    )


def test_acquisition_reproducibility():

    dataset = ProceduralPerceptionDataset("train")

    assert torch.equal(
        dataset[0]["input"],
        dataset[0]["input"],
    )


def test_invalid_split():

    with pytest.raises(ValueError):
        ProceduralPerceptionDataset("invalid")


def test_invalid_index():

    dataset = ProceduralPerceptionDataset("train")

    with pytest.raises(IndexError):
        dataset[-1]

    with pytest.raises(IndexError):
        dataset[len(dataset)]
