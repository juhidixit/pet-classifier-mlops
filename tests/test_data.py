import numpy as np
import torch
from PIL import Image

from petcls.data import (
    CAT_BREEDS,
    denormalize,
    eval_transform,
    species_of,
    stratified_split,
    train_transform,
)


def test_transforms_output_shape():
    img = Image.fromarray(np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8))
    assert train_transform(128)(img).shape == (3, 128, 128)
    assert eval_transform(128)(img).shape == (3, 128, 128)


def test_denormalize_roundtrip_range():
    x = eval_transform(64)(Image.new("RGB", (80, 80), (255, 0, 0)))
    d = denormalize(x)
    assert d.min() >= 0 and d.max() <= 1
    assert torch.allclose(d[:, 32, 32], torch.tensor([1.0, 0.0, 0.0]), atol=1e-2)


def test_stratified_split_is_disjoint_and_balanced():
    labels = [i % 37 for i in range(37 * 20)]
    tr, va = stratified_split(labels, 0.1, seed=0)
    assert set(tr).isdisjoint(va)
    assert len(tr) + len(va) == len(labels)
    counts = np.bincount(np.array(labels)[va], minlength=37)
    assert counts.min() >= 1 and counts.max() <= 3  # every class represented in validation


def test_split_is_deterministic():
    labels = [i % 5 for i in range(100)]
    assert np.array_equal(stratified_split(labels, 0.2, 1)[1], stratified_split(labels, 0.2, 1)[1])


def test_species_mapping():
    assert len(CAT_BREEDS) == 12  # Oxford-IIIT Pet has 12 cat and 25 dog breeds
    assert species_of("Persian") == "cat"
    assert species_of("Beagle") == "dog"
