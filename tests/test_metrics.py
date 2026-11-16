import numpy as np
import pytest

from petcls.metrics import expected_calibration_error, most_confused_pairs, summarize, topk_accuracy


def test_topk():
    probs = np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    y = np.array([1, 1, 0])
    assert topk_accuracy(probs, y, 1) == pytest.approx(1 / 3)
    assert topk_accuracy(probs, y, 2) == pytest.approx(2 / 3)
    assert topk_accuracy(probs, y, 3) == 1.0


def test_ece_perfectly_calibrated_is_zero():
    probs = np.array([[1.0, 0.0]] * 10)
    assert expected_calibration_error(probs, np.zeros(10, dtype=int)) == pytest.approx(0.0)


def test_ece_overconfident():
    probs = np.array([[0.99, 0.01]] * 10)
    y = np.array([0] * 5 + [1] * 5)
    assert expected_calibration_error(probs, y) == pytest.approx(0.49, abs=1e-6)


def test_confused_pairs_and_summary():
    names = ["a", "b", "c"]
    cm = np.array([[5, 3, 0], [1, 4, 0], [0, 0, 6]])
    assert most_confused_pairs(cm, names)[0] == {"true": "a", "predicted": "b", "count": 3}
    probs = np.eye(3)[[0, 1, 2, 1]]
    s = summarize(probs, np.array([0, 1, 2, 0]), names)
    assert s["top1_accuracy"] == 0.75
    assert s["per_class_accuracy"]["a"] == 0.5
