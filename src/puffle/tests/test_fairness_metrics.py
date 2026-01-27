# ABOUTME: Unit tests for FairnessMetrics.
# ABOUTME: Covers dictionary-like access and error handling for backward compatibility.

import pytest

from puffle.Utils.fairness_metrics import FairnessMetrics


@pytest.fixture
def metrics():
    return FairnessMetrics(
        loss=1.0, accuracy=0.8, f1=0.75, disparity=0.1, statistics={"total": 100}
    )


def test_fairness_metrics_to_dict(metrics):
    d = metrics.to_dict()
    assert d["loss"] == 1.0
    assert d["accuracy"] == 0.8
    assert d["f1"] == 0.75
    assert d["disparity"] == 0.1
    assert d["statistics"] == {"total": 100}


def test_fairness_metrics_getitem(metrics):
    assert metrics["loss"] == 1.0
    assert metrics["accuracy"] == 0.8
    assert metrics["statistics"] == {"total": 100}


def test_fairness_metrics_getitem_error(metrics):
    with pytest.raises(TypeError, match="Key must be a string"):
        _ = metrics[123]

    with pytest.raises(KeyError):
        _ = metrics["non_existent"]


def test_fairness_metrics_contains(metrics):
    assert "loss" in metrics
    assert "accuracy" in metrics
    assert "non_existent" not in metrics


def test_fairness_metrics_get(metrics):
    assert metrics.get("loss") == 1.0
    assert metrics.get("non_existent") is None
    assert metrics.get("non_existent", 0.0) == 0.0
