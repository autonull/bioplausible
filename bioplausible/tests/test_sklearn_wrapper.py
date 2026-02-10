"""
Tests for Scikit-Learn Wrapper
"""

import numpy as np
import pytest
import torch

from bioplausible.sklearn_interface import EqPropClassifier


def test_eqprop_classifier_init():
    clf = EqPropClassifier(model_name="EqProp MLP", hidden_dim=64, steps=10)
    assert clf.model_name == "EqProp MLP"
    assert clf.hidden_dim == 64
    assert clf.steps == 10


def test_eqprop_classifier_fit_predict():
    # Simple XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int64)

    clf = EqPropClassifier(
        model_name="EqProp MLP",
        hidden_dim=32,
        steps=10,
        epochs=50,
        batch_size=4,
        device="cpu",
        random_state=42,
    )

    clf.fit(X, y)

    # Check that model is initialized
    assert clf.model_ is not None
    assert clf.trainer_ is not None

    # Predict
    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape

    # Probabilities
    y_prob = clf.predict_proba(X)
    assert y_prob.shape == (4, 2)


def test_eqprop_classifier_partial_fit():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([0, 1, 1, 0], dtype=np.int64)
    classes = np.array([0, 1])

    clf = EqPropClassifier(
        model_name="EqProp MLP", hidden_dim=32, steps=10, device="cpu", random_state=42
    )

    # First call needs classes
    clf.partial_fit(X, y, classes=classes)
    assert clf.model_ is not None

    # Subsequent calls
    for _ in range(10):
        clf.partial_fit(X, y)

    y_pred = clf.predict(X)
    assert y_pred.shape == y.shape


def test_unknown_model_raises_error():
    clf = EqPropClassifier(model_name="NonExistentModel")
    X = np.array([[0, 0]], dtype=np.float32)
    y = np.array([0], dtype=np.int64)

    with pytest.raises(ValueError):
        clf.fit(X, y)
