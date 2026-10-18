# -*- coding: utf-8 -*-
"""Practice 04 - Question 1: Single Neuron with Backpropagation."""

import numpy as np


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int,
) -> tuple[list[float], float, list[float]]:
    """Train a sigmoid neuron with full-batch gradient descent on MSE.

    Returns:
        (weights rounded to 4 decimals as a list,
         bias rounded to 4 decimals,
         pre-update MSE in each epoch, rounded to 4 decimals)
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    w = np.asarray(initial_weights, dtype=np.float64).copy().reshape(-1)
    b = float(initial_bias)
    mse_values: list[float] = []

    n = y.size
    for _ in range(epochs):
        # Forward propagation: a = sigmoid(Xw + b).
        z = x @ w + b
        a = 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

        # Mean squared error before this epoch's weight update.
        error = a - y
        mse_values.append(round(float(np.mean(error ** 2)), 4))

        # Backpropagation via the chain rule.
        delta = (2.0 / n) * error * a * (1.0 - a)
        dw = x.T @ delta
        db = float(np.sum(delta))

        # Gradient descent.
        w -= learning_rate * dw
        b -= learning_rate * db

    return np.round(w, 4).tolist(), round(b, 4), mse_values
