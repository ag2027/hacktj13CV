"""
QML rover decision engine using PennyLane variational model components.

QML pattern follows the PennyLane QML style:
- AngleEmbedding for feature encoding
- StronglyEntanglingLayers as trainable variational block
- gradient-based training on labeled data

This design uses a hybrid policy:
- distance gates decide STOP / TURN-ZONE / FORWARD
- QML model decides LEFT vs RIGHT while in TURN-ZONE
"""

from __future__ import annotations

from dataclasses import dataclass

import pennylane as qml
from pennylane import numpy as np


@dataclass
class QMLConfig:
    n_qubits: int = 2
    n_layers: int = 2
    lr: float = 0.25
    steps: int = 35
    n_samples: int = 64
    seed: int = 13
    stop_distance_cm: float = 12.0
    turn_distance_cm: float = 30.0


class QMLDecisionEngine:
    """Hybrid distance-gated decision engine with QML steering core."""

    def __init__(self, config: QMLConfig | None = None):
        self.config = config or QMLConfig()
        self.dev = qml.device("default.qubit", wires=self.config.n_qubits)

        rng = np.random.default_rng(self.config.seed)
        self.weights = 0.01 * rng.normal(
            size=(self.config.n_layers, self.config.n_qubits, 3), requires_grad=True
        )
        self.trained = False

        @qml.qnode(self.dev)
        def circuit(features, weights):
            qml.AngleEmbedding(features, wires=range(self.config.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(self.config.n_qubits))
            return qml.expval(qml.PauliZ(0))

        self._circuit = circuit

    @staticmethod
    def _features(distance_cm: float, turn_hint: float):
        d = max(0.0, min(distance_cm, 100.0)) / 100.0
        # turn_hint: 0 (prefer left), 1 (prefer right)
        return np.array([d, turn_hint], requires_grad=False)

    def _sample_training_data(self):
        """
        Train only for steering (LEFT/RIGHT) in obstacle turn zone.
        Label 0 => LEFT, label 1 => RIGHT.
        """
        rng = np.random.default_rng(self.config.seed)
        xs = []
        ys = []
        half = self.config.n_samples // 2

        for _ in range(half):
            d = float(rng.uniform(self.config.stop_distance_cm + 1.0, self.config.turn_distance_cm))
            xs.append(self._features(d, 0.0))
            ys.append(0.0)

        for _ in range(self.config.n_samples - half):
            d = float(rng.uniform(self.config.stop_distance_cm + 1.0, self.config.turn_distance_cm))
            xs.append(self._features(d, 1.0))
            ys.append(1.0)

        order = rng.permutation(len(xs))
        x = np.array(xs, requires_grad=False)[order]
        y = np.array(ys, requires_grad=False)[order]
        return x, y

    @staticmethod
    def _to_prob_right(expval_z: float):
        # Map [-1, 1] -> [0, 1]
        return float((expval_z + 1.0) / 2.0)

    def _loss(self, weights, xs, ys):
        losses = []
        for x, y in zip(xs, ys):
            p_right = (self._circuit(x, weights) + 1.0) / 2.0
            losses.append((p_right - float(y)) ** 2)
        return np.mean(np.array(losses))

    def train(self):
        xs, ys = self._sample_training_data()
        opt = qml.GradientDescentOptimizer(stepsize=self.config.lr)

        w = self.weights
        for _ in range(self.config.steps):
            w = opt.step(lambda ww: self._loss(ww, xs, ys), w)

        self.weights = w
        self.trained = True

    def predict(self, distance_cm: float, turn_hint: float = 0.0):
        if not self.trained:
            self.train()

        d = float(distance_cm)
        if d <= self.config.stop_distance_cm:
            return "STOP", [1.0, 0.0, 0.0, 0.0]

        if d > self.config.turn_distance_cm:
            return "FORWARD", [0.0, 0.0, 0.0, 1.0]

        # QML steering decision in turn-zone
        x = self._features(d, float(turn_hint))
        expval = float(self._circuit(x, self.weights))
        p_right = self._to_prob_right(expval)
        p_left = 1.0 - p_right

        command = "RIGHT" if p_right >= 0.5 else "LEFT"

        # Return 4-command probabilities aligned to [STOP, LEFT, RIGHT, FORWARD]
        return command, [0.0, p_left, p_right, 0.0]
