#!/usr/bin/env python3
"""
train_qml_policy.py

Offline training + export pipeline for an Arduino rover distilled steering policy.
This script trains a model for:
    P(right turn | distance_cm, turn_hint)

Then exports an 11x11 lookup table compatible with Arduino Uno deployment:
    QML_RIGHT_PROB_LUT[11][11]
with integer probabilities 0..100.

Why this script exists:
- Arduino cannot run live quantum circuits.
- We train off-board (laptop), distill policy into a static table,
  and deploy that table in PROGMEM on Arduino firmware.

Required libraries (core):
- numpy
- pandas
- scikit-learn

Optional libraries:
- pennylane (preferred QML-style trainer; falls back automatically if unavailable)
- matplotlib (optional heatmap output)

Install (recommended):
    pip install numpy pandas scikit-learn
Optional:
    pip install pennylane matplotlib

Run:
    python train_qml_policy.py
or:
    python train_qml_policy.py --input training_data.csv --output-dir ./artifacts --samples 3000 --seed 42

Arduino integration:
1) Run trainer:
       python train_qml_policy.py
2) Open generated qml_policy_table.h
3) Copy QML_RIGHT_PROB_LUT into Arduino .ino
4) Replace old table and re-upload firmware
5) Re-upload firmware to Arduino

Quick sample CSV (create as training_data.csv if needed):
----------------------------------------------------------
distance_cm,turn_hint,label
12,90,1
18,20,0
35,80,1
42,30,0
8,65,1
25,15,0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# ----------------------------
# Config
# ----------------------------
DISTANCE_BUCKETS = list(range(0, 101, 10))  # 0,10,...,100
HINT_BUCKETS = list(range(0, 101, 10))      # 0,10,...,100
REQUIRED_COLUMNS = ["distance_cm", "turn_hint", "label"]


@dataclass
class TrainConfig:
    input_csv: Optional[Path]
    output_dir: Path
    samples: int
    seed: int
    test_size: float
    prefer_pennylane: bool


# ----------------------------
# Utilities
# ----------------------------
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def clip_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(
                f"CSV missing required column '{col}'. Required columns: {REQUIRED_COLUMNS}"
            )

    out = df[REQUIRED_COLUMNS].copy()
    out["distance_cm"] = pd.to_numeric(out["distance_cm"], errors="coerce")
    out["turn_hint"] = pd.to_numeric(out["turn_hint"], errors="coerce")
    out["label"] = pd.to_numeric(out["label"], errors="coerce")

    out = out.dropna()
    out["distance_cm"] = out["distance_cm"].clip(0, 100)
    out["turn_hint"] = out["turn_hint"].clip(0, 100)
    out["label"] = out["label"].round().clip(0, 1).astype(int)

    if len(out) < 50:
        raise ValueError(
            f"Not enough valid rows after cleaning ({len(out)}). Need at least 50."
        )

    return out


def generate_synthetic_data(n_samples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Distance and hint are both 0..100
    distance = rng.uniform(0, 100, size=n_samples)
    hint = rng.uniform(0, 100, size=n_samples)

    # Behavior model:
    # - very close obstacles => high turn urgency
    # - turn_hint biases right-vs-left
    # - mid-range distances are sensitive to hint
    # - far distances reduce urgency (direction becomes less decisive)
    closeness = sigmoid((35.0 - distance) / 7.5)  # high when close
    mid_range = np.exp(-((distance - 35.0) ** 2) / (2 * 18.0**2))
    hint_centered = (hint - 50.0) / 50.0  # -1..1

    # Blend urgency with hint influence
    hint_strength = 0.20 + 0.55 * closeness + 0.25 * mid_range
    base = 0.50 + hint_strength * hint_centered

    # Add small stochastic effects/noise
    noise = rng.normal(0.0, 0.06, size=n_samples)
    p_right = np.clip(base + noise, 0.03, 0.97)

    label = rng.binomial(1, p_right, size=n_samples).astype(int)

    df = pd.DataFrame(
        {
            "distance_cm": np.round(distance, 3),
            "turn_hint": np.round(hint, 3),
            "label": label,
        }
    )
    return df


# ----------------------------
# Model wrappers
# ----------------------------
class BasePolicyModel:
    name = "base"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        raise NotImplementedError

    def predict_proba_right(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class SklearnPolicyModel(BasePolicyModel):
    name = "sklearn_logistic_poly"

    def __init__(self, seed: int):
        self.seed = seed
        self.model = Pipeline(
            steps=[
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1200, random_state=seed)),
            ]
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict_proba_right(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]


class PennyLanePolicyModel(BasePolicyModel):
    name = "pennylane_variational_binary"

    def __init__(self, seed: int, n_layers: int = 2, steps: int = 80, batch_size: int = 64, lr: float = 0.18):
        self.seed = seed
        self.n_layers = n_layers
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self._trained = False

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        try:
            import pennylane as qml
            from pennylane import numpy as pnp
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"PennyLane import failed: {exc}") from exc

        rng = np.random.default_rng(self.seed)

        # Normalize to angles [0, pi]
        x_train_q = np.clip(x_train / 100.0, 0.0, 1.0) * np.pi
        y_train_q = y_train.astype(float)

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(x, weights):
            qml.AngleEmbedding(x, wires=[0, 1], rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        weights = pnp.array(
            0.05 * rng.normal(size=(self.n_layers, 2, 3)),
            requires_grad=True,
        )
        bias = pnp.array(0.0, requires_grad=True)
        scale = pnp.array(2.0, requires_grad=True)

        def prob_right(x, w, b, s):
            expv = circuit(x, w)
            logit = s * expv + b
            return 1.0 / (1.0 + pnp.exp(-logit))

        def batch_loss(w, b, s, xb, yb):
            losses = []
            for xi, yi in zip(xb, yb):
                p = prob_right(xi, w, b, s)
                p = pnp.clip(p, 1e-6, 1 - 1e-6)
                losses.append(-(yi * pnp.log(p) + (1 - yi) * pnp.log(1 - p)))
            return pnp.mean(pnp.array(losses))

        opt = qml.GradientDescentOptimizer(self.lr)

        for _ in range(self.steps):
            idx = rng.choice(len(x_train_q), size=min(self.batch_size, len(x_train_q)), replace=False)
            xb = x_train_q[idx]
            yb = y_train_q[idx]
            weights, bias, scale = opt.step(
                lambda ww, bb, ss: batch_loss(ww, bb, ss, xb, yb), weights, bias, scale
            )

        self._qml = qml
        self._pnp = pnp
        self._circuit = circuit
        self._weights = weights
        self._bias = bias
        self._scale = scale
        self._trained = True

    def predict_proba_right(self, x: np.ndarray) -> np.ndarray:
        if not self._trained:
            raise RuntimeError("PennyLane model not trained.")

        x_q = np.clip(x / 100.0, 0.0, 1.0) * np.pi
        probs = []
        for xi in x_q:
            expv = self._circuit(xi, self._weights)
            logit = self._scale * expv + self._bias
            p = 1.0 / (1.0 + self._pnp.exp(-logit))
            probs.append(float(self._pnp.clip(p, 0.0, 1.0)))
        return np.array(probs)


# ----------------------------
# Training + export
# ----------------------------
def choose_model(seed: int, prefer_pennylane: bool) -> BasePolicyModel:
    if prefer_pennylane:
        try:
            model = PennyLanePolicyModel(seed=seed)
            # Probe import early
            import pennylane  # noqa: F401
            return model
        except Exception:
            pass
    return SklearnPolicyModel(seed=seed)


def compute_metrics(y_true: np.ndarray, p_right: np.ndarray) -> dict:
    y_pred = (p_right >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, np.clip(p_right, 1e-6, 1 - 1e-6))),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    # ROC AUC can fail if only one class in y_true
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, p_right))
    except Exception:
        metrics["roc_auc"] = None
    return metrics


def build_lut(model: BasePolicyModel) -> np.ndarray:
    lut = np.zeros((11, 11), dtype=np.uint8)
    for i, d in enumerate(DISTANCE_BUCKETS):
        for j, h in enumerate(HINT_BUCKETS):
            p = model.predict_proba_right(np.array([[d, h]], dtype=float))[0]
            lut[i, j] = np.uint8(np.clip(int(round(p * 100)), 0, 100))
    return lut


def arduino_table_snippet(lut: np.ndarray) -> str:
    lines = []
    lines.append("const uint8_t PROGMEM QML_RIGHT_PROB_LUT[11][11] = {")
    for r in range(11):
        row = ", ".join(str(int(v)) for v in lut[r])
        comma = "," if r < 10 else ""
        lines.append(f"  {{ {row} }}{comma}")
    lines.append("};")
    return "\n".join(lines)


def save_table_csv(lut: np.ndarray, output_dir: Path) -> Path:
    df = pd.DataFrame(
        lut.astype(int),
        index=[f"dist_{d}" for d in DISTANCE_BUCKETS],
        columns=[f"hint_{h}" for h in HINT_BUCKETS],
    )
    out_path = output_dir / "qml_policy_table.csv"
    df.to_csv(out_path, index=True)
    return out_path


def save_table_json(lut: np.ndarray, output_dir: Path) -> Path:
    payload = {
        "distance_buckets_cm": DISTANCE_BUCKETS,
        "turn_hint_buckets": HINT_BUCKETS,
        "right_probability_percent_lut": lut.astype(int).tolist(),
    }
    out_path = output_dir / "qml_policy_table.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


def save_table_header(lut: np.ndarray, output_dir: Path) -> Path:
    # Keep this file as a direct paste-ready snippet for Arduino firmware.
    # The structure must match exactly what the .ino expects.
    header = arduino_table_snippet(lut) + "\n"
    out_path = output_dir / "qml_policy_table.h"
    out_path.write_text(header)
    return out_path


def save_report(
    output_dir: Path,
    source_desc: str,
    model_name: str,
    n_samples: int,
    metrics: dict,
    csv_path: Path,
    json_path: Path,
    header_path: Path,
    heatmap_path: Optional[Path],
) -> Path:
    lines = [
        "QML Distilled Policy Training Report",
        "===================================",
        "",
        f"Data source: {source_desc}",
        f"Samples used: {n_samples}",
        f"Model: {model_name}",
        "",
        "Metrics (holdout set):",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Log loss: {metrics['log_loss']:.4f}",
        f"- ROC AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}",
        f"- Confusion matrix: {metrics['confusion_matrix']}",
        "",
        "Artifacts:",
        f"- CSV: {csv_path}",
        f"- JSON: {json_path}",
        f"- Header: {header_path}",
        f"- Heatmap: {heatmap_path if heatmap_path else 'not generated (matplotlib unavailable)'}",
        "",
        "Deployment steps:",
        "1) Open qml_policy_table.h",
        "2) Copy QML_RIGHT_PROB_LUT into your Arduino .ino",
        "3) Replace old table and re-upload firmware",
    ]
    out_path = output_dir / "training_report.txt"
    out_path.write_text("\n".join(lines))
    return out_path


def save_heatmap_if_possible(lut: np.ndarray, output_dir: Path) -> Optional[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(lut, cmap="viridis", vmin=0, vmax=100, origin="lower", aspect="auto")
    ax.set_title("Distilled Right-Turn Probability LUT (%)")
    ax.set_xlabel("Turn Hint Bucket (0..100)")
    ax.set_ylabel("Distance Bucket cm (0..100)")
    ax.set_xticks(range(11), HINT_BUCKETS)
    ax.set_yticks(range(11), DISTANCE_BUCKETS)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("P(RIGHT) %")
    fig.tight_layout()

    out_path = output_dir / "qml_policy_heatmap.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Offline QML-inspired trainer and LUT exporter for Arduino rover steering."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional input CSV path (distance_cm,turn_hint,label). If missing/not found, synthetic data is generated.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts",
        help="Directory for output artifacts.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3000,
        help="Synthetic sample count when CSV is not used (min 2000 recommended).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Holdout fraction for metrics."
    )
    parser.add_argument(
        "--no-pennylane",
        action="store_true",
        help="Force classical fallback model even if PennyLane is installed.",
    )
    args = parser.parse_args()

    return TrainConfig(
        input_csv=Path(args.input) if args.input else None,
        output_dir=Path(args.output_dir),
        samples=max(2000, int(args.samples)),
        seed=int(args.seed),
        test_size=float(args.test_size),
        prefer_pennylane=not bool(args.no_pennylane),
    )


def load_or_generate_data(cfg: TrainConfig) -> Tuple[pd.DataFrame, str]:
    # explicit input path
    if cfg.input_csv is not None:
        if cfg.input_csv.exists():
            try:
                df = pd.read_csv(cfg.input_csv)
                df = clip_and_validate(df)
                return df, f"CSV ({cfg.input_csv})"
            except Exception as exc:
                print(
                    f"[WARN] Failed to parse input CSV '{cfg.input_csv}': {exc}\n"
                    f"[WARN] Falling back to synthetic data.",
                    file=sys.stderr,
                )
        else:
            print(
                f"[WARN] Input CSV '{cfg.input_csv}' not found. Falling back to synthetic data.",
                file=sys.stderr,
            )

    # implicit default CSV if present
    default_csv = Path("training_data.csv")
    if cfg.input_csv is None and default_csv.exists():
        try:
            df = pd.read_csv(default_csv)
            df = clip_and_validate(df)
            return df, f"CSV ({default_csv})"
        except Exception as exc:
            print(
                f"[WARN] Failed to parse default CSV '{default_csv}': {exc}\n"
                f"[WARN] Falling back to synthetic data.",
                file=sys.stderr,
            )

    # synthetic fallback
    df = generate_synthetic_data(cfg.samples, cfg.seed)
    return df, f"Synthetic ({cfg.samples} samples)"


def main() -> int:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        df, source_desc = load_or_generate_data(cfg)
    except Exception as exc:
        print(f"[ERROR] Could not prepare training data: {exc}", file=sys.stderr)
        return 1

    x = df[["distance_cm", "turn_hint"]].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        print("[ERROR] Training labels contain only one class; need both 0 and 1.", file=sys.stderr)
        return 1

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=y,
        )
    except Exception as exc:
        print(f"[ERROR] Train/test split failed: {exc}", file=sys.stderr)
        return 1

    model = choose_model(cfg.seed, cfg.prefer_pennylane)

    # Try selected model; if it fails and it was PennyLane, fallback
    try:
        model.fit(x_train, y_train)
    except Exception as exc:
        if isinstance(model, PennyLanePolicyModel):
            print(
                f"[WARN] PennyLane training failed ({exc}). Falling back to sklearn model.",
                file=sys.stderr,
            )
            model = SklearnPolicyModel(seed=cfg.seed)
            try:
                model.fit(x_train, y_train)
            except Exception as exc2:
                print(f"[ERROR] Fallback model training failed: {exc2}", file=sys.stderr)
                return 1
        else:
            print(f"[ERROR] Model training failed: {exc}", file=sys.stderr)
            return 1

    try:
        p_test = model.predict_proba_right(x_test)
        metrics = compute_metrics(y_test, p_test)
    except Exception as exc:
        print(f"[ERROR] Evaluation failed: {exc}", file=sys.stderr)
        return 1

    try:
        lut = build_lut(model)
    except Exception as exc:
        print(f"[ERROR] LUT generation failed: {exc}", file=sys.stderr)
        return 1

    try:
        csv_path = save_table_csv(lut, cfg.output_dir)
        json_path = save_table_json(lut, cfg.output_dir)
        header_path = save_table_header(lut, cfg.output_dir)
        heatmap_path = save_heatmap_if_possible(lut, cfg.output_dir)
        report_path = save_report(
            output_dir=cfg.output_dir,
            source_desc=source_desc,
            model_name=model.name,
            n_samples=len(df),
            metrics=metrics,
            csv_path=csv_path,
            json_path=json_path,
            header_path=header_path,
            heatmap_path=heatmap_path,
        )
    except Exception as exc:
        print(f"[ERROR] Failed while saving artifacts: {exc}", file=sys.stderr)
        return 1

    # Console summary
    print("=== Training Summary ===")
    print(f"Data source: {source_desc}")
    print(f"Model used: {model.name}")
    print(f"Samples: {len(df)}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc'] if metrics['roc_auc'] is not None else 'N/A'}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print("")
    print("=== Artifacts ===")
    print(csv_path)
    print(json_path)
    print(header_path)
    print(report_path)
    if heatmap_path:
        print(heatmap_path)
    else:
        print("(Heatmap skipped: matplotlib not available)")
    print("")

    # Print exact Arduino snippet
    print("=== Arduino LUT Snippet ===")
    print(arduino_table_snippet(lut))
    print("")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ----------------------------------------------------------
# Integration instructions (copy to workflow docs if needed)
#
# 1) Run the trainer:
#       python train_qml_policy.py
#
# 2) Open generated qml_policy_table.h
#
# 3) Copy the PROGMEM LUT into the Arduino .ino
#
# 4) Replace the old QML_RIGHT_PROB_LUT table
#
# 5) Re-upload firmware to Arduino
# ----------------------------------------------------------
