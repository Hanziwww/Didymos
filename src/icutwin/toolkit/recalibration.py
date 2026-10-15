"""Recalibration of the Twin's risks (Paper P amendment 11, V2 and V2b): per program, a monotone map from the raw risk
(E2 for the five hazard causes, the rollout share otherwise) to the observed rate, fitted on the tuning split's
anchors (inclusion weights).

* isotonic (V2, ``fit``): applied as ``sklearn.isotonic.IsotonicRegression`` predicts, linear interpolation between
  its thresholds, clipped at the ends. Its flat steps can tie risks (and so move AUROC);
* logistic (V2b, ``fit_logistic``): logit(p') = a + b x logit(p) with p clipped to ``CLIP``, by weighted maximum
  likelihood; strictly increasing for b > 0, so ranks (and AUROC) are kept.

The maps are generated outputs (``runs/toolkit/recalibration/<run>.json``, never committed): {program: {"kind"
(isotonic when absent), "x", "y" or "a", "b", "n", "positives", "source"}}. Without a map a risk is returned as it is
and marked uncalibrated.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CLIP = (1e-9, 1 - 1e-9)            # wide enough that no risk the model gives is tied by the clip


def fit(y, p, w=None) -> dict:
    from sklearn.isotonic import IsotonicRegression

    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    ok = np.isfinite(y) & np.isfinite(p)
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p[ok], y[ok], sample_weight=w[ok])
    return {"kind": "isotonic", "x": ir.X_thresholds_.tolist(), "y": ir.y_thresholds_.tolist(), "n": int(ok.sum()),
            "positives": int(y[ok].sum())}


def _logit(p):
    p = np.clip(np.asarray(p, float), *CLIP)
    return np.log(p / (1 - p))


def fit_logistic(y, p, w=None) -> dict:
    """a, b of logit(p') = a + b x logit(p) by weighted maximum likelihood (Newton-Raphson)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(p) if w is None else np.asarray(w, float)
    ok = np.isfinite(y) & np.isfinite(p)
    x, yy, ww = _logit(p[ok]), y[ok], w[ok]
    X = np.stack([np.ones_like(x), x], axis=1)
    beta = np.zeros(2)
    for _ in range(100):
        eta = np.clip(X @ beta, -35, 35)
        q = 1 / (1 + np.exp(-eta))
        g = X.T @ (ww * (yy - q))
        H = (X * (ww * q * (1 - q))[:, None]).T @ X + 1e-10 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"kind": "logistic", "a": float(beta[0]), "b": float(beta[1]), "n": int(ok.sum()),
            "positives": int(yy.sum())}


def path_of(session) -> Path:
    return session.paths.out / "recalibration" / f"{session.run}.json"


def load(path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save(maps: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(maps, indent=1), encoding="utf-8", newline="\n")


def transform(m: dict, p):
    """A map applied to risks (arrays or a scalar)."""
    if m.get("kind") == "logistic":
        return 1 / (1 + np.exp(-np.clip(m["a"] + m["b"] * _logit(p), -35, 35)))
    return np.interp(np.asarray(p, float), np.asarray(m["x"], float), np.asarray(m["y"], float))


def apply(maps: dict, program: str, p):
    """(recalibrated risk, True) with a map for ``program``, else (``p``, False)."""
    m = maps.get(program)
    if m is None or p is None or not np.all(np.isfinite(np.asarray(p, float))):
        return p, False
    out = transform(m, p)
    return (float(out) if np.ndim(out) == 0 else out), True
