"""Public API of the modeling module: train a pooled regressor that predicts forward return.

Estimator: scikit-learn HistGradientBoostingRegressor (histogram GBDT, the LightGBM algorithm).
LightGBM/XGBoost need libomp, which is not installable on Intel macOS via Homebrew any more;
they can be swapped in here on Linux / Apple Silicon without touching other modules.
"""

from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class ModelSpec:
    max_iter: int = 300
    learning_rate: float = 0.04
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 2000
    l2_regularization: float = 5.0
    max_features: float = 0.6
    target_clip: float = 0.03  # clip forward returns to ±3% so a few outliers don't dominate
    seed: int = 7
    extra: dict = field(default_factory=dict)


def _xy(df: pl.DataFrame, features: list[str], target: str, clip: float):
    x = df.select(features).to_numpy().astype(np.float32)
    y = np.clip(df[target].to_numpy(), -clip, clip).astype(np.float32)
    return x, y


def train(df: pl.DataFrame, features: list[str], target: str, spec: ModelSpec = ModelSpec()):
    x, y = _xy(df, features, target, spec.target_clip)
    model = HistGradientBoostingRegressor(
        max_iter=spec.max_iter, learning_rate=spec.learning_rate, max_leaf_nodes=spec.max_leaf_nodes,
        min_samples_leaf=spec.min_samples_leaf, l2_regularization=spec.l2_regularization,
        max_features=spec.max_features, early_stopping=False, random_state=spec.seed,
    )
    return model.fit(x, y)


def predict(model, df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict(df.select(features).to_numpy().astype(np.float32))


def diagnostics(df: pl.DataFrame, pred: np.ndarray, target: str) -> dict:
    """How good are the predictions before any trading: rank correlation and top-bucket payoff."""
    d = df.select("ts", target).with_columns(pl.Series("pred", pred))
    ic = [
        spearmanr(g["pred"].to_numpy(), g[target].to_numpy()).statistic
        for _, g in d.group_by("ts") if g.height >= 10
    ]
    y = d[target].to_numpy()
    out = {"ic_mean": float(np.nanmean(ic)), "ic_hit": float(np.nanmean(np.array(ic) > 0)), "mean_fwd": float(y.mean())}
    for q in (0.9, 0.99, 0.999):
        cut = np.quantile(pred, q)
        sel = pred >= cut
        out[f"top{q}"] = {"pred_cut": float(cut), "mean_fwd": float(y[sel].mean()), "n": int(sel.sum())}
    return out
