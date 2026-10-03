"""Public API of the features module: point-in-time features on 5m bars.

Every feature at bar t uses only bars <= t (the bar is known once it closes at ts + 5 min).
All features are scale-free (returns, ratios, ranks) so one pooled model can learn across symbols.
"""

import polars as pl

from trading.features.builders import FEATURE_COLUMNS, build_features  # noqa: F401
from trading.shared.storage import DATA_DIR, atomic_write_parquet

FEATURE_SET = "v1"


def features_path():
    return DATA_DIR / "features" / f"{FEATURE_SET}.parquet"


def build_and_store(bars: pl.DataFrame, benchmark_symbols: list[str]) -> pl.DataFrame:
    feats = build_features(bars, benchmark_symbols)
    atomic_write_parquet(feats, features_path())
    return feats


def load_features() -> pl.DataFrame:
    return pl.read_parquet(features_path())
