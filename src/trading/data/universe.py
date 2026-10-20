"""Universe: a stock is tradable on a date only while it is in the Nasdaq-100 (no survivorship bias)."""

from datetime import date

import polars as pl

from trading.shared.storage import DATA_DIR

MEMBERSHIP = DATA_DIR / "raw" / "nasdaq-100" / "membership.parquet"


def load_membership() -> pl.DataFrame:
    """One row per (ticker, membership period): ticker, joined, left (null = still a member), series, asof."""
    return pl.read_parquet(MEMBERSHIP)


def members_on(day: date) -> list[str]:
    """Series ids in the index on `day`. Changes take effect at the open of `joined` / `left`."""
    m = load_membership()
    on = m.filter((pl.col("joined") <= day) & (pl.col("left").is_null() | (pl.col("left") > day)))
    return sorted(on["series"].unique().to_list())
