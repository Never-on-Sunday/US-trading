"""Raw 1m bars → clean regular-session 5m bars on a full grid (one row per symbol per 5m slot).

The 5m bar is the decision timeframe because Binance's live kline stream has no 1m interval.
A slot with no trades keeps `has_trade=False`, volume 0 and OHLC equal to the previous close,
so prices never look like a trade happened when none did.
"""

from pathlib import Path

import polars as pl

from trading.shared.storage import DATA_DIR, atomic_write_parquet

ET = "America/New_York"
BAR_MINUTES = 5
CLEAN_5M = DATA_DIR / "clean" / "bars_5m"


def load_calendar() -> pl.DataFrame:
    cal = pl.read_parquet(DATA_DIR / "raw" / "calendar.parquet")
    return cal.with_columns(
        (pl.col("open_et").str.slice(0, 2).cast(pl.Int32) * 60 + pl.col("open_et").str.slice(3, 2).cast(pl.Int32)).alias("open_min"),
        (pl.col("close_et").str.slice(0, 2).cast(pl.Int32) * 60 + pl.col("close_et").str.slice(3, 2).cast(pl.Int32)).alias("close_min"),
    ).select("session_date", "open_min", "close_min")


def _rth_5m(raw: pl.DataFrame, cal: pl.DataFrame) -> pl.DataFrame:
    df = raw.with_columns(pl.col("ts").dt.convert_time_zone(ET).alias("ts_et"))
    df = df.with_columns(
        pl.col("ts_et").dt.date().alias("session_date"),
        (pl.col("ts_et").dt.hour().cast(pl.Int32) * 60 + pl.col("ts_et").dt.minute().cast(pl.Int32)).alias("minute"),
    ).join(cal, on="session_date", how="inner")
    df = df.filter((pl.col("minute") >= pl.col("open_min")) & (pl.col("minute") < pl.col("close_min")))
    df = df.with_columns(((pl.col("minute") - pl.col("open_min")) // BAR_MINUTES).cast(pl.Int16).alias("bar_idx"))
    return (
        df.sort("symbol", "ts")
        .group_by("symbol", "session_date", "bar_idx", maintain_order=True)
        .agg(
            pl.col("open").first(), pl.col("high").max(), pl.col("low").min(), pl.col("close").last(),
            pl.col("volume").sum(), pl.col("trade_count").sum(),
            ((pl.col("vwap") * pl.col("volume")).sum() / pl.col("volume").sum()).alias("vwap"),
        )
    )


def build_clean_5m(raw_dir: Path = DATA_DIR / "raw" / "bars_1m", log=print) -> None:
    cal = load_calendar()
    parts = []
    for f in sorted(raw_dir.glob("month=*.parquet")):
        parts.append(_rth_5m(pl.read_parquet(f), cal))
    bars = pl.concat(parts)
    log(f"traded 5m bars: {bars.height:,}")

    # Full grid: every symbol × every session it has data for × every 5m slot of that session.
    sessions = cal.with_columns(((pl.col("close_min") - pl.col("open_min")) // BAR_MINUTES).alias("n_bars"))
    span = bars.group_by("symbol").agg(pl.col("session_date").min().alias("first"), pl.col("session_date").max().alias("last"))
    grid = (
        span.join(sessions, how="cross")
        .filter(pl.col("session_date").is_between(pl.col("first"), pl.col("last")))
        .with_columns(pl.int_ranges(0, pl.col("n_bars")).alias("bar_idx"))
        .explode("bar_idx")
        .with_columns(pl.col("bar_idx").cast(pl.Int16))
        .select("symbol", "session_date", "bar_idx", "open_min", "n_bars")
    )
    full = grid.join(bars, on=["symbol", "session_date", "bar_idx"], how="left").sort("symbol", "session_date", "bar_idx")
    full = full.with_columns(pl.col("close").is_not_null().alias("has_trade"))
    full = full.with_columns(pl.col("close").forward_fill().over("symbol").alias("close"))
    full = full.with_columns(
        pl.col("open").fill_null(pl.col("close")), pl.col("high").fill_null(pl.col("close")),
        pl.col("low").fill_null(pl.col("close")), pl.col("vwap").fill_null(pl.col("close")),
        pl.col("volume").fill_null(0.0), pl.col("trade_count").fill_null(0),
    ).drop_nulls("close")  # leading slots before a symbol's first trade
    # Bar start timestamp in UTC (decision time = ts + 5 min).
    full = full.with_columns(
        (pl.col("session_date").cast(pl.Datetime("us")).dt.replace_time_zone(ET)
         + pl.duration(minutes=pl.col("open_min") + pl.col("bar_idx").cast(pl.Int32) * BAR_MINUTES))
        .dt.convert_time_zone("UTC").alias("ts"),
        (pl.col("bar_idx") == pl.col("n_bars") - 1).alias("is_last_bar"),
    ).drop("open_min")
    for (sym,), g in full.partition_by("symbol", as_dict=True).items():
        atomic_write_parquet(g, CLEAN_5M / f"symbol={sym}.parquet")
    q = full.group_by("symbol").agg(pl.len().alias("bars"), (1 - pl.col("has_trade").mean()).alias("empty_share"))
    log(q.sort("empty_share", descending=True).head(8))


def load_5m(symbols: list[str] | None = None) -> pl.DataFrame:
    files = sorted(CLEAN_5M.glob("symbol=*.parquet"))
    if symbols is not None:
        wanted = set(symbols)
        files = [f for f in files if f.stem.split("=", 1)[1] in wanted]
    return pl.concat([pl.read_parquet(f) for f in files])
