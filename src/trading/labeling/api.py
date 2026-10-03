"""Public API of the labeling module: forward returns the way the strategy would realise them.

Decision at the close of bar t → buy at the OPEN of bar t+1 → sell at the open of bar t+1+h,
or at the open of the session's last bar if that comes first (the bot is flat overnight).
Returns are gross; costs are subtracted by the backtest and by the trade threshold.
"""

import polars as pl


def make_labels(bars: pl.DataFrame, horizons: tuple[int, ...]) -> pl.DataFrame:
    day = ["symbol", "session_date"]
    df = bars.sort("symbol", "ts").with_columns(
        pl.col("open").shift(-1).over(day).alias("_entry"),
        pl.col("open").last().over(day).alias("_day_close"),
    )
    out = df.with_columns(
        *[(pl.col("open").shift(-(1 + h)).over(day).fill_null(pl.col("_day_close")) / pl.col("_entry") - 1)
          .alias(f"fwd_{h}") for h in horizons]
    )
    return out.select("symbol", "ts", *[f"fwd_{h}" for h in horizons])
