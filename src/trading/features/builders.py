import polars as pl

BARS_PER_DAY = 78
S = "symbol"


def _lr(n: int) -> pl.Expr:
    """Log return over the last n bars (crosses session boundaries, so n=78 ≈ one day)."""
    return (pl.col("close") / pl.col("close").shift(n)).log().over(S)


def _per_symbol(bars: pl.DataFrame) -> pl.DataFrame:
    df = bars.sort(S, "ts")
    day = [S, "session_date"]
    df = df.with_columns(
        _lr(1).alias("r1"), _lr(3).alias("r3"), _lr(6).alias("r6"), _lr(12).alias("r12"),
        _lr(BARS_PER_DAY).alias("r78"), _lr(5 * BARS_PER_DAY).alias("r390"),
        pl.col("open").first().over(day).alias("_day_open"),
        (pl.col("vwap") * pl.col("volume")).cum_sum().over(day).alias("_pv"),
        pl.col("volume").cum_sum().over(day).alias("_cv"),
        pl.col("high").cum_max().over(day).alias("_day_high"),
        pl.col("low").cum_min().over(day).alias("_day_low"),
    )
    prev_close = pl.col("close").shift(1).over(S)
    df = df.with_columns(
        (pl.col("close") / pl.col("_day_open")).log().alias("ret_open"),
        # overnight gap: today's open vs yesterday's last close (constant through the day)
        (pl.col("_day_open") / pl.when(pl.col("bar_idx") == 0).then(prev_close).otherwise(None)
         .forward_fill().over(S)).log().alias("gap"),
        pl.col("r1").rolling_std(12).over(S).alias("vol12"),
        pl.col("r1").rolling_std(BARS_PER_DAY).over(S).alias("vol78"),
        pl.col("r1").rolling_std(5 * BARS_PER_DAY).over(S).alias("vol390"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("bar_range"),
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-12)).alias("close_loc"),
        (pl.col("close") / (pl.col("_pv") / (pl.col("_cv") + 1e-9)) - 1).alias("vwap_dist"),
        ((pl.col("close") - pl.col("_day_low")) / (pl.col("_day_high") - pl.col("_day_low") + 1e-12)).alias("day_pos"),
        (pl.col("r1").clip(lower_bound=0).rolling_mean(14) / (pl.col("r1").abs().rolling_mean(14) + 1e-12)).over(S).alias("rsi14"),
        # volume vs the same time-of-day slot over the previous 20 sessions (removes the U-shape)
        (pl.col("volume") / (pl.col("volume").shift(1).rolling_mean(20).over(S, "bar_idx") + 1.0)).alias("vol_tod"),
        (pl.col("_cv") / (pl.col("_cv").shift(1).rolling_mean(20).over(S, "bar_idx") + 1.0)).alias("cumvol_tod"),
        (pl.col("bar_idx") / pl.col("n_bars")).alias("tod"),
        (pl.col("n_bars") - 1 - pl.col("bar_idx")).cast(pl.Int32).alias("bars_to_close"),
        pl.col("session_date").dt.weekday().alias("dow"),
    )
    return df.with_columns(
        (pl.col("r6") / (pl.col("vol78") + 1e-9)).alias("r6_z"),
        (pl.col("r78") / (pl.col("vol390") + 1e-9)).alias("r78_z"),
        (pl.col("vol12") / (pl.col("vol390") + 1e-9)).alias("vol_ratio"),
        (pl.col("vol_tod") + 1e-6).log().alias("vol_tod"),
        (pl.col("cumvol_tod") + 1e-6).log().alias("cumvol_tod"),
    ).drop("_day_open", "_pv", "_cv", "_day_high", "_day_low")


CONTEXT = ("r1", "r6", "r78", "ret_open")
RANKED = ("r1", "r6", "r12", "r78", "r390", "ret_open", "gap", "vol_tod", "vol78")
BASE = ["r1", "r3", "r6", "r12", "r78", "r390", "ret_open", "gap", "vol12", "vol78", "vol390", "bar_range",
        "close_loc", "vwap_dist", "day_pos", "rsi14", "vol_tod", "cumvol_tod", "tod", "bars_to_close", "dow",
        "r6_z", "r78_z", "vol_ratio"]


def build_features(bars: pl.DataFrame, benchmark_symbols: list[str]) -> pl.DataFrame:
    df = _per_symbol(bars)
    bench = df.filter(pl.col(S).is_in(benchmark_symbols))
    stocks = df.filter(~pl.col(S).is_in(benchmark_symbols))

    # Market context: each benchmark's returns at the same bar.
    ctx_cols: list[str] = []
    for b in benchmark_symbols:
        sub = bench.filter(pl.col(S) == b).select("ts", *[pl.col(c).alias(f"{b.lower()}_{c}") for c in CONTEXT])
        ctx_cols += [f"{b.lower()}_{c}" for c in CONTEXT]
        stocks = stocks.join(sub, on="ts", how="left")

    # Cross-sectional: where each stock stands vs the rest of the universe at the same bar.
    n = pl.len().over("ts")
    stocks = stocks.with_columns(
        *[((pl.col(c).rank().over("ts") - 1) / (n - 1).clip(lower_bound=1)).alias(f"{c}_rank") for c in RANKED],
        *[(pl.col(c) - pl.col(c).mean().over("ts")).alias(f"{c}_xs") for c in ("r6", "r78", "ret_open")],
        (pl.col("r6") - pl.col("smh_r6")).alias("r6_vs_smh"),
        (pl.col("r78") - pl.col("smh_r78")).alias("r78_vs_smh"),
        (pl.col("ret_open") - pl.col("smh_ret_open")).alias("ret_open_vs_smh"),
        (pl.col("ret_open") > 0).mean().over("ts").alias("breadth"),
    )
    return stocks.sort("ts", S)


FEATURE_COLUMNS = (
    BASE
    + [f"{b}_{c}" for b in ("spy", "qqq", "smh") for c in CONTEXT]
    + [f"{c}_rank" for c in RANKED]
    + ["r6_xs", "r78_xs", "ret_open_xs", "r6_vs_smh", "r78_vs_smh", "ret_open_vs_smh", "breadth"]
)
