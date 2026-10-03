"""Buy-and-hold benchmarks, charged the same Binance fee and spread as the bot."""

import polars as pl

from trading.execution.cost_model import CostModel


def buy_and_hold(bars: pl.DataFrame, weights: dict[str, float], start_cash: float, costs: CostModel) -> pl.DataFrame:
    """Buy at the first open of the period, hold, mark at each session close. Returns daily equity."""
    first = bars.sort("ts").group_by("symbol", maintain_order=True).agg(pl.col("open").first())
    qty, cash = {}, start_cash
    for sym, opn in first.iter_rows():
        if sym not in weights:
            continue
        budget = start_cash * weights[sym]
        notional = budget / (1 + costs.fee_rate * costs.multiplier)
        fee = costs.fee(notional)
        notional = min(notional, budget - fee)
        qty[sym] = notional / (opn * (1 + costs.price_impact(sym)))
        cash -= notional + fee
    q = pl.DataFrame({"symbol": list(qty), "qty": list(qty.values())})
    daily = (bars.filter(pl.col("is_last_bar")).join(q, on="symbol")
             .group_by("session_date").agg((pl.col("close") * pl.col("qty")).sum().alias("equity"))
             .sort("session_date").with_columns(pl.col("equity") + cash))
    return daily
