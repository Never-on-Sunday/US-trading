"""Event-driven backtest on 5m bars.

Timeline per bar t: the bar closes → the strategy decides using predictions made from bars <= t
→ orders fill at the OPEN of bar t+1 (never at the close that produced the signal), with
spread, slippage and fees. The bot is long-only, flat overnight. The engine owns execution and
the cash ledger; every trading decision comes from the Strategy (trading.strategy.api).
"""

from dataclasses import dataclass

import numpy as np
import polars as pl

from trading.backtest.costs import CostModel
from trading.strategy.api import Strategy


@dataclass(frozen=True)
class Account:
    starting_cash: float = 2000.0
    min_order: float = 350.0     # below this Binance's $0.35 minimum fee exceeds 0.10%
    proceeds: str = "immediate"  # "immediate" | "next_day" (T+1 settlement of sale proceeds)


@dataclass
class Panel:
    """Dense [time × symbol] matrices for fast simulation."""
    ts: np.ndarray
    session: np.ndarray
    bar_idx: np.ndarray
    n_bars: np.ndarray
    symbols: list[str]
    open: np.ndarray
    close: np.ndarray
    pred: np.ndarray


def make_panel(bars: pl.DataFrame, preds: pl.DataFrame) -> Panel:
    """bars: symbol, ts, session_date, bar_idx, n_bars, open, close. preds: symbol, ts, pred."""
    df = bars.join(preds, on=["symbol", "ts"], how="left")
    symbols = sorted(df["symbol"].unique().to_list())
    times = df.select("ts", "session_date", "bar_idx", "n_bars").unique("ts").sort("ts")

    def mat(col: str) -> np.ndarray:
        wide = df.pivot(on="symbol", index="ts", values=col).sort("ts")
        return wide.select(symbols).to_numpy().astype(np.float64)

    return Panel(times["ts"].to_numpy(), times["session_date"].to_numpy(), times["bar_idx"].to_numpy(),
                 times["n_bars"].to_numpy(), symbols, mat("open"), mat("close"), mat("pred"))


def run(panel: Panel, strategy: Strategy, costs: CostModel, account: Account = Account()) -> dict:
    impact = np.array([costs.price_impact(s) for s in panel.symbols])
    n_t, _ = panel.open.shape
    cash, unsettled, fees_paid = account.starting_cash, 0.0, 0.0
    pos: dict[int, list] = {}  # symbol index -> [qty, entry_bar, cost_incl_fee, entry_px]
    last_close = np.zeros(len(panel.symbols))
    equity = np.empty(n_t)
    trades = []

    for i in range(n_t):
        if i > 0 and panel.session[i] != panel.session[i - 1]:
            cash, unsettled = cash + unsettled, 0.0  # yesterday's sale proceeds settle
        c = panel.close[i]
        ok = ~np.isnan(c)
        last_close[ok] = c[ok]
        equity[i] = cash + unsettled + sum(p[0] * last_close[j] for j, p in pos.items())

        b, nb = panel.bar_idx[i], panel.n_bars[i]
        if b >= nb - 1 or i + 1 >= n_t:
            continue  # last bar of the session: nothing can fill today any more
        nxt_open = panel.open[i + 1]
        pred = panel.pred[i]
        force_flat = b == nb - 2  # sell everything at the open of the last bar

        for j in list(pos):
            qty, entry_bar, cost, entry_px = pos[j]
            if not force_flat:
                action = strategy.manage(i + 1 - entry_bar, pred[j])
                if action == "hold":
                    continue
                if action == "renew":
                    pos[j][1] = i + 1
                    continue
            px = (nxt_open[j] if not np.isnan(nxt_open[j]) else last_close[j]) * (1 - impact[j])
            gross = qty * px
            fee = costs.fee(gross)
            fees_paid += fee
            if account.proceeds == "immediate":
                cash += gross - fee
            else:
                unsettled += gross - fee
            trades.append((panel.symbols[j], entry_bar, i + 1, entry_px, px, cost, gross - fee))
            del pos[j]

        if force_flat or not strategy.can_enter(b, nb):
            continue
        slots = strategy.max_positions - len(pos)
        if slots <= 0:
            continue
        cand = strategy.rank_entries(pred, ~np.isnan(nxt_open))
        if cand.size == 0:
            continue
        target = strategy.target_notional(equity[i])
        for j in cand:
            if slots <= 0:
                break
            if j in pos:
                continue
            budget = min(target, cash)
            notional = budget / (1 + costs.fee_rate * costs.multiplier)
            fee = costs.fee(notional)
            if notional < account.min_order or notional + fee > cash + 1e-9:
                break
            px = nxt_open[j] * (1 + impact[j])
            pos[j] = [notional / px, i + 1, notional + fee, px]
            cash -= notional + fee
            fees_paid += fee
            slots -= 1

    cols = list(zip(*trades)) if trades else [[] for _ in range(7)]
    ts = pl.Series("ts", panel.ts)  # bar indices → timestamps (raw numpy datetimes in tuples become Object dtype)
    tr = pl.DataFrame({
        "symbol": pl.Series(cols[0], dtype=pl.Utf8),
        "entry_ts": ts.gather(pl.Series(cols[1], dtype=pl.Int64)), "exit_ts": ts.gather(pl.Series(cols[2], dtype=pl.Int64)),
        "entry_px": pl.Series(cols[3], dtype=pl.Float64), "exit_px": pl.Series(cols[4], dtype=pl.Float64),
        "cost": pl.Series(cols[5], dtype=pl.Float64), "proceeds": pl.Series(cols[6], dtype=pl.Float64),
    })
    return {"ts": panel.ts, "session": panel.session, "equity": equity, "trades": tr, "fees": fees_paid}
