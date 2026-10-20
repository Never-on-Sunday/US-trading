import numpy as np
import polars as pl


def daily_equity(ts: np.ndarray, session: np.ndarray, equity: np.ndarray) -> pl.DataFrame:
    return (pl.DataFrame({"session_date": session, "equity": equity})
            .group_by("session_date", maintain_order=True).agg(pl.col("equity").last()))


def metrics(daily: pl.DataFrame, start_cash: float, trades: pl.DataFrame | None = None, fees: float = 0.0) -> dict:
    eq = daily["equity"].to_numpy()
    rets = np.diff(np.concatenate([[start_cash], eq])) / np.concatenate([[start_cash], eq[:-1]])
    peak = np.maximum.accumulate(np.concatenate([[start_cash], eq]))
    dd = (np.concatenate([[start_cash], eq]) / peak - 1).min()
    out = {
        "final_equity": float(eq[-1]), "total_return": float(eq[-1] / start_cash - 1),
        "sharpe": float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0,
        "max_drawdown": float(dd), "days": int(len(eq)), "fees": float(fees),
    }
    if trades is not None:
        n = trades.height
        pnl = (trades["proceeds"] - trades["cost"]).to_numpy() if n else np.array([])
        gross = ((trades["exit_px"] / trades["entry_px"] - 1)).to_numpy() if n else np.array([])
        out |= {
            "trades": n, "win_rate": float((pnl > 0).mean()) if n else 0.0,
            "avg_trade_net_pct": float((pnl / trades["cost"].to_numpy()).mean()) if n else 0.0,
            "avg_trade_gross_pct": float(gross.mean()) if n else 0.0,
            "net_pnl": float(pnl.sum()) if n else 0.0,
        }
    return out
