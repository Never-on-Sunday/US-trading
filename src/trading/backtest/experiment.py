"""End-to-end experiment: dataset → train → tune on validation → one holdout run → benchmarks."""

import itertools
import json
import time
from datetime import date

import numpy as np
import polars as pl
import yaml

from trading.backtest.engine import StrategyConfig, make_panel, run
from trading.execution.cost_model import CostModel, spread_tiers
from trading.features.api import FEATURE_COLUMNS, build_and_store
from trading.labeling.api import make_labels
from trading.market_data.api import load_5m
from trading.modeling.api import ModelSpec, diagnostics, predict, train
from trading.reporting.benchmarks import buy_and_hold
from trading.reporting.performance import daily_equity, metrics
from trading.shared.storage import ARTIFACTS_DIR, CONFIG_DIR, DATA_DIR, atomic_write_parquet, ensure_dir
from trading.universe.api import get_universe

BAR_COLS = ["symbol", "ts", "session_date", "bar_idx", "n_bars", "open", "close", "has_trade", "is_last_bar"]


def build_dataset(cfg: dict, log=print) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Returns (bars for all symbols, modelling dataset for tradable stocks)."""
    bench = cfg["benchmarks"]
    bars = load_5m()
    log(f"bars: {bars.height:,} rows, {bars['symbol'].n_unique()} symbols, "
        f"{bars['session_date'].min()} → {bars['session_date'].max()}")
    feats = build_and_store(bars, bench)
    horizons = tuple(cfg["horizons"])
    labels = make_labels(bars.filter(~pl.col("symbol").is_in(bench)), horizons)
    ds = feats.join(labels, on=["symbol", "ts"], how="left")
    ds = ds.with_columns([pl.col(c).cast(pl.Float32) for c in FEATURE_COLUMNS])

    active = pl.DataFrame(
        [(i.symbol, i.active_from or date(1900, 1, 1)) for i in get_universe(cfg["universe"])],
        schema=["symbol", "active_from"], orient="row",
    )
    ds = ds.join(active, on="symbol").filter(pl.col("session_date") >= pl.col("active_from"))
    ds = ds.select(*BAR_COLS, *FEATURE_COLUMNS, *[f"fwd_{h}" for h in horizons])
    atomic_write_parquet(ds, DATA_DIR / "features" / "dataset_v1.parquet")
    log(f"dataset: {ds.height:,} rows × {len(FEATURE_COLUMNS)} features")
    return bars, ds


def trainable(ds: pl.DataFrame, target: str) -> pl.DataFrame:
    """Rows the bot could actually act on: a real trade in the bar, room to enter before the close, warm features."""
    return ds.filter(
        pl.col("has_trade") & (pl.col("bars_to_close") >= 2) & pl.col(target).is_not_null()
        & pl.col("r390").is_not_null() & pl.col("vol_tod").is_not_null() & pl.col("smh_r78").is_not_null()
    )


def _costs(bars: pl.DataFrame, upto: date, multiplier: float = 1.0) -> CostModel:
    dv = (bars.filter(pl.col("session_date") <= upto)
          .group_by("symbol", "session_date").agg((pl.col("close") * pl.col("volume")).sum().alias("dv"))
          .group_by("symbol").agg(pl.col("dv").tail(250).median()))
    return CostModel(half_spread_bps=spread_tiers(dict(dv.iter_rows())), multiplier=multiplier)


def _simulate(bars, preds, strat: StrategyConfig, costs: CostModel) -> tuple[dict, pl.DataFrame, pl.DataFrame]:
    res = run(make_panel(bars, preds), strat, costs)
    daily = daily_equity(res["ts"], res["session"], res["equity"])
    return metrics(daily, strat.starting_cash, res["trades"], res["fees"]), daily, res["trades"]


def run_experiment(name: str = "baseline_v1", log=print) -> dict:
    cfg = yaml.safe_load((CONFIG_DIR / "experiments" / f"{name}.yaml").read_text())
    out_dir = ensure_dir(ARTIFACTS_DIR / "runs" / name)
    t0 = time.time()
    bars, ds = build_dataset(cfg, log)
    train_end, val_end = cfg["periods"]["train_end"], cfg["periods"]["validation_end"]
    cash = float(cfg["starting_cash"])
    feats = list(FEATURE_COLUMNS)
    stock_bars = bars.filter(~pl.col("symbol").is_in(cfg["benchmarks"])).select(BAR_COLS)
    val_bars = stock_bars.filter((pl.col("session_date") > train_end) & (pl.col("session_date") <= val_end))
    hold_bars = stock_bars.filter(pl.col("session_date") > val_end)
    result: dict = {"name": name, "config": cfg, "features": feats, "data": {
        "first_session": str(bars["session_date"].min()), "last_session": str(bars["session_date"].max()),
        "rows_5m": bars.height, "dataset_rows": ds.height,
        "holdout_start": str(hold_bars["session_date"].min()), "holdout_end": str(hold_bars["session_date"].max()),
    }}

    # ---- Stage A: train on 2021–2024, tune everything on 2025 -------------------------------
    val_costs = _costs(bars, train_end)
    grid = cfg["grid"]
    stage_a, best = [], None
    for h in cfg["horizons"]:
        target = f"fwd_{h}"
        tr = trainable(ds.filter(pl.col("session_date") <= train_end), target)
        va = trainable(ds.filter((pl.col("session_date") > train_end) & (pl.col("session_date") <= val_end)), target)
        t1 = time.time()
        model = train(tr, feats, target)
        p = predict(model, va, feats)
        diag = diagnostics(va, p, target)
        log(f"[h={h}] train {tr.height:,} rows in {time.time() - t1:.0f}s | val IC {diag['ic_mean']:.4f} "
            f"| top 1% fwd {diag['top0.99']['mean_fwd'] * 1e4:.1f} bp (cut {diag['top0.99']['pred_cut'] * 1e4:.1f} bp)")
        preds = va.select("symbol", "ts").with_columns(pl.Series("pred", p))
        rows = []
        for thr, mp, ext in itertools.product(grid["threshold"], grid["max_positions"], grid["extend"]):
            strat = StrategyConfig(horizon=h, threshold=thr, max_positions=mp, extend=ext, starting_cash=cash)
            m, _, _ = _simulate(val_bars, preds, strat, val_costs)
            rows.append({"horizon": h, "threshold": thr, "max_positions": mp, "extend": ext, **m})
            if m["trades"] >= cfg["min_validation_trades"] and (best is None or m["total_return"] > best["total_return"]):
                best = rows[-1]
        stage_a.append({"horizon": h, "train_rows": tr.height, "val_rows": va.height, "diagnostics": diag, "grid": rows})
        top = max(rows, key=lambda r: r["total_return"] if r["trades"] >= cfg["min_validation_trades"] else -9)
        log(f"[h={h}] best on validation: thr {top['threshold']} maxpos {top['max_positions']} extend {top['extend']} "
            f"→ {top['total_return']:+.1%}, {top['trades']} trades")
    result["validation"] = {"by_horizon": stage_a, "selected": best}
    val_all = stock_bars.filter((pl.col("session_date") > train_end) & (pl.col("session_date") <= val_end))
    result["validation"]["benchmarks"] = _benchmarks(bars, cfg, train_end, val_end, cash, val_costs)[0]
    if best is None:
        raise RuntimeError("no configuration reached the minimum number of validation trades")
    log(f"selected on validation: {best}")
    (out_dir / "validation.json").write_text(json.dumps(result["validation"], indent=1, default=str))

    # ---- Stage B: frozen design, run once on the 2026 holdout -------------------------------
    h, target = best["horizon"], f"fwd_{best['horizon']}"
    strat = StrategyConfig(horizon=h, threshold=best["threshold"], max_positions=best["max_positions"],
                           extend=best["extend"], starting_cash=cash)
    hold_ds = ds.filter(pl.col("session_date") > val_end)
    hold_pred: dict[str, pl.DataFrame] = {}
    if "static" in cfg["retrain"]:
        model = train(trainable(ds.filter(pl.col("session_date") <= val_end), target), feats, target)
        rows_ = trainable(hold_ds, target)
        hold_pred["static"] = rows_.select("symbol", "ts").with_columns(pl.Series("pred", predict(model, rows_, feats)))
        result["holdout_diagnostics_static"] = diagnostics(rows_, hold_pred["static"]["pred"].to_numpy(), target)
        log("static model trained")
    if "monthly" in cfg["retrain"]:
        parts = []
        months = hold_ds.select(pl.col("session_date").dt.truncate("1mo").alias("m")).unique().sort("m")["m"].to_list()
        for m in months:
            model = train(trainable(ds.filter(pl.col("session_date") < m), target), feats, target)
            rows_ = trainable(hold_ds.filter(pl.col("session_date").dt.truncate("1mo") == m), target)
            parts.append(rows_.select("symbol", "ts").with_columns(pl.Series("pred", predict(model, rows_, feats))))
            log(f"monthly retrain {m}: predicted {rows_.height:,} rows")
        hold_pred["monthly"] = pl.concat(parts)

    hold_costs = _costs(bars, val_end)
    holdout = {}
    for mode, preds in hold_pred.items():
        atomic_write_parquet(preds, out_dir / f"predictions_{mode}.parquet")
        variants = {
            "base": (strat, hold_costs),
            "costs_2x": (strat, _costs(bars, val_end, 2.0)),
            "t1_settlement": (StrategyConfig(**{**strat.__dict__, "proceeds": "next_day"}), hold_costs),
            "no_costs": (strat, CostModel(fee_rate=0, fee_min=0, slippage_bps=0, default_half_spread_bps=0,
                                          half_spread_bps={s: 0.0 for s in hold_costs.half_spread_bps})),
        }
        holdout[mode] = {}
        for vname, (s_, c_) in variants.items():
            m, daily, trades = _simulate(hold_bars, preds, s_, c_)
            holdout[mode][vname] = m
            atomic_write_parquet(daily, out_dir / f"equity_{mode}_{vname}.parquet")
            if vname == "base":
                atomic_write_parquet(trades, out_dir / f"trades_{mode}.parquet")
            log(f"holdout {mode}/{vname}: {m['total_return']:+.2%} final ${m['final_equity']:.0f} "
                f"trades {m['trades']} fees ${m['fees']:.0f} maxDD {m['max_drawdown']:.1%}")
    result["holdout"] = holdout
    bm, bm_daily = _benchmarks(bars, cfg, val_end, date(2100, 1, 1), cash, hold_costs)
    result["holdout_benchmarks"] = bm
    for k, d in bm_daily.items():
        atomic_write_parquet(d, out_dir / f"equity_bench_{k}.parquet")
    for k, v in bm.items():
        log(f"benchmark {k}: {v['total_return']:+.2%} final ${v['final_equity']:.0f} maxDD {v['max_drawdown']:.1%}")
    result["runtime_min"] = (time.time() - t0) / 60
    (out_dir / "result.json").write_text(json.dumps(result, indent=1, default=str))
    return result


def _benchmarks(bars, cfg, after: date, upto: date, cash: float, costs: CostModel):
    period = bars.filter((pl.col("session_date") > after) & (pl.col("session_date") <= upto))
    first_day = period["session_date"].min()
    out, dailies = {}, {}
    for b in cfg["benchmarks"]:
        d = buy_and_hold(period.filter(pl.col("symbol") == b), {b: 1.0}, cash, costs)
        out[b], dailies[b] = metrics(d, cash), d
    stocks = period.filter(~pl.col("symbol").is_in(cfg["benchmarks"]))
    present = stocks.filter(pl.col("session_date") == first_day)["symbol"].unique().to_list()
    d = buy_and_hold(stocks, {s: 1 / len(present) for s in present}, cash, costs)
    out["basket_equal_weight"], dailies["basket_equal_weight"] = metrics(d, cash), d
    out["basket_equal_weight"]["n_stocks"] = len(present)
    return out, dailies
