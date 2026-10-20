"""Downloads from Alpaca: trading calendar and per-series bar history (resumable)."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta

import polars as pl

from trading.data.providers.alpaca import AlpacaProvider
from trading.shared.storage import DATA_DIR, atomic_write_parquet, ensure_dir


def sync_calendar(start: date, end: date) -> pl.DataFrame:
    cal = AlpacaProvider().calendar(start, end)
    atomic_write_parquet(cal, DATA_DIR / "raw" / "calendar.parquet")
    return cal


def resolve_series(periods: pl.DataFrame, provider: AlpacaProvider | None = None) -> pl.DataFrame:
    """Map each membership period to a download series: (series id, ticker, asof).

    Tickers get renamed and reused (FB is now another company; SNDK was SanDisk in 2016 and a new
    Sandisk in 2026), so each period asks Alpaca for the company that held the ticker on its last day
    in the index. Periods of one ticker that resolve to the same company share one series.
    """
    provider = provider or AlpacaProvider()
    periods = periods.with_columns(asof=pl.col("left") - timedelta(days=1))
    out = []
    for (ticker,), grp in periods.sort("joined").group_by("ticker", maintain_order=True):
        rows = grp.to_dicts()
        if len(rows) == 1:
            out.append({**rows[0], "series": ticker})
            continue
        # A different company under the same ticker shows up as a different daily history.
        sigs = {}
        for r in rows:
            df = provider.bars([ticker], datetime(2015, 12, 1, tzinfo=UTC), datetime.now(UTC) - timedelta(days=1),
                               timeframe="1Day", asof=r["asof"])
            sigs[r["asof"]] = tuple(df.head(5)["close"]) + tuple(df.tail(5)["close"])
        latest = rows[-1]["asof"]
        for r in rows:
            same = sigs[r["asof"]] == sigs[latest]
            out.append({**r, "series": ticker if same else f"{ticker}.{r['asof']:%Y}",
                        "asof": latest if same else r["asof"]})
    return pl.DataFrame(out).select("ticker", "joined", "left", "series", "asof")


def sync_series(series: pl.DataFrame, out_dir, start: date, timeframe: str = "20Min", workers: int = 4,
                refresh: bool = False, log=print) -> None:
    """Download each series' full history into `out_dir/bars_<tf>/series=<id>.parquet`. Resumable."""
    provider = AlpacaProvider()
    end_dt = datetime.now(UTC) - timedelta(minutes=16)  # free plan: no SIP in last 15 min
    bars_dir = ensure_dir(out_dir / f"bars_{timeframe.lower()}")
    manifest_path = out_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    jobs = series.unique("series").sort("series").select("series", "ticker", "asof").rows()
    todo = [j for j in jobs if refresh or j[0] not in manifest]
    log(f"{len(jobs)} series, {len(todo)} to download ({timeframe}, {start} → {end_dt:%Y-%m-%d %H:%M}Z)")

    def job(sid: str, ticker: str, asof: date | None):
        df = provider.bars([ticker], datetime(start.year, start.month, start.day, tzinfo=UTC), end_dt,
                           timeframe=timeframe, asof=asof).with_columns(symbol=pl.lit(sid))
        atomic_write_parquet(df, bars_dir / f"series={sid}.parquet")
        return sid, ticker, asof, df

    failed, done = [], 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(job, *j): j[0] for j in todo}
        for f in as_completed(futures):
            try:
                sid, ticker, asof, df = f.result()
            except Exception as exc:  # keep going; a failed series is retried on the next sync
                failed.append(futures[f])
                log(f"{futures[f]}: FAILED {type(exc).__name__}: {exc}")
                continue
            done += 1
            manifest[sid] = {
                "ticker": ticker, "asof": asof.isoformat() if asof else None, "timeframe": timeframe,
                "rows": df.height, "first": str(df["ts"].min()) if df.height else None,
                "last": str(df["ts"].max()) if df.height else None,
                "downloaded_at": datetime.now(UTC).isoformat(),
            }
            manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            log(f"[{done}/{len(todo)}] {sid}: {df.height:,} bars")
    if failed:
        raise RuntimeError(f"{len(failed)} series failed: {sorted(failed)}")
