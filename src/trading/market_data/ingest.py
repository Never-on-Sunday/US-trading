"""Resumable backfill: one raw parquet file per calendar month containing all symbols."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime, timedelta

import polars as pl

from trading.market_data.providers.alpaca import AlpacaProvider
from trading.shared.storage import DATA_DIR, atomic_write_parquet, ensure_dir

RAW_1M = DATA_DIR / "raw" / "bars_1m"
MANIFEST = DATA_DIR / "raw" / "_manifest.json"


def _months(start: date, end: date) -> list[date]:
    out, d = [], date(start.year, start.month, 1)
    while d <= end:
        out.append(d)
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def sync(symbols: list[str], start: date, end_dt: datetime | None = None, workers: int = 3,
         log=print) -> None:
    """Download 1m bars month by month. Completed past months are skipped; the current month is refreshed."""
    provider = AlpacaProvider()
    end_dt = end_dt or (datetime.now(UTC) - timedelta(minutes=16))  # free plan: no SIP in last 15 min
    manifest = _load_manifest()
    key_syms = ",".join(sorted(symbols))
    for f in RAW_1M.glob("month=*.parquet"):  # files are written atomically, so an existing past month is complete
        m = date.fromisoformat(f.stem.split("=")[1] + "-01")
        m_next = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
        if m.isoformat() not in manifest and datetime(m_next.year, m_next.month, 1, tzinfo=UTC) <= end_dt:
            manifest[m.isoformat()] = {"status": "ok", "complete": True, "symbols": key_syms,
                                       "rows": pl.scan_parquet(f).select(pl.len()).collect().item()}
    todo = []
    for m in _months(start, end_dt.date()):
        m_end = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
        complete_month = datetime(m_end.year, m_end.month, m_end.day, tzinfo=UTC) <= end_dt
        entry = manifest.get(m.isoformat())
        if entry and entry["status"] == "ok" and entry["complete"] and entry["symbols"] == key_syms:
            continue
        todo.append((m, min(datetime(m_end.year, m_end.month, m_end.day, tzinfo=UTC), end_dt), complete_month))

    ensure_dir(RAW_1M)
    log(f"{len(todo)} months to download")

    def job(m: date, m_end: datetime, complete: bool):
        df = provider.bars(symbols, datetime(m.year, m.month, m.day, tzinfo=UTC), m_end)
        atomic_write_parquet(df, RAW_1M / f"month={m:%Y-%m}.parquet")
        return m, complete, df.height

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(job, *t): t[0] for t in todo}
        for f in as_completed(futures):
            try:
                m, complete, rows = f.result()
            except Exception as exc:  # keep going; a failed month is retried on the next sync
                failed.append(futures[f])
                log(f"{futures[f]:%Y-%m}: FAILED {type(exc).__name__}: {exc}")
                continue
            manifest[m.isoformat()] = {
                "status": "ok", "complete": complete, "rows": rows, "symbols": key_syms,
                "downloaded_at": datetime.now(UTC).isoformat(),
            }
            MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True))
            log(f"{m:%Y-%m}: {rows:,} bars")
    if failed:
        raise RuntimeError(f"{len(failed)} months failed: {sorted(f'{m:%Y-%m}' for m in failed)}")


def sync_calendar(start: date, end: date) -> pl.DataFrame:
    cal = AlpacaProvider().calendar(start, end)
    atomic_write_parquet(cal, DATA_DIR / "raw" / "calendar.parquet")
    return cal
