"""Nasdaq-100 membership history, built from github.com/jmccarrell/n100tickers.

Each yearly file lists the members on Jan 1 plus dated changes: `union` joins, `difference` leaves,
effective at the open of that date.
"""

from datetime import date

import polars as pl
import requests
import yaml

SOURCE = "https://raw.githubusercontent.com/jmccarrell/n100tickers/HEAD/src/nasdaq_100_ticker_history"


def _year(y: int) -> dict:
    r = requests.get(f"{SOURCE}/n100-ticker-changes-{y}.yaml", timeout=30)
    r.raise_for_status()
    # BaseLoader keeps every value a string; the safe loader would turn the ticker ON into True.
    return yaml.load(r.text, Loader=yaml.BaseLoader)


def membership(start_year: int = 2016, end_year: int | None = None) -> pl.DataFrame:
    """One row per (ticker, membership period). `left` is null for current members."""
    end_year = end_year or date.today().year
    rows, joined, members = [], {}, None
    for y in range(start_year, end_year + 1):
        doc = _year(y)
        if members is None:
            members = set(doc["tickers_on_Jan_1"])
            joined = {t: date(start_year, 1, 1) for t in members}
        for day, change in sorted((doc.get("changes") or {}).items()):
            d = date.fromisoformat(day)
            for t in change.get("difference") or []:
                rows.append((t, joined.pop(t), d))
                members.discard(t)
            for t in change.get("union") or []:
                members.add(t)
                joined[t] = d
    rows += [(t, joined[t], None) for t in members]
    return pl.DataFrame(rows, schema={"ticker": pl.Utf8, "joined": pl.Date, "left": pl.Date},
                        orient="row").sort("ticker", "joined")
