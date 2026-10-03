"""Alpaca market data over plain REST (faster than alpaca-py's per-bar pydantic objects)."""

import threading
import time
from datetime import date, datetime

import polars as pl
import requests

from trading.shared.config import get_secrets

DATA_URL = "https://data.alpaca.markets/v2"
TRADING_URL = {True: "https://paper-api.alpaca.markets/v2", False: "https://api.alpaca.markets/v2"}
BAR_SCHEMA = {
    "symbol": pl.Utf8, "ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "high": pl.Float64,
    "low": pl.Float64, "close": pl.Float64, "volume": pl.Float64, "trade_count": pl.Int64, "vwap": pl.Float64,
}


class RateLimiter:
    """Simple shared limiter: at most `per_minute` calls across threads."""

    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        if delay:
            time.sleep(delay)


class AlpacaProvider:
    def __init__(self, per_minute: int = 190):
        s = get_secrets()
        self.session = requests.Session()
        self.session.headers.update({
            "APCA-API-KEY-ID": s.alpaca_api_key.get_secret_value(),
            "APCA-API-SECRET-KEY": s.alpaca_secret_key.get_secret_value(),
        })
        self.trading_url = TRADING_URL[s.alpaca_paper]
        self.limiter = RateLimiter(per_minute)

    def _get(self, url: str, params: dict) -> dict:
        for attempt in range(8):
            self.limiter.wait()
            try:
                r = self.session.get(url, params=params, timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(60, 2 ** attempt))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Alpaca request failed after retries: {url} {params}")

    def bars(self, symbols: list[str], start: datetime, end: datetime, timeframe: str = "1Min",
             feed: str = "sip", adjustment: str = "split") -> pl.DataFrame:
        params = {
            "symbols": ",".join(symbols), "timeframe": timeframe, "start": start.isoformat(),
            "end": end.isoformat(), "limit": 10000, "adjustment": adjustment, "feed": feed, "sort": "asc",
        }
        cols: dict[str, list] = {k: [] for k in ("symbol", "t", "o", "h", "l", "c", "v", "n", "vw")}
        while True:
            body = self._get(f"{DATA_URL}/stocks/bars", params)
            for sym, rows in (body.get("bars") or {}).items():
                for b in rows:
                    cols["symbol"].append(sym)
                    for k in ("t", "o", "h", "l", "c", "v", "n", "vw"):
                        cols[k].append(b.get(k))
            token = body.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        # Alpaca sends whole-number prices as JSON ints, so force floats instead of inferring.
        floats = {k: pl.Float64 for k in ("o", "h", "l", "c", "v", "vw")}
        df = pl.DataFrame(cols, schema_overrides={"n": pl.Int64, **floats}, strict=False)
        if df.is_empty():
            return pl.DataFrame(schema=BAR_SCHEMA)
        return df.select(
            pl.col("symbol"),
            pl.col("t").str.to_datetime(time_zone="UTC", time_unit="us").alias("ts"),
            pl.col("o").cast(pl.Float64).alias("open"), pl.col("h").cast(pl.Float64).alias("high"),
            pl.col("l").cast(pl.Float64).alias("low"), pl.col("c").cast(pl.Float64).alias("close"),
            pl.col("v").alias("volume"), pl.col("n").alias("trade_count"), pl.col("vw").alias("vwap"),
        )

    def calendar(self, start: date, end: date) -> pl.DataFrame:
        rows = self._get(f"{self.trading_url}/calendar", {"start": start.isoformat(), "end": end.isoformat()})
        return pl.DataFrame(rows).select(
            pl.col("date").str.to_date().alias("session_date"),
            pl.col("open").alias("open_et"), pl.col("close").alias("close_et"),
        )
