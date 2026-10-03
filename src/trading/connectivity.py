"""Smoke checks that API keys work: Alpaca (data + paper trading) and Binance Stocks."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from trading.shared.config import get_secrets


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _run(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name, True, fn())
    except Exception as exc:  # report every failure, keep checking the rest
        return CheckResult(name, False, f"{type(exc).__name__}: {exc}")


def check_alpaca(symbol: str = "NVDA") -> list[CheckResult]:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.client import TradingClient

    s = get_secrets()
    key, secret = s.alpaca_api_key.get_secret_value(), s.alpaca_secret_key.get_secret_value()

    def account() -> str:
        acct = TradingClient(key, secret, paper=s.alpaca_paper).get_account()
        return f"{'paper' if s.alpaca_paper else 'LIVE'} account {acct.status}, equity ${acct.equity}"

    def bars(feed: DataFeed) -> Callable[[], str]:
        def fetch() -> str:
            end = datetime.now(UTC) - timedelta(minutes=20)  # free plan: no SIP for the last 15 min
            req = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
                start=end - timedelta(days=5), end=end, feed=feed,
            )
            df = StockHistoricalDataClient(key, secret).get_stock_bars(req).df
            if df.empty:
                raise RuntimeError("no bars returned")
            return f"{len(df)} 1m bars for {symbol}, last at {df.index[-1][1]}"
        return fetch

    return [
        _run("alpaca.trading_account", account),
        _run("alpaca.bars_sip", bars(DataFeed.SIP)),
        _run("alpaca.bars_iex", bars(DataFeed.IEX)),
    ]


def check_binance(symbol: str = "NVDA") -> list[CheckResult]:
    from binance_common.configuration import ConfigurationRestAPI
    from binance_common.constants import STOCKS_REST_API_PROD_URL
    from binance_sdk_stocks.stocks import Stocks

    s = get_secrets()
    api = Stocks(config_rest_api=ConfigurationRestAPI(
        api_key=s.binance_api_key.get_secret_value(),
        api_secret=s.binance_secret_key.get_secret_value(),
        base_path=STOCKS_REST_API_PROD_URL,
    )).rest_api

    def exchange_info() -> str:  # API key only
        info = api.exchange_info(symbol=symbol).data()
        sym = info.symbols[0]
        return (f"{sym.symbol}: tradability={sym.tradability}, fractionable={sym.fractionable}, "
                f"minNotional={sym.min_notional}")

    def quote() -> str:  # API key only
        q = api.latest_quote(symbol=symbol).data()
        if q is None or q.bid_price is None:
            return "no quote right now (market closed or empty response)"
        return f"{symbol} bid {q.bid_price} x {q.bid_size} / ask {q.ask_price} x {q.ask_size}"

    def signed() -> str:  # needs the secret: proves signing works and the disclaimer state
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        hist = api.equity_order_history(start_time=now_ms - 7 * 86_400_000, end_time=now_ms).data()
        rows = getattr(hist, "rows", None) or []
        return f"signed request ok, {len(rows)} stock orders in the last 7 days"

    return [
        _run("binance.exchange_info", exchange_info),
        _run("binance.latest_quote", quote),
        _run("binance.signed_order_history", signed),
    ]
