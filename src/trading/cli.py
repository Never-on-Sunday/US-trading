import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """Trading bot CLI."""


@app.command("check-connections")
def check_connections(symbol: str = "NVDA") -> None:
    """Verify Alpaca and Binance API keys work. Read-only: places no orders."""
    from trading.connectivity import check_alpaca, check_binance

    results = check_alpaca(symbol) + check_binance(symbol)
    for r in results:
        typer.echo(f"{'OK  ' if r.ok else 'FAIL'}  {r.name:<30} {r.detail}")
    raise typer.Exit(0 if all(r.ok for r in results) else 1)


@app.command("data-sync")
def data_sync(start: str = "2016-01-01", timeframe: str = "20Min", workers: int = 4,
              refresh: bool = False) -> None:
    """Download bars for every stock in the Nasdaq-100 since `start` into data/raw/nasdaq-100/."""
    from datetime import date

    import polars as pl

    from trading.data.api import membership, resolve_series, sync_calendar, sync_series
    from trading.shared.storage import DATA_DIR, atomic_write_parquet

    out_dir = DATA_DIR / "raw" / "nasdaq-100"
    path = out_dir / "membership.parquet"
    if path.exists() and not refresh:
        series = pl.read_parquet(path)
    else:
        periods = membership(date.fromisoformat(start).year)
        typer.echo(f"membership: {periods['ticker'].n_unique()} tickers, {periods.height} periods")
        series = resolve_series(periods)
        atomic_write_parquet(series, path)
    cal = sync_calendar(date.fromisoformat(start), date.today())  # one request: sessions + half-days
    typer.echo(f"calendar: {cal.height} sessions")
    sync_series(series, out_dir, date.fromisoformat(start), timeframe=timeframe, workers=workers,
                refresh=refresh, log=lambda m: typer.echo(m))
