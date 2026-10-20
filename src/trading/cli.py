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
def data_sync(start: str = "2021-01-01", workers: int = 3) -> None:
    """Download 1m bars (universe + benchmarks) and the trading calendar from Alpaca."""
    from datetime import UTC, date, datetime, timedelta

    from trading.data.api import symbols, sync, sync_calendar

    syms = symbols("ai_chain") + symbols("benchmarks")
    cal = sync_calendar(date.fromisoformat(start), datetime.now(UTC).date())
    typer.echo(f"calendar: {cal.height} sessions; {len(syms)} symbols")
    sync(syms, date.fromisoformat(start), workers=workers, log=lambda m: typer.echo(m))


@app.command("data-sync-n100")
def data_sync_n100(start: str = "2016-01-01", timeframe: str = "20Min", workers: int = 4,
                   refresh: bool = False) -> None:
    """Download bars for every stock in the Nasdaq-100 since `start` into data/raw/nasdaq-100/."""
    from datetime import date

    import polars as pl

    from trading.data.api import membership, resolve_series, sync_series
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
    sync_series(series, out_dir, date.fromisoformat(start), timeframe=timeframe, workers=workers,
                refresh=refresh, log=lambda m: typer.echo(m))


@app.command("data-clean")
def data_clean() -> None:
    """Raw 1m bars → clean regular-session 5m bars."""
    from trading.data.api import build_clean_5m

    build_clean_5m(log=lambda m: typer.echo(m))


@app.command("experiment")
def experiment(name: str = "baseline_v1", reuse_dataset: bool = False) -> None:
    """Run an end-to-end experiment: features → train → validate → holdout → benchmarks."""
    from trading.pipeline.experiment import run_experiment

    run_experiment(name, log=lambda m: typer.echo(m), reuse_dataset=reuse_dataset)


@app.command("report")
def report(name: str = "baseline_v1") -> None:
    """Build the HTML report for a finished experiment run."""
    from trading.report.api import build_report

    typer.echo(f"report written to {build_report(name)}")
