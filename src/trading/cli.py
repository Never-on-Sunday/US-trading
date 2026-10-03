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
