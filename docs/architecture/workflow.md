# Trading Bot — Architecture & Workflow

Source of requirements: [idea_v2.md](idea_v2.md) (original: [archive/idea_v1.md](archive/idea_v1.md)). Data sources, universe and storage (v1): [archive/data.md](archive/data.md). Universe research: [research/AIChain.md](research/AIChain.md).

**Style:** modular monolith. It is one Python package and one deployable, split into modules with hard boundaries. Every module has a single public entry point (`api.py`). Any module can later be pulled out into its own service (for example `live/` or `data/`) without rewriting the others.

**Tech stack:** Python 3.12, `uv`, `polars` + `pyarrow` (Parquet), `duckdb`, `alpaca-py`, `scikit-learn` (HistGradientBoosting), `optuna` (planned), `pydantic`, `typer`, `binance-sdk-stocks` (official Binance Stocks SDK, `Future`), `pytest`, `import-linter`, `ruff`.

---

## Part 1 — Folder Architecture

Folder-level view of the repository as of 2026-10-20. Only key files are shown. `(planned)` marks parts of the target design that don't exist yet.

```text
trading/
├── pyproject.toml              # deps + `trading` CLI entry point
├── .env.example                # Alpaca + Binance API keys
│
├── configs/                    # every run is driven by config, not code edits
│   ├── universe/               # ai_chain.yaml (~50 symbols), benchmarks.yaml (SPY, QQQ, SMH)
│   ├── data.yaml               # feed, start date, 1m timeframe, 5m decision timeframe
│   └── experiments/            # one yaml = one reproducible run (baseline_v1, train_from_2023)
│
├── data/                       # gitignored — market data
│   ├── raw/                    # downloads: nasdaq-100/ research set (membership + 20m bars)
│   ├── clean/                  # cleaned bars ready for features
│   └── archive/                # v1 data: 1m bars, 5m bars, features
│
├── artifacts/                  # gitignored — run outputs
│   └── runs/<experiment>/      # predictions, trades, equity curves, result.json, report.html
│
├── src/trading/                # one folder per workflow step (idea_v2.md); each exposes api.py
│   ├── data/                   # 1. DATA: universe, Nasdaq-100 membership, Alpaca download, 1m → 5m clean
│   ├── features/               # 2. FEATURES: X (builders.py) + y labels (labels.py)
│   ├── models/                 # 3. MODELS: train / predict / diagnostics
│   ├── strategy/               # 4. STRATEGY: predictions → buy/hold/sell decisions (one file per strategy)
│   ├── backtest/               # 5. BACKTEST: engine (fills, cash ledger) + Binance Stocks costs
│   ├── report/                 # 6. REPORT: metrics, benchmarks, HTML report
│   ├── pipeline/               # runs steps 2 → 6 from one experiment config (train → validate → holdout)
│   ├── shared/                 # config, storage paths, shared types (used by every step)
│   ├── cli.py                  # `trading data-sync | data-clean | experiment | report | …`
│   └── live/                   # (planned, Future) live stream, brokers, reconcile, runner
│
├── tests/                      # (planned)
│
└── docs/
    ├── architecture/           # idea_v2.md (requirements), workflow.md (this file), research/, archive/
    ├── experiments/            # experiment write-ups, grouped by version (v1/)
    └── book/                   # reading notes
```

---

## Part 2 — Workflow

The six steps of [idea_v2.md](idea_v2.md). Each step is one folder in `src/trading/` and hands its output to the next through files on disk, so any step can be re-run on its own.

```text
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ 1 Data       │──>│ 2 Features   │──>│ 3 Models     │
 │   data/      │   │   features/  │   │   models/    │
 └──────────────┘   └──────────────┘   └──────────────┘
                                              │ predictions
 ┌──────────────┐   ┌──────────────┐   ┌──────▼───────┐
 │ 6 Report     │<──│ 5 Backtest   │<──│ 4 Strategy   │
 │   report/    │   │   backtest/  │   │   strategy/  │
 └──────────────┘   └──────────────┘   └──────────────┘
        │
        └──> next improvement ──> back to step 1–4

 pipeline/ runs steps 2 → 6 for one config in configs/experiments/.
 Future: paper (Alpaca) ──> shadow (Binance data, no orders) ──> live (Binance), reusing strategy/.
```
