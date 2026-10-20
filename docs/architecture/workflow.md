# Trading Bot — Architecture & Workflow

Source of requirements: [idea_v2.md](idea_v2.md) (original: [archive/idea_v1.md](archive/idea_v1.md)). Data sources, universe and storage (v1): [archive/data.md](archive/data.md). Universe research: [research/AIChain.md](research/AIChain.md).

**Style:** modular monolith. It is one Python package and one deployable, split into modules with hard boundaries. Every module has a single public entry point (`api.py`). Any module can later be pulled out into its own service (for example `data/`) without rewriting the others.

**Tech stack:** Python 3.12, `uv`, `polars` + `pyarrow` (Parquet), `duckdb`, `alpaca-py`, `scikit-learn` (HistGradientBoosting), `optuna` (planned), `pydantic`, `typer`, `binance-sdk-stocks` (official Binance Stocks SDK, `Future`), `pytest`, `import-linter`, `ruff`.

---

## Part 1 — Folder Architecture

Folder-level view of the repository as of 2026-10-20. Only key files are shown. The v1 pipeline (AI-chain universe, 5-minute bars) is retired; its code is kept in the git tag `v1` and its results in [../experiments/v1/](../experiments/v1/).

```text
trading/
├── pyproject.toml              # deps + `trading` CLI entry point
├── .env.example                # Alpaca + Binance API keys
│
├── data/                       # gitignored — market data
│   ├── raw/nasdaq-100/         # membership.parquet (who was in the index, when) + bars_20min/ per series
│   ├── raw/calendar.parquet    # trading sessions and half-days
│   ├── clean/                  # (to build) cleaned bars ready for features
│   └── archive/                # v1 data, not used by the new pipeline
│
├── artifacts/                  # gitignored — run outputs (models, predictions, backtests, reports)
│
├── src/trading/                # one folder per workflow step (idea_v2.md); each exposes api.py
│   ├── data/                   # 1. DATA: Nasdaq-100 membership, Alpaca download, universe = members on each date
│   │   └── notebook/           # inspect_raw.ipynb: look at the raw data (each step keeps its notebooks here)
│   ├── features/               # 2. FEATURES: (to build)
│   ├── models/                 # 3. MODELS: (to build)
│   ├── strategy/               # 4. STRATEGY: (to build)
│   ├── backtest/               # 5. BACKTEST: Binance Stocks cost model; engine (to build)
│   ├── report/                 # 6. REPORT: (to build)
│   ├── shared/                 # config, storage paths (used by every step)
│   └── cli.py                  # `trading check-connections | data-sync`
│
└── docs/
    ├── architecture/           # idea_v2.md (requirements), workflow.md (this file), research/, archive/
    ├── experiments/            # experiment write-ups, grouped by version (v1/)
    └── book/                   # reading notes
```

**Universe rule:** a stock is tradable on a date only while it is a Nasdaq-100 member (`data.api.members_on`). Stocks that later left the index are kept for the period they were in it, so backtests are free of survivorship bias.

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

 Future: paper (Alpaca) ──> shadow (Binance data, no orders) ──> live (Binance), reusing strategy/.
```
