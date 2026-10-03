# Trading Bot — Architecture & Workflow

Source of requirements: [idea.md](idea.md). Data sources, universe and storage: [data.md](data.md). Universe research: [research/AIChain.md](research/AIChain.md).

**Style:** modular monolith. It is one Python package and one deployable, split into modules with hard boundaries. Every module has a single public entry point (`api.py`). Any module can later be pulled out into its own service (for example `live/` or `market_data/`) without rewriting the others.

**v1 scope:** ingest minute bars for about 50 AI-chain stocks, train RF/XGBoost/LightGBM models, and backtest them with realistic costs from **$2,000 starting on 2026-01-02**. Costs are modelled on **Binance Stocks**, the venue you trade on. Live trading there is `Future`; the broker interface it needs exists from day one.

**Data split by job:** **Alpaca** supplies historical data for training and backtesting. **Binance** supplies live market data and executes orders. The two meet at the **5-minute bar**, because Binance's live candle stream has no 1-minute interval (see *Binance Stocks API*).

**Tech stack:** Python 3.12, `uv`, `polars` + `pyarrow` (Parquet), `duckdb`, `alpaca-py`, `scikit-learn` (HistGradientBoosting; see *Implementation status*), `optuna` (planned), `pydantic`, `typer`, `binance-sdk-stocks` (official Binance Stocks SDK, `Future`), `pytest`, `import-linter`, `ruff`.

---

## Part 1 — Folder Architecture

```text
trading/
├── pyproject.toml                  # deps, CLI entry point, import-linter contracts (module boundaries)
├── .env.example                    # ALPACA_API_KEY/SECRET (data + paper), BINANCE_API_KEY/SECRET (live, Future)
├── README.md
│
├── configs/                        # every run is driven by config, not by code edits
│   ├── universe/
│   │   ├── ai_chain.yaml           # ~50 symbols: symbol, category, layer, active_from
│   │   └── benchmarks.yaml         # SPY, QQQ, SMH — market context features + comparison only
│   ├── data.yaml                   # feed (sip|iex), start date, adjustment=split, session hours, decision_timeframe: 5Min
│   ├── costs.yaml                  # fee schedule per venue (binance_stocks default, binance_perp, alpaca) + spread/slippage per liquidity tier
│   ├── account.yaml                # starting_cash: 2000, venue: binance_stocks, settlement + day-trade rules (venue-specific)
│   └── experiments/
│       ├── baseline_v1.yaml        # one file = one reproducible run (features, label, model, strategy, dates)
│       └── retrain_cadence.yaml    # Experiment: static vs monthly vs weekly retraining
│
├── data/                           # gitignored; owned by market_data/ and features/
│   ├── raw/bars_1m/month=2024-06.parquet   # one file per month, all symbols; raw/_manifest.json tracks completed months
│   ├── raw/calendar.parquet                # trading sessions and half-days (from Alpaca)
│   ├── raw/corporate_actions/
│   ├── clean/bars_5m/symbol=NVDA.parquet   # regular hours, full 5m grid, has_trade flag
│   ├── features/<feature_set>@<version>/symbol=.../part-0.parquet
│   └── catalog.duckdb              # SQL views over the parquet files for ad-hoc analysis
│
├── artifacts/                      # gitignored; owned by modeling/, backtest/, reporting/
│   ├── models/<run_id>/<fold_id>/{model.bin, metadata.json}
│   └── runs/<run_id>/
│       ├── config.snapshot.yaml    # exact config + git hash + data version → reproducibility
│       ├── predictions.parquet
│       ├── orders.parquet, fills.parquet, equity.parquet
│       └── report.html
│
├── src/trading/
│   ├── shared/                     # kernel: no business logic; the only module everyone may import
│   │   ├── config.py               # pydantic models that load and validate configs/*.yaml
│   │   ├── calendar.py             # NYSE sessions, half-days, holidays, minutes-to-close
│   │   ├── clock.py                # Clock protocol → SimClock (backtest) / WallClock (live)
│   │   ├── types.py                # Instrument, Bar, Signal, OrderIntent, Order, Fill, Position
│   │   ├── schemas.py              # column contracts for the DataFrames passed between modules
│   │   ├── storage.py              # parquet partition paths, atomic writes
│   │   └── logging.py
│   │
│   ├── universe/                   # WHICH stocks
│   │   ├── api.py                  # get_universe(name, as_of) -> list[Instrument]
│   │   ├── registry.py             # load yaml, apply active_from (IPO dates), liquidity filter
│   │   └── taxonomy.py             # categories/layers: semis, equipment, networking, … (Experiment)
│   │
│   ├── market_data/                # GET and STORE bars
│   │   ├── api.py                  # sync(universe, start, end); load_bars(symbols, start, end)
│   │   ├── providers/
│   │   │   ├── base.py             # MarketDataProvider protocol (Future: other vendors)
│   │   │   └── alpaca.py           # alpaca-py historical bars: paging, rate limits, retries
│   │   ├── ingest.py               # resumable backfill per symbol-month + incremental daily update
│   │   ├── quality.py              # gaps, duplicates, out-of-session bars, split sanity checks
│   │   ├── resample.py             # 1m → 5m bars (decision timeframe; must match Binance kline_5m)
│   │   └── store.py                # raw → clean parquet layout
│   │
│   ├── features/                   # X: what the model sees (point-in-time only)
│   │   ├── api.py                  # build(feature_set, bars) -> features frame
│   │   ├── registry.py             # @feature decorator; versioned feature sets
│   │   ├── price.py                # returns over 1/5/15/60m, volatility, gap, distance to VWAP
│   │   ├── volume.py               # volume z-score, dollar volume, volume surge
│   │   ├── session.py              # minute-of-day, minutes-to-close, day-of-week
│   │   ├── market_context.py       # SPY/QQQ/SMH returns and relative strength
│   │   ├── cross_sectional.py      # rank of each stock vs the universe at the same bar
│   │   └── category.py             # strength relative to own category (Experiment)
│   │
│   ├── labeling/                   # y: what the model predicts
│   │   ├── api.py                  # make_labels(bars, label_spec) -> labels frame
│   │   ├── forward_return.py       # return over horizon h, minus round-trip cost
│   │   └── triple_barrier.py       # take-profit / stop-loss / time-limit labels
│   │
│   ├── modeling/                   # LEARN
│   │   ├── api.py                  # train(dataset, model_spec) -> ModelArtifact; predict(...)
│   │   ├── dataset.py              # join features + labels, warm-up drop, NaN policy, class balance
│   │   ├── splits.py               # walk-forward folds with purge + embargo (no leakage)
│   │   ├── estimators/
│   │   │   ├── base.py             # Estimator protocol: fit / predict_proba / save / load
│   │   │   ├── baselines.py        # logistic regression, simple momentum rule
│   │   │   ├── random_forest.py
│   │   │   ├── xgboost_model.py
│   │   │   └── lightgbm_model.py
│   │   ├── tuning.py               # optuna, on validation folds ONLY
│   │   ├── metrics.py              # AUC, precision@top-k, information coefficient, calibration
│   │   └── artifacts.py            # save/load models + metadata (features, data range, params)
│   │
│   ├── strategy/                   # DECIDE: predictions → trade intents
│   │   ├── api.py                  # Strategy protocol: on_bar(ctx) -> list[OrderIntent]
│   │   ├── ml_signal.py            # probability threshold / top-k selection, entries + exits
│   │   ├── sizing.py               # equal weight, volatility target, fractional shares
│   │   └── rules.py                # no trades in first/last N min, cooldown, flat at close
│   │
│   ├── risk/                       # GUARD: veto anything unsafe or not allowed
│   │   ├── api.py                  # check(intents, portfolio) -> approved intents + rejections
│   │   ├── limits.py               # max % per position, max open positions, daily loss stop
│   │   └── compliance.py           # venue rules: when sale proceeds are reusable, day-trade limits
│   │
│   ├── portfolio/                  # ACCOUNT STATE
│   │   ├── api.py
│   │   └── ledger.py               # cash (settled/unsettled), positions, realized/unrealized PnL
│   │
│   ├── execution/                  # ACT: one Broker interface, several implementations
│   │   ├── api.py                  # Broker protocol: submit, cancel, positions, account
│   │   ├── simulated.py            # SimBroker: fill at next bar open + spread + slippage + fees
│   │   ├── cost_model.py           # reads configs/costs.yaml; Binance Stocks: max($0.35, 0.10% × order) per order
│   │   ├── alpaca_broker.py        # Future: paper trading (Alpaca paper API), same protocol as SimBroker
│   │   ├── shadow_broker.py        # Future: logs orders and fills them at live Binance quotes; no money moves
│   │   └── binance_broker.py       # Future: live via binance-sdk-stocks; tokenize=false, MARKET buy by notional
│   │
│   ├── backtest/                   # ORCHESTRATE simulation
│   │   ├── api.py                  # run_backtest(experiment_cfg) -> BacktestResult
│   │   ├── engine.py               # bar loop: decide every 5m bar, fill on next 1m bar → strategy → risk → broker → portfolio
│   │   └── walk_forward.py         # retrain every N (static|month|week), stitch out-of-sample
│   │
│   ├── reporting/                  # JUDGE
│   │   ├── api.py
│   │   ├── performance.py          # return, Sharpe, Sortino, max drawdown, turnover, cost drag
│   │   ├── benchmarks.py           # buy & hold equal-weight basket, SPY, QQQ, SMH
│   │   └── report.py               # html report per run + comparison table across runs
│   │
│   ├── live/                       # Future
│   │   ├── api.py
│   │   ├── stream.py               # Binance Stocks WebSocket: kline_5m, quote, tradingStatus, calendar, orderReport
│   │   ├── warmup.py               # at startup, load feature lookback history from Alpaca (historical REST)
│   │   ├── reconcile.py            # rebuild positions from Binance trade history; halt if ledger disagrees
│   │   ├── parity.py               # compare Binance 5m klines with Alpaca 5m bars for the same symbols/times
│   │   └── runner.py               # SAME engine with WallClock + live Broker (Alpaca paper | shadow | Binance)
│   │
│   └── cli.py                      # `trading data sync | features build | train | backtest | walk-forward | report`
│
├── notebooks/                      # exploration only; src/ never imports from here
├── tests/
│   ├── unit/<module>/
│   ├── integration/                # small fixture dataset, end-to-end pipeline
│   ├── leakage/                    # shuffle-future tests: features must not change when future bars change
│   └── architecture/               # import-linter contracts run in CI
└── docs/architecture/
    ├── idea.md
    ├── workflow.md
    └── research/
```

### Implementation status (2026-10-03)

The first full pass of Steps 0–9 is built and has been run end to end. Results: [../experiments/001-baseline.md](../experiments/001-baseline.md).

| Module | Status | Notes |
|---|---|---|
| `shared/`, `universe/` | Built | Config, secrets, storage paths, universe YAML |
| `market_data/` | Built | `providers/alpaca.py`, `ingest.py` (resumable, by month), `resample.py` (1m → clean 5m). `quality.py` is still a manual check. |
| `features/`, `labeling/` | Built | 52 features in `features/builders.py`; forward-return labels. Triple-barrier labels not built. |
| `modeling/` | Built (one estimator) | Everything is in `modeling/api.py`. No `tuning.py` yet. |
| `execution/cost_model.py` | Built | Binance Stocks fees + spread tiers |
| `backtest/` | Built | `engine.py` (simulator) and `experiment.py` (train → validate → holdout, resumable) |
| `reporting/` | Built | Metrics, benchmarks, HTML report |
| `strategy/`, `risk/`, `portfolio/` | **Not split out yet** | Their v1 logic (entry threshold, sizing, flat-by-close, cash ledger) lives inside `backtest/engine.py`. They must be extracted before live trading so backtest and live share them. |
| `live/`, `execution/*_broker.py` | Not built | `Future` |

CLI: `trading check-connections | data-sync | data-clean | experiment --name <cfg> | report`.

**What building it taught us:**

1. **LightGBM and XGBoost don't run on this machine.** Both need the OpenMP library `libomp`, and Homebrew no longer supports Intel Macs. v1 uses scikit-learn's `HistGradientBoostingRegressor`, the same histogram-boosting algorithm. Because every model sits behind `modeling/api.py`, either library can be swapped in on Linux or Apple Silicon.
2. **Raw data is stored one file per month, not per symbol and year.** A month is the unit of download, so this makes resume trivial. The clean layer is one 5-minute file per symbol. Clean 1-minute bars aren't stored, because nothing needs them: the open of the next 5-minute bar *is* the first 1-minute open after a decision.
3. **Alpaca returns whole-number prices as JSON integers.** Parsing must force floats, or some months fail.
4. **A failed month must be logged at once.** The first downloader hid one failure until all other months had finished.
5. **Every trained model's predictions are saved as soon as they exist.** Monthly retraining takes about an hour, and a crash in a later step once threw that away.
6. **Fees dominate.** Random entries lost about 30% in two months, almost all of it fees. Only the strictest entry threshold (predicted gain ≥ 0.6%) was profitable on validation, and the bot then trades on only about 1 day in 5.

### Module rules

1. **Public API only.** Module A may import only `trading.B.api` from module B, plus anything in `trading.shared`. Internals such as `trading.features.price` are private. `import-linter` contracts in `pyproject.toml` enforce this, and the check runs in `tests/architecture/`.
2. **One-way dependencies.** Arrows point toward the modules being depended on. Cycles are forbidden.

   ```text
   cli ─┬─> backtest ──> strategy ──> modeling ──> features ──> market_data ──> universe
        │        │                         └─────> labeling ───┘
        │        ├─────> risk ──> portfolio
        │        └─────> execution ──> portfolio
        ├─> reporting ──> (reads artifacts/runs only)
        └─> live (Future) ──> same as backtest, with WallClock + a live Broker
   all modules ──> shared
   ```
3. **Data contracts, not shared objects.** Modules exchange the dataclasses in `shared/types.py` or DataFrames whose columns are declared in `shared/schemas.py`.
4. **Each module owns its storage.** Only `market_data/` writes to `data/raw` and `data/clean`. Only `features/` writes to `data/features`. Other modules read through the owner's `api.py`.
5. **Backtest and live share code.** `strategy`, `risk`, `portfolio` and `execution.api` are identical in both modes. Only the `Clock` and the `Broker` implementation change. This is the main protection against "worked in backtest, fails live".

### Execution venue & costs

You trade on Binance, so Binance's fee schedule is what the backtest charges. Binance offers three ways to trade US stocks (as of Oct 2026):

| Product | What you hold | Fees (base tier) | Role here |
|---|---|---|---|
| **Binance Stocks** (default) | Real US shares. The broker is Nest Trading (Abu Dhabi), and **Alpaca** executes, clears and holds them. Fractional from $5. Some stocks trade 24/5. REST API trading since 2026-07-20. Not available to US persons. | "Zero commission", but a platform fee of **max($0.35, 0.10% of order value)** per order | v1 cost model; live broker (`Future`) |
| Binance stock perpetuals | USDT-margined futures on NVDA, META, GOOGL, MSFT, AMZN, … (list growing). Up to ~10× leverage, trade 24/7, funding every 8h. | 0.02% maker / 0.05% taker + funding | `Experiment`: allows shorting and costs less, but carries liquidation risk and the price can drift from the stock when the US market is closed |
| Binance bStocks | Tokens backed 1:1 by shares (NVDAB, MUB, …), USDT spot, 24/7 | spot fee tier | Not used: only a handful of our symbols, thin liquidity |

Binance Stocks orders go through Alpaca, so **Alpaca's historical SIP data is the right market data** for the backtest, and Alpaca's paper API is a close stand-in for paper trading.

**What the fee means for a $2,000 account:**

- A **$500 order** pays $0.50, which is 0.10% per side, so **≥ 0.20% per round trip** before spread and slippage.
- A **$100 order** pays the $0.35 minimum, which is **0.35% per side**. So `strategy/sizing.py` enforces a **minimum order size of $350**, and $2,000 supports at most ~5 open positions.
- **5 full-capital round trips per day** would cost ~$20/day in fees, about **1% of the account per day**. The model has to predict moves larger than ~0.25–0.3% over its horizon to be worth trading. That is why the label horizon is 15–60 min or longer, not 1 min.

### Binance Stocks API — what the bot can use

Checked against Binance's official SDK [`binance-sdk-stocks`](https://github.com/binance/binance-connector-python/tree/master/clients/stocks) v1.3.0 (2026-09-23) and the [Stocks API docs](https://developers.binance.com/en/docs/products/stocks/introduction). REST endpoints live under `/sapi/v1/equity/*`. The WebSocket base is `wss://nbstream.binance.com/equity`.

| Need | Binance mechanism | Notes |
|---|---|---|
| Live candles | WebSocket `<SYMBOL>@kline_<interval>` | Intervals: **5m, 1h, 1d, 1w, 1M only, no 1m**. Fields: OHLC, volume, trade count, VWAP, `x` = candle closed. Public, no API key. |
| Live bid/ask | WebSocket `<SYMBOL>@quote`; REST `GET /market/quote` | Best bid/ask with sizes. The stream pushes at most every 200 ms per symbol; the REST quote is up to ~5 s stale. Used for spread checks and shadow fills. |
| Last price, all symbols | WebSocket `price` | One snapshot every ~3 s |
| Halts / market phase | WebSocket `<SYMBOL>@tradingStatus`, `calendar`, `<SYMBOL>@tradability` | Halts, LULD pauses, short-sale restriction; session transitions. `risk/` blocks orders on halted symbols. |
| Historical candles | **None.** No REST klines endpoint. | History and the startup warm-up must come from Alpaca. This confirms your plan. |
| Symbol rules | REST `GET /market/exchangeInfo` | Per symbol: tradable, fractionable, `minNotional`, `maxNotional`, `minQty`, `stepSize`, `maxNumOrders`, extended/overnight support |
| Place order | REST `POST /order/place` | `MARKET` or `LIMIT`; `DAY` or `GTC`; session `RTH` / `EXTENDED` / `24H`. **MARKET BUY uses `notional` ($)**, MARKET SELL uses `quantity`. LIMIT needs `price` + `quantity` + `tradingSession`. Fractional orders allowed; fractional `GTC` needs `EXTENDED`/`24H`. Limit: **200 orders/min**. |
| Cancel | REST `POST /order/cancel`, `/order/cancel-all` | |
| Order & fill updates | WebSocket `<listenKey>@orderReport` (listen key from REST, **60-min TTL, must be renewed**) | Status, filled qty, avg price, total cost incl. fee. Uses lowercase `buy`/`sell` and asset code `EQ_NVDA`, so the adapter normalises them. |
| Order / fill history | REST `GET /order/history`, `/order/detail`, `/trade/history` | Includes the real fee charged, which calibrates `costs.yaml`. |
| Balance & positions | **No endpoint in the Stocks API** | `portfolio/ledger.py` tracks positions from our own fills. `live/reconcile.py` rebuilds them from `/trade/history` at startup and every few minutes, and the bot halts on any mismatch. Holdings may also appear in the Binance wallet API as `EQ_*` assets (unverified). |
| Testnet / sandbox | **None** (the SDK ships only production URLs) | Testing goes Alpaca paper → shadow mode on Binance → small live (Steps 10–12) |
| One-time setup | REST `POST /account/disclaimer` | Must be signed once, or orders are rejected (error 486410) |

**Things `binance_broker.py` must get right:**

- **`tokenize` defaults to `true`.** For symbols that have a bStock (NVDA, MU, …), a buy is converted into the token (NVDAB) on settlement unless the order sends **`tokenize=false`**. The bot always sends `false` so it holds real shares.
- **`walletType`:** buys pay from the `CARD` wallet by default, and sells always settle to `CARD`. Keep the bot's money in that one wallet so it can reuse sale proceeds.
- **Order type:** v1 uses **MARKET, DAY, regular hours**. Buys are sized by `notional` (e.g. $500), and sells close the exact `quantity` held. Marketable LIMIT orders, which cap slippage, are an `Experiment`.
- **Before every order,** check the rules from `exchangeInfo` (`minNotional`, `stepSize`, tradable). Refresh them daily.

**Why decisions use 5-minute bars:** live candles from Binance are 5 m at the smallest. The model must see the same kind of bar live as in training, so:

- 1-minute Alpaca data is kept and **resampled to 5-minute bars** (`market_data/resample.py`) for features and decisions.
- 1-minute bars are still used in the backtest to simulate *when* an order fills (the next minute after the 5-minute bar closes).
- With ~0.2% round-trip fees, a 5-minute decision cycle loses almost nothing. Edges smaller than that couldn't pay for themselves anyway.
- `live/parity.py` records Binance 5m klines for a few weeks and compares them with Alpaca's 5m bars. Training and live data must agree on OHLC and volume within a small tolerance before trusting the model live.

**Still unverified** (ask Binance support, or test with tiny orders): whether sale proceeds can be reused the same day or wait for T+1 settlement; any day-trade limits; whether holdings show in the wallet API.

### Where `Future` / `Experiment` items plug in

| Item (from idea.md) | Plugs in at | Change needed |
|---|---|---|
| `Experiment` stock categories | `configs/universe/ai_chain.yaml` (`category` field present from v1), `universe/taxonomy.py`, `features/category.py` | Turn the feature set on in the experiment config |
| `Experiment` data from 2021 | `configs/data.yaml: start` | Config only |
| `Experiment` monthly → weekly retraining | `configs/experiments/retrain_cadence.yaml`, `backtest/walk_forward.py` | Config only |
| `Future` other sectors | new `configs/universe/<sector>.yaml` | Config only, then retrain |
| `Future` other data vendors | `market_data/providers/<vendor>.py` | New provider implementing `base.py` |
| `Future` paper → shadow → live trading | `execution/alpaca_broker.py` (paper), `execution/shadow_broker.py`, `execution/binance_broker.py` (live), `live/` | New Brokers + runner; strategy code untouched |
| `Experiment` Binance stock perpetuals (shorting, lower fees) | `configs/costs.yaml` venue `binance_perp`, `portfolio/ledger.py` (funding payments), `risk/limits.py` (leverage, liquidation) | New cost profile + funding/leverage handling |

---

## Part 2 — Workflow

```text
 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ 1 Universe  │──>│ 2 Ingest     │──>│ 3 Clean & QA │──>│ 4 Features   │──┐
 └─────────────┘   └──────────────┘   └──────────────┘   └──────────────┘  │
                                              │          ┌──────────────┐  │
                                              └─────────>│ 5 Labels     │──┤
                                                         └──────────────┘  │
 ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐  │
 │ 9 Report    │<──│ 8 Walk-fwd   │<──│ 7 Backtest   │<──│ 6 Train      │<─┘
 └─────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
        │
        └──> 10 Paper (Alpaca) ──> 11 Shadow (Binance data, no orders) ──> 12 Live (Binance)   (Future)
```

### Step 0 — Setup

Set up the environment, credentials and configs.

| | |
|---|---|
| **Where** | `pyproject.toml`, `.env.example`, `configs/*`, `src/trading/shared/config.py` |
| **Output** | Validated config objects. Bad config fails fast at startup. |

### Step 1 — Define the universe

Choose the stocks the bot is allowed to trade.

| | |
|---|---|
| **Where** | `configs/universe/ai_chain.yaml`, `configs/universe/benchmarks.yaml`, `universe/registry.py`, `universe/api.py` |
| **Input** | Ticker list from `research/AIChain.md` |
| **Output** | `list[Instrument]` for a given date |
| **Rules** | Every symbol carries `category` and `layer` from day one, even though v1 doesn't use them as features. `active_from` stops the backtest from trading a stock before its IPO (ARM 2023-09, GEV 2024-04, CRWV 2025-03). Drop symbols that Alpaca doesn't list or that trade thinly (see notes on SIEGY). |

```yaml
# configs/universe/ai_chain.yaml (excerpt)
- {symbol: NVDA, category: semiconductor, layer: compute}
- {symbol: ASML, category: semiconductor, layer: equipment}
- {symbol: ANET, category: networking,    layer: networking}
- {symbol: ARM,  category: semiconductor, layer: compute, active_from: 2023-09-14}
- {symbol: VRT,  category: infrastructure, layer: power_cooling}
```

### Step 2 — Ingest market data

Download 1-minute bars, 2021-01-01 → now, for the universe plus the benchmarks.

| | |
|---|---|
| **Where** | `market_data/providers/alpaca.py`, `market_data/ingest.py`, `market_data/store.py`, CLI `trading data sync` |
| **Output** | `data/raw/bars_1m/symbol=*/year=*/` and `data/raw/corporate_actions/`; 5-minute bars built from them by `market_data/resample.py` |
| **Rules** | Use `adjustment=split` so splits are handled (NVDA, AVGO, SMCI and LRCX all split 10:1 in 2024). Dividends are left unadjusted; see [data.md](data.md) §1.4 for why. Prefer the **SIP** feed for history, because IEX-only bars are sparse for smaller names. Make ingestion resumable per symbol-month and idempotent, so re-running doesn't create duplicates. |
| **Size** | About 1,450 trading days × 390 min × ~53 symbols ≈ 30M rows. That is ~1–3 GB as Parquet and fits easily on a laptop. |

### Step 3 — Clean & quality-check

| | |
|---|---|
| **Where** | `market_data/quality.py`, `shared/calendar.py`, `market_data/store.py` |
| **Output** | `data/clean/bars_1m/` plus a QA report (gaps per symbol per day) |
| **Rules** | Keep regular session only (09:30–16:00 ET) and handle half-days. Mark minutes with no trade as missing rather than forward-filling them as real trades. Flag days with large unexplained jumps, which are usually missed splits. Never drop data in a way that depends on future values. |

### Step 4 — Build features

| | |
|---|---|
| **Where** | `features/*.py`, `features/registry.py`, CLI `trading features build --set v1` |
| **Output** | `data/features/<set>@<version>/` with one row per (symbol, 5-minute bar) |
| **Rules** | A feature at bar *t* may use only bars ≤ *t*. `tests/leakage/` checks this by perturbing future bars and asserting that past features don't change. Feature sets are versioned, so a model artifact always records which set it used. Prefer **scale-free** features (returns, z-scores, ranks), so one pooled model can learn across all 50 stocks instead of training 50 small models. |

### Step 5 — Build labels

| | |
|---|---|
| **Where** | `labeling/forward_return.py`, `labeling/triple_barrier.py` |
| **Output** | Label frame aligned to the feature rows |
| **Rules** | Labels are **net of costs**. "Up" means the forward return over horizon *h* is greater than the round-trip cost from `configs/costs.yaml`, not merely > 0. The horizon *h* (15 / 30 / 60 min, or end of day) is an experiment parameter. Try triple-barrier labels too, since they match how the strategy actually exits (take-profit / stop-loss / timeout). |

### Step 6 — Assemble dataset, split, train

| | |
|---|---|
| **Where** | `modeling/dataset.py`, `modeling/splits.py`, `modeling/estimators/*`, `modeling/tuning.py`, `modeling/metrics.py`, CLI `trading train` |
| **Output** | `artifacts/models/<run_id>/<fold_id>/` |
| **Rules** | Splits are chronological only, with **purge + embargo** at each boundary so that labels spanning the boundary can't leak. Train the baselines first; RF/XGBoost/LightGBM must beat them. Hyperparameters and feature choices are tuned on the validation period only. |

```text
2021-01 ─────────────── 2024-12 │ 2025-01 ──────── 2025-12 │ 2026-01 ──────────── now
   TRAIN (walk-forward folds)   │  VALIDATION: tune, pick  │  HOLDOUT: the $2,000 test
                                │  features / model / h    │  run once per frozen design
```

The 2026 holdout is the "how much would $2,000 have earned" number. If we keep tuning until 2026 looks good, it stops being a test and the number becomes fiction. Design decisions are frozen on 2025 before 2026 is run.

### Step 7 — Backtest (simulation)

Replay the market bar by bar through the same components a live bot would use. Decisions happen on 5-minute bars, and fills are simulated on 1-minute bars.

| | |
|---|---|
| **Where** | `backtest/engine.py`, `strategy/*`, `risk/*`, `portfolio/ledger.py`, `execution/simulated.py`, `execution/cost_model.py`, CLI `trading backtest --exp baseline_v1` |
| **Output** | `artifacts/runs/<run_id>/{predictions, orders, fills, equity}.parquet` |

What happens on each simulated 5-minute bar:

```text
for bar t in 5m_bars(2026-01-02 … now):               # bar t closes at t+5m
    clock.advance(t + 5m)                              # shared/clock.py (SimClock)
    probs   = predictions[t]                           # precomputed per fold (vectorised, point-in-time)
    intents = strategy.on_bar(ctx(t, probs, portfolio))# strategy/ml_signal.py, sizing.py, rules.py
    orders  = risk.check(intents, portfolio)           # risk/limits.py, risk/compliance.py
    fills   = broker.submit_and_fill(orders,           # execution/simulated.py: fill at OPEN of the 1m bar
                  bar_1m_open[t + 5m])                 #   after the decision, + half-spread + slippage + fees
    portfolio.apply(fills); portfolio.mark(close[t])   # portfolio/ledger.py
```

| | |
|---|---|
| **Rules** | Decide when 5-minute bar *t* closes, then fill at the open of the next 1-minute bar. Filling at the close of the bar that generated the signal is the most common fake-profit bug. Costs are always on, and every run is also repeated at **2× costs** as a stress test. Fees follow the venue in `account.yaml` (default Binance Stocks: max($0.35, 0.10%) per order; see *Execution venue & costs*). Settlement and day-trade rules are switches in `account.yaml`, so the backtest can be re-run once Binance's rules are confirmed. |

### Step 8 — Walk-forward retraining (`Experiment`: monthly vs weekly)

| | |
|---|---|
| **Where** | `backtest/walk_forward.py`, `configs/experiments/retrain_cadence.yaml`, CLI `trading walk-forward --exp retrain_cadence` |
| **Output** | One stitched out-of-sample equity curve per cadence |
| **How** | For each period *k* (month or week), train on all data before period *k* minus the embargo, trade period *k*, then roll forward. Compare **static** (train once on ≤2025) vs **monthly** vs **weekly**. Weekly retraining is only worth its compute and complexity if it beats monthly *after costs* and on more than one period. |

### Step 9 — Report & judge

| | |
|---|---|
| **Where** | `reporting/performance.py`, `reporting/benchmarks.py`, `reporting/report.py`, CLI `trading report <run_id>` |
| **Output** | `artifacts/runs/<run_id>/report.html` and a cross-run comparison table |
| **Metrics** | Final equity from $2,000, total return, Sharpe, Sortino, max drawdown, trade count, win rate, avg trade net of cost, cost drag (fees ÷ gross PnL), exposure time |
| **Benchmarks** | The bot must beat **buy & hold of the same equal-weight basket** over the same dates, plus SPY, QQQ and SMH. "Made a profit" isn't enough if holding the basket made more. |

### Step 10 — Paper trading on Alpaca (`Future`)

Check that the full live loop works, with no real money.

| | |
|---|---|
| **Where** | `live/runner.py`, `live/warmup.py`, `execution/alpaca_broker.py` (paper endpoint), `shared/clock.py` (WallClock) |
| **Rules** | Binance Stocks executes through Alpaca, so Alpaca paper is the closest free stand-in. Binance has no stock testnet. `cost_model.py` adds Binance fees, because Alpaca paper charges none. Strategy, risk and portfolio code is the same as Step 7. |

### Step 11 — Shadow mode on Binance (`Future`)

The real live pipeline: Binance data in, Binance-shaped orders out, but nothing is sent.

| | |
|---|---|
| **Where** | `live/stream.py` (Binance kline_5m + quote + tradingStatus), `live/parity.py`, `execution/shadow_broker.py` |
| **Rules** | Run for 2–4 weeks. Each order is "filled" at the live Binance bid/ask, plus fees. Pass criteria: (1) `parity.py` shows Binance 5m bars match Alpaca's; (2) shadow PnL is within a set tolerance of what the backtest gives for the same days; (3) no missed bars, reconnect failures or halted-symbol orders in the logs. |

Live loop (shadow and live use the same loop):

```text
startup:  live/warmup.py loads lookback history from Alpaca → features ready
          execution/binance_broker.py refreshes exchangeInfo; live/reconcile.py rebuilds positions
on kline_5m event with x=true (bar closed):                  # live/stream.py
    features  = features.api.update(bar)                     # same code as Step 4
    probs     = modeling.api.predict(features)
    intents   = strategy.on_bar(ctx)                         # same code as Step 7
    orders    = risk.check(intents, portfolio)               # + tradingStatus halts, exchangeInfo rules
    broker.submit(orders)                                    # shadow: log + fill at quote | live: POST /order/place
on orderReport event:                                        # live only
    portfolio.apply(fill)                                    # ledger; reconcile.py cross-checks periodically
```

### Step 12 — Live trading on Binance (`Future`)

| | |
|---|---|
| **Where** | `live/runner.py` with `execution/binance_broker.py` (`binance-sdk-stocks`), `BINANCE_API_KEY` |
| **Gate** | Go live only after (1) holdout + walk-forward results beat the benchmarks at 2× costs, (2) Alpaca paper trading matched the simulation, and (3) shadow mode passed. Start with a fraction of the $2,000. Have a kill switch (`POST /order/cancel-all` + stop the runner), a daily loss limit, and alerting in place first. |
| **Keys** | Create an API key with only *Reading* and *Spot/Stocks trading* permissions. **Never enable withdrawals**, and restrict the key to your server's IP. |
