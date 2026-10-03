# Data

Everything the bot learns from and trades on: where it comes from, which stocks, which dates, how it is stored, and what can go wrong with it.

Related: [workflow.md](workflow.md) (Steps 1–3 use this data) · [research/AIChain.md](research/AIChain.md) (where the stock list comes from).

## At a glance

| | |
|---|---|
| **Provider** | Alpaca Market Data API v2 (`alpaca-py`) |
| **Live data + execution** | Binance Stocks: WebSocket 5m candles, quotes, halts; REST orders. It has no historical candles, so history always comes from Alpaca. Binance's orders are executed and cleared by Alpaca, so it is the same market. Details: [workflow.md](workflow.md) → *Binance Stocks API*. |
| **Feed** | SIP: consolidated trades from all US exchanges |
| **Granularity** | 1-minute bars stored (main) · resampled to **5-minute bars for decisions**, to match Binance's live candles · 1-day bars (benchmarks, sanity checks) |
| **Universe** | 50 AI-chain stocks + 3 context/benchmark ETFs |
| **Date range** | 2021-01-04 → today, refreshed daily after the close |
| **Price adjustment** | Split-adjusted (`adjustment=split`) |
| **Trading hours used** | Regular session 09:30–16:00 ET. Extended hours are stored but not traded. |
| **Size** | ≈ 30M regular-session bars (~1,440 days × 390 min × 53 symbols), ~1–3 GB as Parquet |
| **Storage** | `data/` (gitignored), Parquet partitioned by `symbol/year`, DuckDB views on top |
| **Owner module** | `src/trading/market_data/`. Only this module writes `data/raw` and `data/clean`. |

---

## 1. Sources

### 1.1 Alpaca endpoints we use

| Data | Endpoint | Used for | Module |
|---|---|---|---|
| Minute bars | `GET data.alpaca.markets/v2/stocks/bars` `timeframe=1Min` | Features, labels, backtest fills | `market_data/providers/alpaca.py` |
| Daily bars | same, `timeframe=1Day` | Buy-and-hold benchmarks, QA cross-check vs minute bars | same |
| Corporate actions | Alpaca corporate-actions endpoint | Detect splits, spin-offs and symbol changes | `market_data/ingest.py` |
| Assets | `GET api.alpaca.markets/v2/assets/{symbol}` | Is it tradable? fractionable? exchange? (Re-check against Binance Stocks' list before going live) | `universe/registry.py` |
| Market calendar | `GET api.alpaca.markets/v2/calendar` | Trading days, half-days (cross-checked with `exchange_calendars`) | `shared/calendar.py` |
| Quotes (bid/ask) | `GET data.alpaca.markets/v2/stocks/quotes` | `Experiment`: sample a few days per quarter to measure real spreads → `configs/costs.yaml` | `market_data/` (later) |
| Recent history for live warm-up | same bars endpoint | `Future`: at bot startup, load feature lookback | `live/warmup.py` |

### 1.2 Bar request parameters

```text
symbols     = up to ~50 per request (comma-separated)
timeframe   = 1Min
start, end  = RFC-3339, fetched one calendar month per call (resumable unit)
feed        = sip
adjustment  = split
limit       = 10000            # max bars per page; follow next_page_token until empty
sort        = asc
asof        = (default: today) # maps old tickers to current ones, e.g. FB → META
```

Alpaca returns these fields per bar: `t` (bar **start** time, UTC), `o`, `h`, `l`, `c`, `v` (volume), `n` (trade count), `vw` (volume-weighted average price).

### 1.3 Plan limits (free "Basic" plan)

- Historical SIP data is available, except for the most recent ~15 minutes. That doesn't matter for backtesting.
- Real-time streaming on the free plan is IEX only. We don't need it, because live data comes from Binance's stream.
- The rate limit is about 200 requests/min. The full backfill is roughly 3–6k paged requests, so **about 15–30 minutes** one time. Daily refreshes take seconds.
- Alpaca's data terms are for our own use. The `data/` folder is gitignored and never shared.

### 1.4 Decisions on the source

| Decision | Why |
|---|---|
| **SIP, not IEX** | IEX handles only a small slice of US volume. Its minute bars have many empty minutes for mid-caps such as AMKR, ASX and FN, which distorts volume features and fill simulation. |
| **`adjustment=split`, not `all`** | With `all`, every quarterly dividend changes the whole price history, so we would have to re-download constantly and old backtests would stop being reproducible. Splits are rare (see §5). Dividends on these names are under ~1–2%/yr and don't matter for intraday trading. If we later hold overnight across ex-dividend dates, we add dividends as a separate cash flow in `portfolio/`. |
| **Store extended hours, trade regular hours only** | Storing pre- and post-market bars costs almost nothing, and pre-market gaps may be a useful feature later. Liquidity outside the regular session is thin and spreads are wide, so the bot doesn't trade there. |

### 1.5 Live data from Binance (`Future`)

| Stream | Used for | Module |
|---|---|---|
| `<SYMBOL>@kline_5m` (5m is the smallest interval) | Feature updates and decisions when a candle closes (`x=true`) | `live/stream.py` |
| `<SYMBOL>@quote` | Spread check before ordering; shadow-mode fills | `live/stream.py`, `execution/shadow_broker.py` |
| `<SYMBOL>@tradingStatus`, `calendar` | Block orders on halted symbols; session open/close | `risk/` |
| `<listenKey>@orderReport` | Fills → ledger | `execution/binance_broker.py` |

Binance's live 5m candles are recorded to `data/live/klines_5m/` (owned by `live/`) and compared with Alpaca 5m bars by `live/parity.py`. If they disagree on OHLC or volume beyond tolerance, the model was trained on different data than it sees live, and the bot doesn't go live.

---

## 2. Universe — 50 stocks

The source of truth is `configs/universe/ai_chain.yaml`. The table below mirrors it.

- **layer** = position in the AI chain, following `research/AIChain.md`. This is the fine-grained `Experiment` category.
- **category** = a coarser group: `semis`, `hardware`, `infra`, `cloud_software`.
- **active_from** = first day the stock trades. The backtest ignores a stock before this date plus a feature warm-up of about 20 trading days.
- **source**: `idea` = your list in idea.md, `research` = AIChain.md, `added` = my suggestion (confirm or remove).

| # | Symbol | Company | layer | category | active_from | source |
|---|---|---|---|---|---|---|
| 1 | NVDA | NVIDIA | compute | semis | — | idea |
| 2 | AMD | Advanced Micro Devices | compute | semis | — | idea |
| 3 | AVGO | Broadcom | compute | semis | — | idea |
| 4 | MRVL | Marvell | compute | semis | — | idea |
| 5 | QCOM | Qualcomm | compute | semis | — | idea |
| 6 | INTC | Intel | compute | semis | — | idea |
| 7 | ARM | Arm Holdings (ADR) | compute | semis | 2023-09-14 | idea |
| 8 | MPWR | Monolithic Power | compute | semis | — | added: power-delivery chips on AI GPU boards |
| 9 | SNPS | Synopsys | eda | semis | — | idea |
| 10 | CDNS | Cadence | eda | semis | — | idea |
| 11 | ASML | ASML (NY shares) | equipment | semis | — | idea |
| 12 | AMAT | Applied Materials | equipment | semis | — | idea |
| 13 | LRCX | Lam Research | equipment | semis | — | idea |
| 14 | KLAC | KLA | equipment | semis | — | idea |
| 15 | TER | Teradyne | equipment | semis | — | added: tests AI chips and HBM |
| 16 | TSM | TSMC (ADR) | foundry_packaging | semis | — | idea |
| 17 | ASX | ASE Technology (ADR) | foundry_packaging | semis | — | idea |
| 18 | AMKR | Amkor | foundry_packaging | semis | — | idea |
| 19 | MU | Micron | memory | semis | — | idea |
| 20 | ANET | Arista Networks | networking_optical | hardware | — | idea |
| 21 | COHR | Coherent | networking_optical | hardware | — | idea (see §5.2) |
| 22 | LITE | Lumentum | networking_optical | hardware | — | idea |
| 23 | FN | Fabrinet | networking_optical | hardware | — | idea |
| 24 | CIEN | Ciena | networking_optical | hardware | — | added: data-center interconnect optics |
| 25 | CRDO | Credo | networking_optical | hardware | 2022-01-27 | added: AI cluster cables |
| 26 | ALAB | Astera Labs | networking_optical | hardware | 2024-03-20 | added: PCIe/CXL connectivity in AI servers |
| 27 | DELL | Dell Technologies | servers | hardware | — | idea |
| 28 | SMCI | Super Micro Computer | servers | hardware | — | idea |
| 29 | HPE | Hewlett Packard Enterprise | servers | hardware | — | idea |
| 30 | CLS | Celestica | servers | hardware | — | added: AI switches/servers for hyperscalers |
| 31 | VRT | Vertiv | power_cooling | infra | — | research |
| 32 | ETN | Eaton | power_cooling | infra | — | research |
| 33 | GEV | GE Vernova | power_cooling | infra | 2024-04-02 | research |
| 34 | TT | Trane Technologies | power_cooling | infra | — | research |
| 35 | EQIX | Equinix | data_center | infra | — | research |
| 36 | DLR | Digital Realty | data_center | infra | — | research |
| 37 | CRWV | CoreWeave | data_center | infra | 2025-03-28 | research |
| 38 | MSFT | Microsoft | cloud | cloud_software | — | research |
| 39 | AMZN | Amazon | cloud | cloud_software | — | research |
| 40 | GOOGL | Alphabet (Class A) | cloud | cloud_software | — | research |
| 41 | META | Meta Platforms | cloud | cloud_software | — | research (was FB) |
| 42 | ORCL | Oracle | cloud | cloud_software | — | research |
| 43 | PLTR | Palantir | ai_software | cloud_software | — | research |
| 44 | SNOW | Snowflake | ai_software | cloud_software | — | research |
| 45 | DDOG | Datadog | ai_software | cloud_software | — | research |
| 46 | MDB | MongoDB | ai_software | cloud_software | — | research |
| 47 | NET | Cloudflare | ai_software | cloud_software | — | research |
| 48 | NOW | ServiceNow | ai_software | cloud_software | — | research |
| 49 | CRM | Salesforce | ai_software | cloud_software | — | research |
| 50 | ADBE | Adobe | ai_software | cloud_software | — | research |

Counts: semis 19 · hardware 11 · infra 7 · cloud_software 13.

`—` in **active_from** means the stock was trading before 2021-01-04. The IPO and listing dates are from memory, so `universe/registry.py` cross-checks them against each symbol's first available bar.

### 2.1 Excluded, and why

| Symbol(s) | Reason |
|---|---|
| SIEGY | OTC-traded Siemens ADR. It is thinly traded, and EDA software is a small part of Siemens. It's a poor proxy for EDA and expensive to trade. |
| 8035.T, 005930.KS, 000660.KS, 0992.HK, 6669.TW, 2382.TW, 2317.TW, SU.PA, ENR.DE | Not listed on US exchanges, so neither Alpaca data nor Binance Stocks covers them. SK Hynix and Samsung (HBM) are partly represented by MU. |
| CAT, CMI, JCI | AI data centers are a small share of their revenue, so their price moves are mostly driven by other businesses. |

### 2.2 Context & benchmark symbols (not traded)

| Symbol | What | Used as |
|---|---|---|
| SPY | S&P 500 ETF | Market regime feature + benchmark |
| QQQ | Nasdaq-100 ETF | Tech regime feature + benchmark |
| SMH | VanEck Semiconductor ETF | Sector regime feature + benchmark |

The fourth benchmark is a **buy & hold equal-weight basket of the 50 stocks**, computed in `reporting/benchmarks.py` from daily bars. No extra download is needed.

### 2.3 Known bias

The list was chosen in 2026 by knowing which companies became AI winners. A backtest on it will look better than a strategy picked in 2021 would have done. We can't fully remove this effect. What we can do is judge the bot only **relative to the buy-and-hold basket of the same 50 stocks**, which shares the bias.

---

## 3. Time periods

```text
2021-01-04 ─────────────────── 2024-12-31 │ 2025-01-02 ──────── 2025-12-31 │ 2026-01-02 ─────── today
   TRAIN  (~1,005 days)                   │  VALIDATION (~250 days)      │  HOLDOUT ($2,000 test, ~190 days)
   includes 2022 bear market, 2023–24 AI  │  tune features, model, label │  run once per frozen design
   rally                                  │  horizon                     │
```

- **2021 start:** I agree with your choice. It covers a bull run, the 2022 drawdown and the AI rally, so the model sees more than one market regime. Going back further would add mostly pre-AI behaviour.
- Feature warm-up uses data from the start of the range, so the first ~20 trading days of 2021 produce features but no training rows.

---

## 4. Storage

### 4.1 Layout

```text
data/
├── raw/                                    # exactly what Alpaca returned (split-adjusted), never edited
│   ├── bars_1m/symbol=NVDA/year=2024/part-0.parquet
│   ├── bars_1d/symbol=NVDA/part-0.parquet
│   ├── corporate_actions/part-0.parquet
│   └── _manifest.parquet                   # one row per (symbol, month) download → resumable
├── clean/                                  # regular session, full minute grid, QA flags
│   └── bars_1m/symbol=NVDA/year=2024/part-0.parquet
├── features/<feature_set>@<version>/…      # owned by features/
├── reports/qa/<date>.html                  # data quality report per sync
└── catalog.duckdb                          # views: raw_bars_1m, clean_bars_1m, bars_1d, …
```

### 4.2 Schemas

The schemas are declared in `src/trading/shared/schemas.py` and checked on every write.

**`raw/bars_1m`**

| column | type | notes |
|---|---|---|
| `symbol` | string | current ticker |
| `ts` | timestamp[UTC] | bar **start**; the bar covers `[ts, ts + 1 min)` |
| `open`, `high`, `low`, `close` | float64 | split-adjusted |
| `volume` | float64 | split-adjusted (can be fractional after adjustment) |
| `trade_count` | int64 | |
| `vwap` | float64 | |

**`clean/bars_1m`** adds:

| column | type | notes |
|---|---|---|
| `session_date` | date | trading date in America/New_York |
| `minute_idx` | int16 | 0 = 09:30 bar … 389 = 15:59 bar (209 on half-days) |
| `has_trade` | bool | `false` = no trades that minute → OHLC null, volume 0 |
| `qa_flag` | string? | e.g. `price_jump`, `zero_volume_day` (null if clean) |

The clean layer has **one row per minute of every regular session**, even when nothing traded. Empty minutes stay empty (`has_trade=false`). We never forward-fill them into fake trades.

**`raw/_manifest`**: `symbol, month, feed, adjustment, rows, downloaded_at, sha256, status`. A re-run skips months with `status=ok` and retries the rest.

### 4.3 Time rule (leakage guard)

A bar stamped `ts = 10:00` holds trades from 10:00:00–10:00:59. It is **known at 10:01:00**. A decision based on it can fill no earlier than the **10:01 bar's open**. `backtest/engine.py` and `tests/leakage/` both enforce this.

### 4.4 Access

Other modules never read the Parquet files directly. They call:

```python
from trading.market_data.api import load_bars
bars = load_bars(symbols=["NVDA", "AMD"], start="2025-01-02", end="2025-12-31",
                 timeframe="1Min", layer="clean")          # -> polars.DataFrame
```

For ad-hoc analysis, use `duckdb data/catalog.duckdb` → `SELECT … FROM clean_bars_1m WHERE symbol = 'NVDA'`.

---

## 5. Corporate actions & symbol history

### 5.1 Splits in range (examples QA must reproduce)

These come from memory and serve as a test fixture. The corporate-actions feed is the source of truth, and `quality.py` flags any overnight price ratio near a split ratio that the feed doesn't explain.

| Symbol | Split | Effective |
|---|---|---|
| NVDA | 4:1 | 2021-07-20 |
| ANET | 4:1 | 2021-11-18 |
| AMZN | 20:1 | 2022-06-06 |
| GOOGL | 20:1 | 2022-07-18 |
| NVDA | 10:1 | 2024-06-10 |
| AVGO | 10:1 | 2024-07-15 |
| SMCI | 10:1 | 2024-10-01 |
| LRCX | 10:1 | 2024-10-03 |
| ANET | 4:1 | 2024-12-04 |

**When a new split happens:** split-adjusted history changes for that symbol. The daily sync detects the split in corporate actions and re-downloads **that symbol's** full history. Every run snapshot records the data download date, so old results can be explained.

### 5.2 Symbol changes & special cases

| Case | Risk | Handling |
|---|---|---|
| **FB → META** (2022-06-09) | Querying `META` for 2021 might return nothing. | Alpaca's `asof` parameter maps the current ticker to its history. QA checks there's no gap around 2022-06-09. |
| **COHR**: II-VI (`IIVI`) bought the old Coherent Inc. (`COHR`) in mid-2022, renamed itself Coherent Corp., then took the `COHR` ticker. | For 2021–mid-2022, `COHR` may return the **old, different company**. | Verify which entity `asof` returns. We want the II-VI history, since that is today's company. If it can't be fixed, set `active_from` to the date of the switch. |
| **DELL** spun off VMware (2021-11-01) | The price drops on the spin-off date, and split-only adjustment doesn't remove the drop. | QA flags it. Labels that span 2021-11-01 for DELL are dropped. |
| **PLTR** moved NYSE → Nasdaq (2024-11) | None for bars (same ticker). | Nothing. |

---

## 6. Data quality checks

These run in `market_data/quality.py` after every sync. Results go to `data/reports/qa/<date>.html`. A **fail** blocks feature building. A **warn** only reports.

| Check | Level |
|---|---|
| Schema & types match `shared/schemas.py` | fail |
| No duplicate `(symbol, ts)` | fail |
| `low ≤ open, close ≤ high`, prices > 0, volume ≥ 0 | fail |
| Every trading day in the calendar has bars for every active symbol | fail if a whole day is missing |
| Share of empty regular-session minutes per symbol-day | warn if > 5% (expected near 0 for NVDA, higher for FN or ASX) |
| Overnight price ratio ≈ 2, 3, 4, 10, 20… not explained by a corporate action | fail (missed split) |
| Daily bar close vs last 1-minute close of the day | warn if the gap is > 0.5% (the closing auction can differ slightly) |
| Bars outside 04:00–20:00 ET | fail |
| First bar date vs `active_from` in the universe file | warn |

---

## 7. Refresh

| When | What | Command |
|---|---|---|
| Once | Full backfill 2021-01-04 → today | `trading data sync --full` |
| Each trading day, after 16:15 ET | Fetch the last 3 trading days (overlap covers late corrections), upsert, rebuild clean data for affected days, run QA | `trading data sync` |
| On a split or new symbol | Re-download that symbol's full history | handled automatically by `sync` |
| `Future` | Live stream for paper/live trading | `live/stream.py` |

---

## 8. Open decisions

1. **Six `added` names** (MPWR, TER, CIEN, CRDO, ALAB, CLS): keep them, or swap for others?
2. **Software names** (PLTR → ADBE) behave differently from hardware. Should they be in v1, or should v1 be hardware/infrastructure only (37 names) with software added as a category experiment?
3. **Spread data:** should we sample quotes in v1 to measure real costs, or start with conservative fixed costs in `costs.yaml` and measure later?
