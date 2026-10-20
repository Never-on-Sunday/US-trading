# Experiment 001 — First end-to-end run

Run on 2026-10-03. Configs: `configs/experiments/baseline_v1.yaml`, `configs/experiments/train_from_2023.yaml`.
Reproduce with `trading data-sync && trading data-clean && trading experiment --name baseline_v1 && trading report`.
The full HTML report is written to `artifacts/runs/baseline_v1/report.html` (not in git).

## Question

Starting with $2,000 on 2026-01-02 and paying Binance Stocks fees, does an ML bot trading 50 AI-chain stocks on 5-minute bars beat simply buying an ETF?

## Answer

**No.** The bot made money after fees in every version (+0.7% to +23.0%), and its best version beat SPY and QQQ. None came close to holding semiconductors (SMH +70.7%) or the 50 stocks (+66.6%). The +23.0% is also not yet distinguishable from luck.

## Setup

| | |
|---|---|
| Data | Alpaca SIP 1-minute bars, 2021-01-04 → 2026-10-02, 53 symbols, 35.9M bars → 5.67M five-minute bars |
| Model | `HistGradientBoostingRegressor` on 52 features, predicting the forward return over the holding time |
| Split | Train 2021–2024 · tune on 2025 · test once on 2026-01-02 → 2026-10-02 (189 sessions) |
| Rules | Decide at a 5-minute bar close, fill at the next bar's open. Long only, ≥ $350 per order, flat overnight. |
| Costs | Fee max($0.35, 0.10%) per order + half-spread (1 / 2 / 4 bp by liquidity tier) + 1 bp slippage per fill |
| Setting chosen on 2025 | Enter when predicted gain ≥ 0.6%, hold to near the close (72 bars), at most 2 positions |

## Results on the 2026 test period

| Strategy | Final value | Return | Sharpe | Worst drop | Trades | Fees |
|---|---|---|---|---|---|---|
| Bot — trained once (on 2021–2025) | $2,460 | +23.0% | 1.46 | −9.8% | 69 | $158 |
| Bot — retrained monthly | $2,165 | +8.2% | 0.62 | −7.1% | 80 | $164 |
| SMH | $3,415 | +70.7% | 1.91 | −24.6% | — | — |
| Buy & hold all 50 stocks | $3,332 | +66.6% | 1.83 | −24.7% | — | — |
| QQQ | $2,415 | +20.7% | 1.33 | −11.8% | — | — |
| SPY | $2,242 | +12.1% | 1.23 | −9.1% | — | — |

Cost sensitivity (trained once / retrained monthly): no costs +35.3% / +20.8% · realistic +23.0% / +8.2% · double costs +11.9% / −3.0%. T+1 settlement of sale proceeds changes nothing, because the bot trades at most once a day.

### Experiment: train only from 2023

| Training data | Setting chosen on 2025 | 2025 return | 2026, trained once | 2026, retrained monthly |
|---|---|---|---|---|
| From 2021-01 | 360 min, ≥ 0.6%, 2 positions | +31.8% | +23.0% | +8.2% |
| From 2023-01 | 30 min, ≥ 0.4%, 5 positions | +19.9% | +0.7% | +6.3% |

Training only on the AI-rally years was worse on validation and on the test. Keep the 2021 start.

## What the evidence says

1. **The profit could be luck.** Re-drawing the trained-once version's 69 trades gives a 90% profit range of −$77 to +$975. For the monthly version it is −$343 to +$661.
2. **The four versions disagree** (+23.0%, +8.2%, +0.7%, +6.3%). A robust edge wouldn't swing this much with retraining cadence or training window.
3. **The chosen setting was a corner of the grid.** On 2025, only 6 of 34 settings at that holding time made money, all at the strictest entry threshold. In 2025 buy and hold of the 50 stocks made +52.0%, more than the bot's +31.8%.
4. **There is a real signal in the strongest predictions.** On the 2026 test, the top 1% of predictions were followed by an average move of +0.40% and the top 0.1% by +1.8%, before costs. A round trip costs about 0.25%. Everything below the top ~1% is unprofitable after fees.
5. **The bot is in cash most of the time.** It traded on 38 of 189 days and made no trades in August–September, when no prediction reached the threshold. The AI stocks kept rising meanwhile.
6. **Most of 2026's gain happened overnight.** Splitting returns into overnight (close → next open) and intraday (open → close): SMH about +63% overnight vs +5% intraday; the 50 stocks about +39% vs +10%. A bot that is flat overnight cannot capture this by design. Against the intraday-only return of the same stocks, the bot's result looks much better.
7. **Retraining monthly did not clearly help.** It was worse with 2021 data and better with 2023 data. Weekly retraining isn't justified yet.

## Limits

- The 50 stocks were chosen in 2026 knowing they were winners.
- Spreads are assumed, not measured, and 39 of 69 entries were in the first half hour, when real spreads are widest.
- One model family, one test period.

## Next experiments, in order of expected value

1. **Allow overnight holds** (multi-day horizon, daily decisions). This is where the return was, and it cuts fees per unit of exposure.
2. **Core + overlay:** hold the basket or SMH as the core position and let the model decide only when to add or reduce.
3. **Pick settings by robustness, not by the single best validation result**, e.g. require neighbouring settings to be profitable too.
4. **Measure real spreads** from Binance/Alpaca quotes, especially 09:30–10:00.
5. **Binance stock perpetuals** (0.05% fee, shorting allowed) as a cheaper venue for intraday signals.
