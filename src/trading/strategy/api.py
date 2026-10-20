"""Public API of the strategy module (workflow step 4): turn predictions into trading decisions.

A strategy only decides; the backtest (and later the live runner) executes, charges costs and keeps
the cash ledger. To try a new idea, add a class that follows `Strategy` and register it in STRATEGIES;
an experiment config then picks it with `strategy: <name>`.
"""

from typing import Literal, Protocol

import numpy as np

from trading.strategy.threshold import ThresholdStrategy

Action = Literal["hold", "renew", "exit"]


class Strategy(Protocol):
    max_positions: int

    def manage(self, bars_held: int, pred: float) -> Action:
        """For an open position: keep it, restart its holding clock, or sell at the next open."""

    def can_enter(self, bar_idx: int, n_bars: int) -> bool:
        """May new positions be opened after bar `bar_idx` of a session with `n_bars` bars?"""

    def rank_entries(self, pred: np.ndarray, fillable: np.ndarray) -> np.ndarray:
        """Symbol indices worth buying now, best first."""

    def target_notional(self, equity: float) -> float:
        """Dollar size of one new position."""


STRATEGIES: dict[str, type] = {"threshold": ThresholdStrategy}


def make_strategy(name: str = "threshold", **params) -> Strategy:
    return STRATEGIES[name](**params)
