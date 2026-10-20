"""Buy the stocks with the highest predicted return above a threshold, equal weight, hold for a fixed horizon."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThresholdStrategy:
    horizon: int                 # bars to hold (5m each)
    threshold: float             # minimum predicted return to enter
    max_positions: int = 4
    no_entry_last_bars: int = 6  # no new positions in the last 30 min
    extend: bool = True          # keep a position past its horizon while the signal stays strong
    hold_threshold: float | None = None  # defaults to threshold / 2

    def manage(self, bars_held: int, pred: float) -> str:
        if bars_held < self.horizon:
            return "hold"
        hold_thr = self.threshold / 2 if self.hold_threshold is None else self.hold_threshold
        if self.extend and pred >= hold_thr:
            return "renew"  # signal still strong: restart the holding clock, save two fees
        return "exit"

    def can_enter(self, bar_idx: int, n_bars: int) -> bool:
        return bar_idx <= n_bars - 2 - self.no_entry_last_bars

    def rank_entries(self, pred: np.ndarray, fillable: np.ndarray) -> np.ndarray:
        cand = np.where((pred >= self.threshold) & fillable)[0]
        return cand[np.argsort(-pred[cand])]

    def target_notional(self, equity: float) -> float:
        return equity / self.max_positions
