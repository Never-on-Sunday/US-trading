"""Trading costs. Default profile = Binance Stocks: fee max($0.35, 0.10% of order) per order,
plus half the bid-ask spread and slippage on every fill."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CostModel:
    fee_rate: float = 0.001
    fee_min: float = 0.35
    slippage_bps: float = 1.0
    default_half_spread_bps: float = 3.0
    half_spread_bps: dict[str, float] = field(default_factory=dict)  # per symbol, by liquidity tier
    multiplier: float = 1.0  # 2.0 = stress test at double costs

    def fee(self, notional: float) -> float:
        return max(self.fee_min, self.fee_rate * notional) * self.multiplier

    def price_impact(self, symbol: str) -> float:
        """Fractional price penalty per fill (paid on both buy and sell)."""
        hs = self.half_spread_bps.get(symbol, self.default_half_spread_bps)
        return (hs + self.slippage_bps) * 1e-4 * self.multiplier


def spread_tiers(median_dollar_volume: dict[str, float]) -> dict[str, float]:
    """Half-spread assumption by daily dollar volume: mega-caps 1 bp, liquid 2 bp, the rest 4 bp."""
    out = {}
    for sym, dv in median_dollar_volume.items():
        out[sym] = 1.0 if dv >= 2e9 else 2.0 if dv >= 3e8 else 4.0
    return out
