from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Instrument:
    symbol: str
    layer: str
    category: str
    active_from: date | None = None
