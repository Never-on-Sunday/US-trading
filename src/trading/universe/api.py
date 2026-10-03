"""Public API of the universe module: which symbols exist and when they are tradable."""

from datetime import date
from functools import lru_cache

import yaml

from trading.shared.storage import CONFIG_DIR
from trading.shared.types import Instrument


@lru_cache
def get_universe(name: str = "ai_chain") -> tuple[Instrument, ...]:
    raw = yaml.safe_load((CONFIG_DIR / "universe" / f"{name}.yaml").read_text())
    return tuple(
        Instrument(s["symbol"], s["layer"], s["category"], s.get("active_from"))
        for s in raw["symbols"]
    )


def symbols(name: str = "ai_chain", as_of: date | None = None) -> list[str]:
    return [
        i.symbol for i in get_universe(name)
        if as_of is None or i.active_from is None or i.active_from <= as_of
    ]
