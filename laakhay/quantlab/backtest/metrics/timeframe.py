from __future__ import annotations

import re

_TIMEFRAME_PATTERN = re.compile(r"^\s*(\d+)\s*([mhdwM])\s*$")


def infer_periods_per_year(timeframe: str, fallback: int = 252) -> int:
    """Infer annualization periods from a timeframe string like 1m/5m/1h/1d/1w."""
    if not timeframe:
        return fallback

    match = _TIMEFRAME_PATTERN.match(timeframe)
    if not match:
        return fallback

    value = int(match.group(1))
    unit = match.group(2)
    if value <= 0:
        return fallback

    if unit == "m":
        return max(1, (365 * 24 * 60) // value)
    if unit == "h":
        return max(1, (365 * 24) // value)
    if unit == "d":
        return max(1, 365 // value)
    if unit == "w":
        return max(1, 52 // value)
    if unit == "M":
        return max(1, 12 // value)
    return fallback
