"""
Shared utilities for Greeks calculations.
"""

from __future__ import annotations

from dataclasses import replace

from ...options.base import OptionContract, OptionLeg
from ...options.strategies import CompositeOptionContract
from .base import Priceable


def shift_option_expiry(obj: Priceable, dt: float) -> Priceable:
    """
    Return a copy of obj with expiry shifted by dt.
    """
    if isinstance(obj, OptionLeg):
        return replace(obj, contract=shift_option_expiry(obj.contract, dt))

    if isinstance(obj, CompositeOptionContract):
        new_legs = [shift_option_expiry(leg, dt) for leg in obj.legs]
        return CompositeOptionContract.from_legs(new_legs)

    if hasattr(obj, "expiry") and isinstance(obj, OptionContract):
        new_expiry = None if obj.expiry is None else max(obj.expiry + dt, 0.0)

        # Extract init params (rough heuristic)
        # Note: In quantlab, EuropeanCall etc use strike, expiry
        kwargs = {"strike": obj.strike, "expiry": new_expiry}

        # Optional constructor parameters (duck-typed)
        for attr in ["payout", "barrier", "barrier_direction", "knock_type"]:
            if hasattr(obj, attr):
                kwargs[attr] = getattr(obj, attr)

        return obj.__class__(**kwargs)

    return obj


def resolve_market_overrides(base_market, provided_market=None, spot=None, rate=None, vol=None):
    """
    Resolve final MarketData from optional overrides.
    """
    m = provided_market or base_market
    if spot is not None:
        m = m.with_spot(spot)
    if rate is not None:
        m = m.with_rate(rate)
    if vol is not None:
        m = m.with_vol(vol)
    return m
