"""
Composite option contracts and strategy builders.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from laakhay.quantlab.backend import Backend

from ..utils import infer_backend
from .base import BaseLeg, OptionContract, OptionLeg, PayoffType, Side
from .contracts import EuropeanCall, EuropeanPut
from .digital import DigitalCall, DigitalPut


@dataclass(frozen=True)
class CompositeOptionContract(OptionContract):
    """
    Multi-leg option strategy combining individual positions.
    """

    legs: Sequence[BaseLeg] = field(default_factory=list)

    def __post_init__(self):
        """Validate that composite has legs and all are proper BaseLeg types."""
        if not self.legs:
            raise ValueError("CompositeOptionContract requires at least one leg")

        for leg in self.legs:
            if not isinstance(leg, BaseLeg):
                raise TypeError(f"Expected BaseLeg, got {type(leg).__name__}")

    # === Factory Methods ===

    @classmethod
    def from_legs(cls, legs: Sequence[BaseLeg]) -> CompositeOptionContract:
        """Create composite from list of legs."""
        return cls(legs=legs)

    @classmethod
    def combine(
        cls, *composites_or_legs: CompositeOptionContract | BaseLeg | Sequence[BaseLeg]
    ) -> CompositeOptionContract:
        """Combine multiple composites and/or legs into single strategy."""
        all_legs = []

        for item in composites_or_legs:
            if isinstance(item, CompositeOptionContract):
                all_legs.extend(item.legs)
            elif isinstance(item, BaseLeg):
                all_legs.append(item)
            else:
                all_legs.extend(item)

        return cls(legs=all_legs)

    @classmethod
    def merge(
        cls, composite1: CompositeOptionContract, composite2: CompositeOptionContract
    ) -> CompositeOptionContract:
        """Merge two composite strategies into one."""
        return cls(legs=list(composite1.legs) + list(composite2.legs))

    # === String Representations ===

    def __repr__(self) -> str:
        legs_str = ",".join(
            f"{leg.qty:+}×{leg.contract.__class__.__name__}"
            if hasattr(leg, "contract")
            else f"{leg.qty:+}×{leg.__class__.__name__}"
            for leg in self.legs
        )
        return f"Composite({legs_str})"

    def __str__(self) -> str:
        legs_str = ", ".join(str(leg) for leg in self.legs)
        return f"Composite({legs_str})"

    # === Contract Properties ===

    @property
    def side(self) -> Side:
        """Always COMPOSITE side."""
        return Side.COMPOSITE

    @property
    def payoff_type(self) -> PayoffType:
        """COMPOSITE if mixed types, otherwise the unique type."""
        types = {leg.payoff_type for leg in self.legs}
        return types.pop() if len(types) == 1 else PayoffType.COMPOSITE

    @property
    def is_analytic(self) -> bool:
        """True only if ALL legs have analytic solutions."""
        return all(leg.is_analytic for leg in self.legs)

    def is_expired(self, current_time: float = 0.0) -> bool:
        """True if ALL legs have expired."""
        return all(leg.is_expired(current_time) for leg in self.legs)

    # === Payoff Computation ===

    @infer_backend
    def __call__(self, price_data, backend: Backend = None):
        """Compute total payoff across all legs."""
        if not self.legs:
            if backend.ndim(price_data) == 0:
                return 0.0
            elif backend.ndim(price_data) == 1:
                return backend.zeros_like(price_data)
            else:
                n_paths = backend.shape(price_data)[0]
                return backend.zeros((n_paths,))

        total = None
        for leg in self.legs:
            leg_payoff = leg.compute_payoff(price_data, backend=backend)

            total = leg_payoff if total is None else backend.add(total, leg_payoff)

        return total

    def payoff(self, spot_prices, backend: Backend = None):
        """Compute payoff for spot prices."""
        return self(spot_prices, backend=backend)


# === Strategy Builders ===


def vertical_spread(
    lower_strike: float,
    upper_strike: float,
    expiry: float,
    is_call: bool = True,
    is_digital: bool = False,
) -> CompositeOptionContract:
    """Create a vertical spread strategy."""
    if lower_strike >= upper_strike:
        raise ValueError("Lower strike must be less than upper strike")

    if is_digital:
        option_class_lower = DigitalCall if is_call else DigitalPut
        option_class_upper = DigitalCall if is_call else DigitalPut
    else:
        option_class_lower = EuropeanCall if is_call else EuropeanPut
        option_class_upper = EuropeanCall if is_call else EuropeanPut

    legs = [
        OptionLeg(option_class_lower(strike=lower_strike, expiry=expiry), qty=1),
        OptionLeg(option_class_upper(strike=upper_strike, expiry=expiry), qty=-1),
    ]
    return CompositeOptionContract.from_legs(legs)


def calendar_spread(
    strike: float,
    near_expiry: float,
    far_expiry: float,
    is_call: bool = True,
) -> CompositeOptionContract:
    """Create a calendar spread strategy."""
    if near_expiry >= far_expiry:
        raise ValueError("Near expiry must be less than far expiry")

    option_class = EuropeanCall if is_call else EuropeanPut
    legs = [
        OptionLeg(option_class(strike=strike, expiry=near_expiry), qty=-1),
        OptionLeg(option_class(strike=strike, expiry=far_expiry), qty=1),
    ]
    return CompositeOptionContract.from_legs(legs)


def butterfly_spread(
    lower_strike: float,
    middle_strike: float,
    upper_strike: float,
    expiry: float,
    is_call: bool = True,
) -> CompositeOptionContract:
    """Create a butterfly spread strategy."""
    if not (lower_strike < middle_strike < upper_strike):
        raise ValueError("Strikes must be in ascending order")

    option_class = EuropeanCall if is_call else EuropeanPut
    legs = [
        OptionLeg(option_class(strike=lower_strike, expiry=expiry), qty=1),
        OptionLeg(option_class(strike=middle_strike, expiry=expiry), qty=-2),
        OptionLeg(option_class(strike=upper_strike, expiry=expiry), qty=1),
    ]
    return CompositeOptionContract.from_legs(legs)


def straddle(
    strike: float,
    expiry: float,
    is_digital: bool = False,
) -> CompositeOptionContract:
    """Create a straddle strategy."""
    if is_digital:
        call_class, put_class = DigitalCall, DigitalPut
    else:
        call_class, put_class = EuropeanCall, EuropeanPut

    legs = [
        OptionLeg(call_class(strike=strike, expiry=expiry), qty=1),
        OptionLeg(put_class(strike=strike, expiry=expiry), qty=1),
    ]
    return CompositeOptionContract.from_legs(legs)


def strangle(
    call_strike: float,
    put_strike: float,
    expiry: float,
    is_digital: bool = False,
) -> CompositeOptionContract:
    """Create a strangle strategy."""
    if is_digital:
        call_class, put_class = DigitalCall, DigitalPut
    else:
        call_class, put_class = EuropeanCall, EuropeanPut

    legs = [
        OptionLeg(call_class(strike=call_strike, expiry=expiry), qty=1),
        OptionLeg(put_class(strike=put_strike, expiry=expiry), qty=1),
    ]
    return CompositeOptionContract.from_legs(legs)


def iron_condor(
    put_strike_long: float,
    put_strike_short: float,
    call_strike_short: float,
    call_strike_long: float,
    expiry: float,
) -> CompositeOptionContract:
    """Create an iron condor strategy."""
    if not (put_strike_long < put_strike_short < call_strike_short < call_strike_long):
        raise ValueError("Strikes must be in ascending order")

    legs = [
        OptionLeg(EuropeanPut(strike=put_strike_long, expiry=expiry), qty=1),
        OptionLeg(EuropeanPut(strike=put_strike_short, expiry=expiry), qty=-1),
        OptionLeg(EuropeanCall(strike=call_strike_short, expiry=expiry), qty=-1),
        OptionLeg(EuropeanCall(strike=call_strike_long, expiry=expiry), qty=1),
    ]
    return CompositeOptionContract.from_legs(legs)
