"""Greeks data structures with type safety and vectorized operations."""

from __future__ import annotations

from dataclasses import dataclass, fields
from laakhay.quantlab.types import Scalar


@dataclass
class Greeks:
    """Container for option Greeks with vectorized operations."""

    delta: float | object = 0.0
    gamma: float | object = 0.0
    vega: float | object = 0.0
    theta: float | object = 0.0
    rho: float | object = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, float | object]) -> Greeks:
        """Create Greeks from dictionary."""
        kwargs = {f.name: data[f.name] for f in fields(cls) if f.name in data}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, float | object]:
        """Convert to dictionary."""
        return {
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }

    # ------------------------------------------------------------------
    # Mapping-like compatibility layer (legacy tests expect dict interface)
    # ------------------------------------------------------------------

    def keys(self):
        """Return iterable of greek names – behaves like dict.keys()."""
        return ("delta", "gamma", "vega", "theta", "rho")

    def __getitem__(self, item: str):
        if item not in self.keys():
            raise KeyError(item)
        return getattr(self, item)

    def __iter__(self):  # Allows dict(greeks) casting
        for k in self.keys():
            yield k

    def __len__(self):
        return 5

    def items(self):
        """Yield (name, value) pairs like dict.items()."""
        for k in self.keys():
            yield (k, self[k])

    def __add__(self, other: Greeks) -> Greeks:
        """Portfolio aggregation via addition."""
        if not isinstance(other, Greeks):
            raise TypeError("Can only add Greeks to Greeks")

        return Greeks(
            delta=self._add_values(self.delta, other.delta),
            gamma=self._add_values(self.gamma, other.gamma),
            vega=self._add_values(self.vega, other.vega),
            theta=self._add_values(self.theta, other.theta),
            rho=self._add_values(self.rho, other.rho),
        )

    def __mul__(self, scalar: float | int) -> Greeks:
        """Position sizing via scalar multiplication."""
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only multiply Greeks by scalar")

        return Greeks(
            delta=self._mul_value(self.delta, scalar),
            gamma=self._mul_value(self.gamma, scalar),
            vega=self._mul_value(self.vega, scalar),
            theta=self._mul_value(self.theta, scalar),
            rho=self._mul_value(self.rho, scalar),
        )

    def __rmul__(self, scalar: float | int) -> Greeks:
        return self.__mul__(scalar)

    def __neg__(self) -> Greeks:
        return self * -1

    def __sub__(self, other: Greeks) -> Greeks:
        return self + (-other)

    def is_scalar(self) -> bool:
        """Check if all Greeks are scalar values."""
        values = [self.delta, self.gamma, self.vega, self.theta, self.rho]
        return all(isinstance(val, (int, float)) for val in values)

    def is_array(self) -> bool:
        """Check if Greeks contain array values."""
        return not self.is_scalar()

    def shape(self) -> tuple[int, ...] | None:
        """Get shape of array Greeks."""
        if self.is_scalar():
            return None
        for value in [self.delta, self.gamma, self.vega, self.theta, self.rho]:
            if hasattr(value, "shape"):
                return tuple(value.shape)
        return None

    def delta_equivalent(self, spot: float | object) -> float | object:
        """Calculate delta-equivalent notional exposure."""
        return self.delta * spot

    def gamma_pnl(self, price_move: float | object) -> float | object:
        """Estimate P&L from gamma exposure."""
        return 0.5 * self.gamma * (price_move**2)

    def vega_pnl(self, vol_change: float | object) -> float | object:
        """Estimate P&L from vega exposure (vol_change in percentage points)."""
        return self.vega * (vol_change * 100)

    def theta_decay(self) -> float | object:
        """Expected 1-day theta decay."""
        return self.theta

    def summary(self) -> str:
        """Concise summary string."""
        if self.is_scalar():
            return (
                f"Greeks(δ={self.delta:.4f}, γ={self.gamma:.6f}, "
                f"ν={self.vega:.4f}, θ={self.theta:.4f}, ρ={self.rho:.4f})"
            )
        return f"Greeks(shape={self.shape()}, arrays=[δ,γ,ν,θ,ρ])"

    @staticmethod
    def _add_values(a: float | object, b: float | object) -> float | object:
        """Add two tensors using appropriate backend operations."""
        return a + b

    @staticmethod
    def _mul_value(value: float | object, scalar: float | int) -> float | object:
        return value * scalar

    def __repr__(self) -> str:
        return self.summary()


def combine_greeks(*greeks_list: Greeks) -> Greeks:
    """Combine multiple Greeks (portfolio aggregation)."""
    if not greeks_list:
        return Greeks()

    result = greeks_list[0]
    for greeks in greeks_list[1:]:
        result = result + greeks
    return result


def scale_greeks(greeks: Greeks, factor: float | int) -> Greeks:
    """Scale Greeks by factor (position sizing)."""
    return greeks * factor


def greeks_from_legs(leg_greeks: list[Greeks], quantities: list[float | int]) -> Greeks:
    """Combine leg Greeks with quantities."""
    if len(leg_greeks) != len(quantities):
        raise ValueError("Greeks and quantities length mismatch")

    scaled = [greeks * qty for greeks, qty in zip(leg_greeks, quantities)]
    return combine_greeks(*scaled)
