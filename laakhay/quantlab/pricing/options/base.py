"""Base option contract and leg definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

from laakhay.quantlab.backend import Backend

from ..utils import infer_backend


class Side(Enum):
    """Represents the side of an option contract."""

    CALL = auto()
    PUT = auto()
    COMPOSITE = auto()


class PositionType(Enum):
    """Represents the type of position in an option contract."""

    LONG = +1
    SHORT = -1


class PayoffType(Enum):
    """Represents the type of payoff for an option contract."""

    EUROPEAN = auto()
    DIGITAL = auto()
    ASIAN = auto()
    BARRIER = auto()
    GEOMETRIC_ASIAN = auto()
    COMPOSITE = auto()
    CASH = auto()


class PositionSide(Enum):
    """Position side for option legs."""

    LONG = 1
    SHORT = -1


@dataclass(frozen=True)
class OptionContract(ABC):
    """Pure contract definition without position/quantity."""

    strike: float | None = None
    expiry: float | None = None

    def __repr__(self) -> str:
        return f"{self.payoff_type.name.lower()}–{self.side.name.lower()}(K={self.strike}, T={self.expiry})"

    @property
    @abstractmethod
    def side(self) -> Side:
        """Which side (call/put/composite)."""
        pass

    @property
    @abstractmethod
    def payoff_type(self) -> PayoffType:
        """Type of payoff."""
        pass

    @property
    def is_analytic(self) -> bool:
        """Has known analytic solution."""
        return self.payoff_type in {
            PayoffType.EUROPEAN,
            PayoffType.DIGITAL,
            PayoffType.GEOMETRIC_ASIAN,
        }

    def is_expired(self, current_time: float = 0.0) -> bool:
        """True if option has expired."""
        if self.expiry is None:
            return False
        return current_time >= self.expiry

    @abstractmethod
    def __call__(self, data, backend: Backend = None):
        """Compute option payoff."""
        pass


@dataclass(frozen=True)
class PathIndependentOption(OptionContract):
    """Base class for path-independent options."""

    def __post_init__(self):
        if self.strike is None or self.strike < 0:
            raise ValueError(f"Invalid strike price: {self.strike}")
        if self.expiry is None or self.expiry < 0:
            raise ValueError(f"Invalid expiry: {self.expiry}")

    @infer_backend
    def __call__(self, data, backend: Backend = None):
        if backend.ndim(data) == 1:
            return self.payoff(data, backend=backend)
        else:
            return self.payoff(data[:, -1], backend=backend)

    @abstractmethod
    def intrinsic_value(self, spot: float, backend: Backend = None) -> float:
        """Intrinsic value at a given spot."""
        pass

    @abstractmethod
    def payoff(self, spot_prices, backend: Backend = None):
        """Payoff for terminal spot price(s)."""
        pass

    def is_itm(self, spot: float) -> bool:
        return spot > self.strike if self.side == Side.CALL else spot < self.strike

    def is_atm(self, spot: float, tol: float = 1e-6) -> bool:
        return abs(spot - self.strike) < tol

    def is_otm(self, spot: float) -> bool:
        return not (self.is_itm(spot) or self.is_atm(spot))

    def moneyness(self, spot: float) -> float:
        if spot <= 0:
            raise ValueError("Spot must be positive")
        return spot / self.strike


@dataclass(frozen=True)
class PathDependentOption(OptionContract):
    """Base class for path-dependent options (Asian, Barrier, etc.)."""

    def __post_init__(self):
        """Validate strike and expiry."""
        if self.strike is None or self.strike < 0:
            raise ValueError(f"Invalid strike price: {self.strike}")
        if self.expiry is None or self.expiry < 0:
            raise ValueError(f"Invalid expiry: {self.expiry}")

    @infer_backend
    def __call__(self, price_paths, backend: Backend = None):
        """Compute payoff from full price paths."""
        paths = backend.ensure_ndim(price_paths, 2, "PathDependentOption requires 2D paths")
        return self._path_payoff(paths, backend=backend)


class BaseLeg(ABC):
    """Base class for all leg types."""

    @property
    @abstractmethod
    def qty(self) -> float:
        pass

    @property
    @abstractmethod
    def is_analytic(self) -> bool:
        pass

    @property
    @abstractmethod
    def payoff_type(self) -> PayoffType:
        pass

    @abstractmethod
    def is_expired(self, current_time: float = 0.0) -> bool:
        pass

    @abstractmethod
    def compute_payoff(self, price_data, backend: Backend = None):
        pass


@dataclass(frozen=True)
class OptionLeg(BaseLeg):
    """Wraps an OptionContract with quantity."""

    contract: OptionContract
    qty: float = 1

    def __post_init__(self):
        if not isinstance(self.contract, OptionContract):
            raise TypeError(f"Expected OptionContract, got {type(self.contract)}")
        if self.qty == 0:
            raise ValueError("Quantity must be non-zero")

    @property
    def side(self) -> PositionSide:
        return PositionSide.LONG if self.qty > 0 else PositionSide.SHORT

    @property
    def is_analytic(self) -> bool:
        return self.contract.is_analytic

    @property
    def payoff_type(self) -> PayoffType:
        return self.contract.payoff_type

    def is_expired(self, current_time: float = 0.0) -> bool:
        return self.contract.is_expired(current_time)

    @infer_backend
    def compute_payoff(self, price_data, backend: Backend = None):
        contract_payoff = self.contract(price_data, backend=backend)
        return backend.mul(contract_payoff, self.qty)

    def __call__(self, S) -> object:
        return self.compute_payoff(S)

    def __repr__(self) -> str:
        side_str = "Long" if self.qty > 0 else "Short"
        qty_str = f"{abs(self.qty)}x " if abs(self.qty) != 1 else ""
        return f"{side_str} {qty_str}{self.contract}"


@dataclass(frozen=True)
class CashLeg(BaseLeg):
    """Wraps a cash amount with expiry."""

    amount: float
    expiry: float

    @property
    def qty(self) -> float:
        return self.amount

    @property
    def is_analytic(self) -> bool:
        return True

    @property
    def payoff_type(self) -> PayoffType:
        return PayoffType.COMPOSITE

    def is_expired(self, current_time: float = 0.0) -> bool:
        return current_time >= self.expiry

    @infer_backend
    def compute_payoff(self, price_data, backend: Backend = None):
        if backend.ndim(price_data) == 0:
            return self.amount
        elif backend.ndim(price_data) == 1:
            return backend.full_like(price_data, self.amount)
        else:
            n_paths = backend.shape(price_data)[0]
            return backend.full((n_paths,), self.amount)

    def __repr__(self) -> str:
        side_str = "Long" if self.amount > 0 else "Short"
        return f"{side_str} {abs(self.amount):.2f} Cash@{self.expiry:.2f}"
