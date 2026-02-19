"""Market data container for option pricing."""

from __future__ import annotations

from dataclasses import dataclass, replace

from laakhay.quantlab.backend import Backend, get_backend


@dataclass(frozen=True)
class MarketData:
    """Immutable market data bundle: spot, rate, vol."""

    spot: float | object = 100.0
    rate: float | object = 0.0
    vol: float | object = 0.2
    backend: Backend | None = None

    def __post_init__(self) -> None:
        """Initialize backend and validate inputs."""
        # Set default backend if None
        if self.backend is None:
            object.__setattr__(self, "backend", get_backend())

        b = self.backend
        # Note: In quantlab, get_backend returns an AbstractBackend/Backend instance

        # Convert inputs
        object.__setattr__(self, "spot", b.convert(self.spot))
        object.__setattr__(self, "rate", b.convert(self.rate))
        object.__setattr__(self, "vol", b.convert(self.vol))

        # Validate
        if b.any(b.less(self.spot, 0)):
            raise ValueError("spot must be non-negative")
        if b.any(b.less(self.vol, 0)):
            raise ValueError("vol must be non-negative")

    @classmethod
    def create(
        cls,
        *,
        spot: float | object = 100.0,
        rate: float | object = 0.0,
        vol: float | object = 0.2,
        backend: Backend | None = None,
    ) -> MarketData:
        """Create market data with sensible defaults."""
        return cls(spot=spot, rate=rate, vol=vol, backend=backend)

    def with_spot(self, spot: float | object) -> MarketData:
        """Return copy with new spot price."""
        return replace(self, spot=spot)

    def with_rate(self, rate: float | object) -> MarketData:
        """Return copy with new interest rate."""
        return replace(self, rate=rate)

    def with_vol(self, vol: float | object) -> MarketData:
        """Return copy with new volatility."""
        return replace(self, vol=vol)

    @property
    def shape(self) -> tuple:
        """Broadcast shape of market data arrays."""
        b = self.backend
        for arr in (self.spot, self.rate, self.vol):
            if b.ndim(arr) > 0:
                return b.shape(arr)
        return ()

    @property
    def is_scalar(self) -> bool:
        """True if all fields are scalars."""
        b = self.backend
        return not any(b.ndim(a) > 0 for a in (self.spot, self.rate, self.vol))

    def __repr__(self) -> str:
        backend_name = getattr(self.backend, "name", str(self.backend))
        return (
            f"MarketData(spot={self.spot}, rate={self.rate}, "
            f"vol={self.vol}, backend={backend_name})"
        )
