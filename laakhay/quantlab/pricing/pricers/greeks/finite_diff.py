"""Finite difference Greeks calculator that works with any backend."""

from __future__ import annotations

from laakhay.quantlab.backend import Backend, get_backend
from laakhay.quantlab.pricing.greeks import Greeks
from laakhay.quantlab.pricing.market import MarketData
from laakhay.quantlab.pricing.options.base import BaseLeg, CashLeg, OptionContract, OptionLeg
from laakhay.quantlab.pricing.options.strategies import CompositeOptionContract

from ..pricer import Pricer
from .base import GreeksCalculator, Priceable
from .utils import (
    resolve_market_overrides,
    shift_option_expiry,
)


class FiniteDiffGreeksCalculator(GreeksCalculator):
    """Calculate Greeks using finite differences around price."""

    # Bump sizes for finite differences
    _SPOT_BUMP = 0.01  # 1-percent relative bump
    _VOL_BUMP = 0.001  # absolute bump (0.10% vol)
    _RATE_BUMP = 0.0001  # 1-bp bump (0.01%)
    _TIME_BUMP = 1.0 / 365.0  # one calendar day

    def __init__(
        self,
        backend: str | Backend | None = None,
        market: MarketData | None = None,
        *,
        spot: float | None = None,
        rate: float | None = None,
        vol: float | None = None,
        spot_bump: float | None = None,
        vol_bump: float | None = None,
        rate_bump: float | None = None,
        time_bump: float | None = None,
    ) -> None:
        super().__init__(market=market, spot=spot, rate=rate, vol=vol)
        self.backend = get_backend(backend)
        self._create_market(self.backend)
        self._pricer = Pricer(backend=self.backend, market=self.market)

        # Allow custom bump sizes for stability studies
        self.spot_bump = spot_bump if spot_bump is not None else self._SPOT_BUMP
        self.vol_bump = vol_bump if vol_bump is not None else self._VOL_BUMP
        self.rate_bump = rate_bump if rate_bump is not None else self._RATE_BUMP
        self.time_bump = time_bump if time_bump is not None else self._TIME_BUMP

    def supports(self, option: Priceable) -> bool:
        """Check if finite differences can be used."""
        if isinstance(option, (list, tuple, set)):
            return all(self.supports(opt) for opt in option)

        if isinstance(option, CashLeg):
            return True

        if isinstance(option, OptionLeg):
            return self.supports(option.contract)

        if isinstance(option, CompositeOptionContract):
            return all(self.supports(leg) for leg in option.legs)

        return self._pricer.supports(option)

    def calculate(
        self,
        option: Priceable,
        *,
        market: MarketData | None = None,
        spot: float | None = None,
        rate: float | None = None,
        vol: float | None = None,
    ) -> Greeks:
        """Calculate Greeks using finite differences."""
        if isinstance(option, (list, tuple, set)):
            total_greeks = None
            for single_option in option:
                single_greeks = self.calculate(
                    single_option, market=market, spot=spot, rate=rate, vol=vol
                )
                if total_greeks is None:
                    total_greeks = single_greeks
                else:
                    total_greeks = total_greeks + single_greeks
            return total_greeks or Greeks()

        # Resolve market data
        m = resolve_market_overrides(self.market, market, spot, rate, vol)
        b = self.backend

        # Helper for bumped markets
        def _bump(relative: bool = False, **kwargs) -> MarketData:
            new = m
            if "spot" in kwargs:
                spot_val = kwargs["spot"]
                if relative:
                    spot_val = b.mul(m.spot, 1 + spot_val)
                new = new.with_spot(spot_val)
            if "vol" in kwargs:
                new = new.with_vol(kwargs["vol"])
            if "rate" in kwargs:
                new = new.with_rate(kwargs["rate"])
            return new

        # Core finite-difference prices
        price_base = self._pricer.price(option, market=m)

        # Spot bumps
        price_up_S = self._pricer.price(option, market=_bump(relative=True, spot=self.spot_bump))
        price_dn_S = self._pricer.price(option, market=_bump(relative=True, spot=-self.spot_bump))

        # Vol bumps
        price_up_vol = self._pricer.price(option, market=_bump(vol=b.add(m.vol, self.vol_bump)))
        price_dn_vol = self._pricer.price(option, market=_bump(vol=b.add(m.vol, -self.vol_bump)))

        # Rate bumps
        price_up_r = self._pricer.price(option, market=_bump(rate=b.add(m.rate, self.rate_bump)))
        price_dn_r = self._pricer.price(option, market=_bump(rate=b.add(m.rate, -self.rate_bump)))

        # Greeks component-wise
        delta = b.divide(b.subtract(price_up_S, price_dn_S), b.mul(m.spot, 2 * self.spot_bump))
        gamma = b.divide(
            b.add(price_up_S, b.add(price_dn_S, -b.mul(price_base, 2))),
            b.mul(b.power(m.spot, 2), self.spot_bump**2),
        )
        vega = b.divide(b.subtract(price_up_vol, price_dn_vol), 2 * self.vol_bump)
        vega = b.divide(vega, 100.0)

        rho = b.divide(b.subtract(price_up_r, price_dn_r), 2 * self.rate_bump)
        rho = b.divide(rho, 100.0)

        # Theta requires time bump
        option_minus_dt = shift_option_expiry(option, -self.time_bump)
        if option_minus_dt is option:
            theta = b.zeros_like(price_base) if hasattr(b, "zeros_like") else 0.0
        else:
            price_dt = self._pricer.price(option_minus_dt, market=m)
            theta = b.divide(b.subtract(price_dt, price_base), self.time_bump)
            theta = b.divide(theta, 365.0)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    def calculate_greeks(
        self,
        option: OptionContract | BaseLeg | CompositeOptionContract,
    ) -> Greeks:
        """Calculate Greeks for the given option using finite differences."""
        return self.calculate(option)
