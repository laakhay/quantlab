"""Black-Scholes analytical formulas for European options."""

from ....greeks import Greeks
from ....market import MarketData
from ....options import EuropeanCall, EuropeanPut
from ..calculations import _normal_pdf, compute_d1_d2, compute_discount_factor
from ..registry import PricingFormula


class EuropeanGreeks:
    """Analytical Greeks computation for European options."""

    @staticmethod
    def compute_greeks(option, market: MarketData, d1, d2, is_call: bool) -> Greeks:
        backend = market.backend

        if option.expiry <= 0:
            if is_call:
                is_itm = backend.greater(market.spot, option.strike)
                delta = backend.where(is_itm, 1.0, 0.0)
            else:
                is_itm = backend.less(market.spot, option.strike)
                delta = backend.where(is_itm, -1.0, 0.0)
            zero = backend.zeros_like(delta) if hasattr(backend, "zeros_like") else 0.0
            return Greeks(delta=delta, gamma=zero, vega=zero, theta=zero, rho=zero)

        sqrt_T = backend.sqrt(backend.array(option.expiry))
        discount = compute_discount_factor(option.expiry, market)

        phi_d1 = _normal_pdf(d1, backend)
        N_d1 = backend.norm_cdf(d1)
        N_d2 = backend.norm_cdf(d2)

        if is_call:
            delta = N_d1
            rho = backend.divide(
                backend.mul(backend.mul(option.strike, option.expiry), backend.mul(discount, N_d2)),
                100.0,
            )
            theta_term1 = backend.divide(
                backend.mul(backend.mul(market.spot, phi_d1), market.vol), backend.mul(2.0, sqrt_T)
            )
            theta_term2 = backend.mul(
                backend.mul(market.rate, backend.mul(option.strike, discount)), N_d2
            )
            theta = backend.divide(backend.mul(-1, backend.add(theta_term1, theta_term2)), 365.0)
        else:
            N_minus_d1 = backend.norm_cdf(backend.mul(-1, d1))
            delta = backend.mul(-1, N_minus_d1)
            N_minus_d2 = backend.norm_cdf(backend.mul(-1, d2))
            rho = backend.divide(
                backend.mul(
                    -1,
                    backend.mul(
                        backend.mul(option.strike, option.expiry), backend.mul(discount, N_minus_d2)
                    ),
                ),
                100.0,
            )
            theta_term1 = backend.divide(
                backend.mul(backend.mul(market.spot, phi_d1), market.vol), backend.mul(2.0, sqrt_T)
            )
            theta_term2 = backend.mul(
                backend.mul(market.rate, backend.mul(option.strike, discount)), N_minus_d2
            )
            theta = backend.divide(backend.add(backend.mul(-1, theta_term1), theta_term2), 365.0)

        gamma = backend.divide(phi_d1, backend.mul(backend.mul(market.spot, market.vol), sqrt_T))
        vega = backend.divide(backend.mul(backend.mul(market.spot, phi_d1), sqrt_T), 100.0)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


class EuropeanCallFormula(PricingFormula):
    """European call pricing formula."""

    def price(self, option: EuropeanCall, market: MarketData) -> object:
        backend = market.backend
        d1, d2 = compute_d1_d2(option.strike, option.expiry, market)
        discount = compute_discount_factor(option.expiry, market)
        N_d1 = backend.norm_cdf(d1)
        N_d2 = backend.norm_cdf(d2)
        return backend.add(
            backend.mul(market.spot, N_d1), backend.mul(backend.mul(-option.strike, discount), N_d2)
        )

    def supports(self, option) -> bool:
        return isinstance(option, EuropeanCall)

    def price_with_greeks(self, option: EuropeanCall, market: MarketData) -> tuple[object, Greeks]:
        backend = market.backend
        d1, d2 = compute_d1_d2(option.strike, option.expiry, market)
        discount = compute_discount_factor(option.expiry, market)
        N_d1, N_d2 = backend.norm_cdf(d1), backend.norm_cdf(d2)
        price = backend.add(
            backend.mul(market.spot, N_d1), backend.mul(backend.mul(-option.strike, discount), N_d2)
        )
        greeks = EuropeanGreeks.compute_greeks(option, market, d1, d2, is_call=True)
        return price, greeks


class EuropeanPutFormula(PricingFormula):
    """European put pricing formula."""

    def price(self, option: EuropeanPut, market: MarketData) -> object:
        backend = market.backend
        d1, d2 = compute_d1_d2(option.strike, option.expiry, market)
        discount = compute_discount_factor(option.expiry, market)
        N_minus_d1 = backend.norm_cdf(backend.mul(-1, d1))
        N_minus_d2 = backend.norm_cdf(backend.mul(-1, d2))
        return backend.add(
            backend.mul(backend.mul(option.strike, discount), N_minus_d2),
            backend.mul(-market.spot, N_minus_d1),
        )

    def supports(self, option) -> bool:
        return isinstance(option, EuropeanPut)

    def price_with_greeks(self, option: EuropeanPut, market: MarketData) -> tuple[object, Greeks]:
        backend = market.backend
        d1, d2 = compute_d1_d2(option.strike, option.expiry, market)
        discount = compute_discount_factor(option.expiry, market)
        N_minus_d1, N_minus_d2 = (
            backend.norm_cdf(backend.mul(-1, d1)),
            backend.norm_cdf(backend.mul(-1, d2)),
        )
        price = backend.add(
            backend.mul(backend.mul(option.strike, discount), N_minus_d2),
            backend.mul(-market.spot, N_minus_d1),
        )
        greeks = EuropeanGreeks.compute_greeks(option, market, d1, d2, is_call=False)
        return price, greeks
