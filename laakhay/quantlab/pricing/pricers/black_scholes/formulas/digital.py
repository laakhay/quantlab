"""Black-Scholes formulas for digital options."""

from ....greeks import Greeks
from ....market import MarketData
from ....options.digital import DigitalCall, DigitalPut

from ..registry import PricingFormula
from ..calculations import compute_d1_d2, compute_discount_factor


def _normal_pdf(x, backend):
    """Calculate normal probability density function φ(x)."""
    sqrt_2pi = backend.sqrt(backend.mul(2.0, backend.pi))
    exp_term = backend.exp(backend.mul(-0.5, backend.power(x, 2)))
    return backend.divide(exp_term, sqrt_2pi)


class DigitalGreeks:
    """Modular Greeks computation for Digital options."""

    @staticmethod
    def compute_greeks(option, market: MarketData, d1, d2, is_call: bool) -> Greeks:
        backend = market.backend

        if option.expiry <= 0:
            zero = backend.zeros_like(market.spot) if hasattr(backend, "zeros_like") else 0.0
            return Greeks(delta=zero, gamma=zero, vega=zero, theta=zero, rho=zero)

        sqrt_T = backend.sqrt(backend.convert(option.expiry))
        discount = compute_discount_factor(option.expiry, market)

        phi_d2 = _normal_pdf(d2, backend)
        N_d2 = backend.norm_cdf(d2)

        if is_call:
            # Delta = payout * e^(-rT) * φ(d₂) / (S * vol * √T)
            delta = backend.divide(
                backend.mul(backend.mul(option.payout, discount), phi_d2),
                backend.mul(backend.mul(market.spot, market.vol), sqrt_T),
            )

            # Rho = payout * T * e^(-rT) * N(d₂) / 100
            rho = backend.divide(
                backend.mul(backend.mul(backend.mul(option.payout, option.expiry), discount), N_d2),
                100.0,
            )

            # Theta = payout * e^(-rT) * [r * N(d₂) + φ(d₂) * (r - vol²/2) / (vol * √T)] / 365
            theta_term1 = backend.mul(market.rate, N_d2)
            theta_term2 = backend.divide(
                backend.mul(
                    phi_d2,
                    backend.add(market.rate, backend.mul(-0.5, backend.power(market.vol, 2))),
                ),
                backend.mul(market.vol, sqrt_T),
            )
            theta = backend.divide(
                backend.mul(
                    backend.mul(option.payout, discount), backend.add(theta_term1, theta_term2)
                ),
                365.0,
            )

            # Gamma = -payout * e^(-rT) * φ(d₂) * d₁ / (S² * vol² * T)
            gamma = backend.divide(
                backend.mul(
                    -1, backend.mul(backend.mul(backend.mul(option.payout, discount), phi_d2), d1)
                ),
                backend.mul(
                    backend.mul(backend.power(market.spot, 2), backend.power(market.vol, 2)),
                    option.expiry,
                ),
            )

            # Vega = -payout * e^(-rT) * φ(d₂) * d₁ / (100 * vol)
            vega = backend.divide(
                backend.mul(
                    -1, backend.mul(backend.mul(backend.mul(option.payout, discount), phi_d2), d1)
                ),
                backend.mul(100.0, market.vol),
            )
        else:
            N_minus_d2 = backend.norm_cdf(backend.mul(-1, d2))

            # Delta = -payout * e^(-rT) * φ(d₂) / (S * vol * √T)
            delta = backend.divide(
                backend.mul(-1, backend.mul(backend.mul(option.payout, discount), phi_d2)),
                backend.mul(backend.mul(market.spot, market.vol), sqrt_T),
            )

            # Rho = -payout * T * e^(-rT) * N(-d₂) / 100
            rho = backend.divide(
                backend.mul(
                    -1,
                    backend.mul(
                        backend.mul(backend.mul(option.payout, option.expiry), discount), N_minus_d2
                    ),
                ),
                100.0,
            )

            # Theta = -payout * e^(-rT) * [r * N(-d₂) - φ(d₂) * (r - vol²/2) / (vol * √T)] / 365
            theta_term1 = backend.mul(market.rate, N_minus_d2)
            theta_term2 = backend.divide(
                backend.mul(
                    phi_d2,
                    backend.add(market.rate, backend.mul(-0.5, backend.power(market.vol, 2))),
                ),
                backend.mul(market.vol, sqrt_T),
            )
            theta = backend.divide(
                backend.mul(
                    -1,
                    backend.mul(
                        backend.mul(option.payout, discount),
                        backend.add(theta_term1, backend.mul(-1, theta_term2)),
                    ),
                ),
                365.0,
            )

            # Gamma = payout * e^(-rT) * φ(d₂) * d₁ / (S² * vol² * T)
            gamma = backend.divide(
                backend.mul(backend.mul(backend.mul(option.payout, discount), phi_d2), d1),
                backend.mul(
                    backend.mul(backend.power(market.spot, 2), backend.power(market.vol, 2)),
                    option.expiry,
                ),
            )

            # Vega = payout * e^(-rT) * φ(d₂) * d₁ / (100 * vol)
            vega = backend.divide(
                backend.mul(backend.mul(backend.mul(option.payout, discount), phi_d2), d1),
                backend.mul(100.0, market.vol),
            )

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


class DigitalCallFormula(PricingFormula):
    """Digital call pricing formula."""

    def price(self, option: DigitalCall, market: MarketData):
        backend = market.backend
        _, d2 = compute_d1_d2(option.strike, option.expiry, market)
        discount = compute_discount_factor(option.expiry, market)
        N_d2 = backend.norm_cdf(d2)
        return backend.mul(backend.mul(option.payout, discount), N_d2)

    def supports(self, option) -> bool:
        return isinstance(option, DigitalCall)

    def greeks(self, option: DigitalCall, market: MarketData, d1, d2) -> Greeks:
        return DigitalGreeks.compute_greeks(option, market, d1, d2, is_call=True)

    def price_with_greeks(self, option: DigitalCall, market: MarketData):
        d1, d2 = compute_d1_d2(option.strike, option.expiry, market)
        p = self.price(option, market)
        g = self.greeks(option, market, d1, d2)
        return p, g


class DigitalPutFormula(PricingFormula):
    """Digital put pricing formula."""

    def price(self, option: DigitalPut, market: MarketData):
        backend = market.backend
        _, d2 = compute_d1_d2(option.strike, option.expiry, market)
        discount = compute_discount_factor(option.expiry, market)
        N_minus_d2 = backend.norm_cdf(backend.mul(-1, d2))
        return backend.mul(backend.mul(option.payout, discount), N_minus_d2)

    def supports(self, option) -> bool:
        return isinstance(option, DigitalPut)

    def greeks(self, option: DigitalPut, market: MarketData, d1, d2) -> Greeks:
        return DigitalGreeks.compute_greeks(option, market, d1, d2, is_call=False)

    def price_with_greeks(self, option: DigitalPut, market: MarketData):
        d1, d2 = compute_d1_d2(option.strike, option.expiry, market)
        p = self.price(option, market)
        g = self.greeks(option, market, d1, d2)
        return p, g
