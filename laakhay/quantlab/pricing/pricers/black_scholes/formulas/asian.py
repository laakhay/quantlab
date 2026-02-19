"""Black-Scholes formulas for geometric Asian options."""

from ....greeks import Greeks
from ....market import MarketData
from ....options.asian import GeometricAsianCall, GeometricAsianPut
from ..calculations import compute_asian_adjusted_params, compute_d1_d2, compute_discount_factor
from ..registry import PricingFormula


def _normal_pdf(x, backend):
    """Calculate normal probability density function φ(x)."""
    sqrt_2pi = backend.sqrt(backend.mul(2.0, backend.pi))
    exp_term = backend.exp(backend.mul(-0.5, backend.power(x, 2)))
    return backend.divide(exp_term, sqrt_2pi)


class AsianGreeks:
    """Modular Greeks computation for Geometric Asian options."""

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

        # Get adjusted parameters for Asian option
        vol_hat, drift_hat = compute_asian_adjusted_params(market)

        sqrt_T = backend.sqrt(backend.array(option.expiry))
        discount = compute_discount_factor(option.expiry, market)

        phi_d1 = _normal_pdf(d1, backend)
        N_d1 = backend.norm_cdf(d1)
        N_d2 = backend.norm_cdf(d2)

        # Adjusted spot price: S₀ * exp(drift_hat * T)
        adjusted_spot = backend.mul(market.spot, backend.exp(backend.mul(drift_hat, option.expiry)))

        if is_call:
            # Delta uses adjusted vol but original spot
            delta = backend.mul(
                discount, backend.mul(backend.exp(backend.mul(drift_hat, option.expiry)), N_d1)
            )

            # Rho calculation
            rho = backend.divide(
                backend.mul(backend.mul(option.strike, option.expiry), backend.mul(discount, N_d2)),
                100.0,
            )

            # Theta calculation
            theta_term1 = backend.divide(
                backend.mul(backend.mul(adjusted_spot, phi_d1), vol_hat), backend.mul(2.0, sqrt_T)
            )
            theta_term2 = backend.mul(
                backend.mul(market.rate, backend.mul(option.strike, discount)), N_d2
            )
            theta = backend.divide(
                backend.mul(-1.0, backend.mul(discount, backend.add(theta_term1, theta_term2))),
                365.0,
            )
        else:
            N_minus_d1 = backend.norm_cdf(backend.mul(-1.0, d1))
            N_minus_d2 = backend.norm_cdf(backend.mul(-1.0, d2))

            delta = backend.mul(
                -1.0,
                backend.mul(
                    discount,
                    backend.mul(backend.exp(backend.mul(drift_hat, option.expiry)), N_minus_d1),
                ),
            )

            rho = backend.divide(
                backend.mul(
                    -1.0,
                    backend.mul(
                        backend.mul(option.strike, option.expiry), backend.mul(discount, N_minus_d2)
                    ),
                ),
                100.0,
            )

            theta_term1 = backend.divide(
                backend.mul(backend.mul(adjusted_spot, phi_d1), vol_hat), backend.mul(2.0, sqrt_T)
            )
            theta_term2 = backend.mul(
                backend.mul(market.rate, backend.mul(option.strike, discount)), N_minus_d2
            )
            theta = backend.divide(
                backend.mul(discount, backend.add(backend.mul(-1.0, theta_term1), theta_term2)),
                365.0,
            )

        gamma = backend.divide(
            backend.mul(
                backend.mul(discount, phi_d1), backend.exp(backend.mul(drift_hat, option.expiry))
            ),
            backend.mul(backend.mul(market.spot, vol_hat), sqrt_T),
        )

        vega = backend.divide(
            backend.mul(backend.mul(backend.mul(discount, adjusted_spot), phi_d1), sqrt_T), 100.0
        )

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


class GeometricAsianCallFormula(PricingFormula):
    """Geometric Asian call pricing formula."""

    def price(self, option: GeometricAsianCall, market: MarketData):
        backend = market.backend
        vol_hat, drift_hat = compute_asian_adjusted_params(market)
        adjusted_market = market.with_vol(vol_hat)
        d1, d2 = compute_d1_d2(option.strike, option.expiry, adjusted_market, drift_hat)
        discount = compute_discount_factor(option.expiry, market)
        N_d1 = backend.norm_cdf(d1)
        N_d2 = backend.norm_cdf(d2)
        adjusted_spot = backend.mul(market.spot, backend.exp(backend.mul(drift_hat, option.expiry)))
        return backend.mul(
            discount,
            backend.add(backend.mul(adjusted_spot, N_d1), backend.mul(-option.strike, N_d2)),
        )

    def supports(self, option) -> bool:
        return isinstance(option, GeometricAsianCall)

    def greeks(self, option: GeometricAsianCall, market: MarketData, d1, d2) -> Greeks:
        return AsianGreeks.compute_greeks(option, market, d1, d2, is_call=True)

    def price_with_greeks(self, option: GeometricAsianCall, market: MarketData):
        vol_hat, drift_hat = compute_asian_adjusted_params(market)
        adjusted_market = market.with_vol(vol_hat)
        d1, d2 = compute_d1_d2(option.strike, option.expiry, adjusted_market, drift_hat)
        p = self.price(option, market)
        g = self.greeks(option, market, d1, d2)
        return p, g


class GeometricAsianPutFormula(PricingFormula):
    """Geometric Asian put pricing formula."""

    def price(self, option: GeometricAsianPut, market: MarketData):
        backend = market.backend
        vol_hat, drift_hat = compute_asian_adjusted_params(market)
        adjusted_market = market.with_vol(vol_hat)
        d1, d2 = compute_d1_d2(option.strike, option.expiry, adjusted_market, drift_hat)
        discount = compute_discount_factor(option.expiry, market)
        N_minus_d1 = backend.norm_cdf(backend.mul(-1.0, d1))
        N_minus_d2 = backend.norm_cdf(backend.mul(-1.0, d2))
        adjusted_spot = backend.mul(market.spot, backend.exp(backend.mul(drift_hat, option.expiry)))
        return backend.mul(
            discount,
            backend.add(
                backend.mul(option.strike, N_minus_d2), backend.mul(-adjusted_spot, N_minus_d1)
            ),
        )

    def supports(self, option) -> bool:
        return isinstance(option, GeometricAsianPut)

    def greeks(self, option: GeometricAsianPut, market: MarketData, d1, d2) -> Greeks:
        return AsianGreeks.compute_greeks(option, market, d1, d2, is_call=False)

    def price_with_greeks(self, option: GeometricAsianPut, market: MarketData):
        vol_hat, drift_hat = compute_asian_adjusted_params(market)
        adjusted_market = market.with_vol(vol_hat)
        d1, d2 = compute_d1_d2(option.strike, option.expiry, adjusted_market, drift_hat)
        p = self.price(option, market)
        g = self.greeks(option, market, d1, d2)
        return p, g
