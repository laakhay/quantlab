"""Common calculation utilities for Black-Scholes pricing."""

from ...market import MarketData


def compute_d1_d2(
    strike: float, expiry: float, market: MarketData, drift: object | None = None
) -> tuple[object, object]:
    """Compute d1 and d2 for Black-Scholes formulas."""
    backend = market.backend

    d = drift if drift is not None else market.rate
    expiry_val = max(expiry, 1e-8)

    # Convert scalars to backend format
    strike_arr = backend.array(strike)
    expiry_arr = backend.array(expiry_val)

    sqrt_t = backend.mul(market.vol, backend.sqrt(expiry_arr))

    log_moneyness = backend.log(backend.divide(market.spot, strike_arr))
    drift_term = backend.mul(
        backend.add(d, backend.mul(0.5, backend.power(market.vol, 2))), expiry_arr
    )

    d1 = backend.divide(backend.add(log_moneyness, drift_term), sqrt_t)
    d2 = backend.add(d1, backend.mul(-1, sqrt_t))

    return d1, d2


def compute_discount_factor(expiry: float, market: MarketData) -> object:
    """Compute risk-free discount factor."""
    backend = market.backend
    expiry_arr = backend.array(expiry)
    return backend.exp(backend.mul(backend.mul(-1, market.rate), expiry_arr))


def compute_barrier_terms(
    option, market: MarketData, phi: int = 1, eta: int = 1
) -> dict[str, object]:
    """Compute Reiner & Rubinstein barrier option terms A, B, C, D."""
    backend = market.backend
    K = option.strike
    H = option.barrier
    T = option.expiry

    if T <= 0 or backend.any(backend.less_equal(market.vol, 0.0)):
        if not backend.is_scalar(market.spot):
            shape = backend.shape(market.spot)
            return {
                "A": backend.zeros(shape),
                "B": backend.zeros(shape),
                "C": backend.zeros(shape),
                "D": backend.zeros(shape),
            }
        return {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

    sigma_sq = backend.power(market.vol, 2)
    mu = backend.divide(backend.add(market.rate, backend.mul(-0.5, sigma_sq)), sigma_sq)

    d1, d2 = compute_d1_d2(K, T, market)
    d3, d4 = compute_d1_d2(H, T, market)

    h_sq_over_s = backend.divide(backend.power(H, 2), market.spot)

    d5, d6 = compute_d1_d2(K, T, market.with_spot(h_sq_over_s))
    d7, d8 = compute_d1_d2(H, T, market.with_spot(h_sq_over_s))

    h_over_s = backend.divide(H, market.spot)

    # Correct powers for C and D terms
    # Term C spot factor: (H/S)^(2*mu + 2)
    # Term C strike factor: (H/S)^(2*mu)
    pow_spot = backend.add(backend.mul(2.0, mu), 2.0)
    pow_strike = backend.mul(2.0, mu)

    phi_arr = backend.array(float(phi))
    eta_arr = backend.array(float(eta))

    N_phi_d1, N_phi_d2 = (
        backend.norm_cdf(backend.mul(phi_arr, d1)),
        backend.norm_cdf(backend.mul(phi_arr, d2)),
    )
    N_phi_d3, N_phi_d4 = (
        backend.norm_cdf(backend.mul(phi_arr, d3)),
        backend.norm_cdf(backend.mul(phi_arr, d4)),
    )
    N_eta_d5, N_eta_d6 = (
        backend.norm_cdf(backend.mul(eta_arr, d5)),
        backend.norm_cdf(backend.mul(eta_arr, d6)),
    )
    N_eta_d7, N_eta_d8 = (
        backend.norm_cdf(backend.mul(eta_arr, d7)),
        backend.norm_cdf(backend.mul(eta_arr, d8)),
    )

    discount = compute_discount_factor(T, market)

    A = backend.add(
        backend.mul(market.spot, N_phi_d1), backend.mul(backend.mul(-K, discount), N_phi_d2)
    )
    B = backend.add(
        backend.mul(market.spot, N_phi_d3), backend.mul(backend.mul(-K, discount), N_phi_d4)
    )

    reflection_spot = backend.mul(market.spot, backend.power(h_over_s, pow_spot))
    reflection_strike = backend.mul(backend.mul(K, discount), backend.power(h_over_s, pow_strike))

    C = backend.add(
        backend.mul(reflection_spot, N_eta_d5),
        backend.mul(backend.mul(-1.0, reflection_strike), N_eta_d6),
    )
    D = backend.add(
        backend.mul(reflection_spot, N_eta_d7),
        backend.mul(backend.mul(-1.0, reflection_strike), N_eta_d8),
    )

    return {"A": A, "B": B, "C": C, "D": D}


def compute_asian_adjusted_params(market: MarketData) -> tuple[object, object]:
    """Compute adjusted vol and drift for geometric Asian options."""
    backend = market.backend

    vol_hat = backend.divide(market.vol, backend.sqrt(backend.array(3.0)))

    original_drift = backend.add(market.rate, backend.mul(-0.5, backend.power(market.vol, 2)))
    vol_hat_sq = backend.power(vol_hat, 2)

    drift_hat = backend.add(backend.mul(0.5, original_drift), backend.mul(0.5, vol_hat_sq))

    return vol_hat, drift_hat


def _normal_pdf(x, backend):
    """Calculate normal probability density function φ(x)."""
    sqrt_2pi = backend.sqrt(backend.mul(2.0, backend.pi))
    exp_term = backend.exp(backend.mul(-0.5, backend.power(x, 2)))
    return backend.divide(exp_term, sqrt_2pi)
