"""Black-Scholes formulas for barrier options (Reiner & Rubinstein 1991)."""

from __future__ import annotations

from ....greeks import Greeks
from ....market import MarketData
from ....options.barrier import (
    DownAndInCall,
    DownAndInPut,
    DownAndOutCall,
    DownAndOutPut,
    UpAndInCall,
    UpAndInPut,
    UpAndOutCall,
    UpAndOutPut,
)
from ..calculations import compute_barrier_terms
from ..registry import PricingFormula


class BarrierFormula(PricingFormula):
    """Unified barrier option pricing using coefficient matrices."""

    # Coefficient matrices: [A, B, C, D] for [K <= H, K > H] cases
    COEFFICIENTS = {
        "UpAndOutCall": [[1, -1, 1, -1], [0, 0, 0, 0]],
        "UpAndOutPut": [[1, 0, -1, 0], [0, 1, 0, -1]],
        "DownAndOutCall": [[0, 1, 0, -1], [1, 0, -1, 0]],
        "DownAndOutPut": [[0, 0, 0, 0], [1, -1, 1, -1]],
        "UpAndInCall": [[0, 1, -1, 1], [1, 0, 0, 0]],
        "UpAndInPut": [[0, 0, 1, 0], [1, -1, 0, 1]],
        "DownAndInCall": [[1, -1, 0, 1], [0, 0, 1, 0]],
        "DownAndInPut": [[1, 0, 0, 0], [0, 1, -1, 1]],
    }

    def __init__(self, option_type: str, contract_class):
        self.option_type = option_type
        self.contract_class = contract_class
        self.coeffs = self.COEFFICIENTS[option_type]

    def price(self, option, market: MarketData):
        """Price barrier option using coefficient matrix."""
        backend = market.backend

        # Determine phi (Call: 1, Put: -1) and eta (Down: 1, Up: -1)
        phi = 1 if "Call" in self.option_type else -1
        eta = 1 if "Down" in self.option_type else -1

        terms = compute_barrier_terms(option, market, phi=phi, eta=eta)

        # Stack terms: [A, B, C, D]
        terms_array = backend.convert([terms["A"], terms["B"], terms["C"], terms["D"]])

        # Select coefficients based on K <= H condition
        condition = backend.array(option.strike <= option.barrier)
        coeffs_low = backend.array(self.coeffs[0])  # K <= H
        coeffs_high = backend.array(self.coeffs[1])  # K > H

        # selected_coeffs = backend.where(condition, coeffs_low, coeffs_high)
        # We need to handle vectorization if condition is a vector
        if backend.is_scalar(condition):
            selected_coeffs = coeffs_low if condition else coeffs_high
            return backend.sum(backend.mul(selected_coeffs, terms_array), axis=0)
        else:
            # Vectorized selection: this is tricky with the coefficients
            # We'll do it term by term
            price = backend.zeros_like(terms["A"])
            for i in range(4):
                c = backend.where(condition, coeffs_low[i], coeffs_high[i])
                price = backend.add(price, backend.mul(c, terms_array[i]))
            return price

    def supports(self, option) -> bool:
        return isinstance(option, self.contract_class)

    def price_with_greeks(self, option, market: MarketData) -> tuple[object, Greeks]:
        """Price barrier option and compute Greeks (fallback to zeros for now)."""
        price = self.price(option, market)
        # TODO: Implement finite difference greeks if needed
        zero = market.backend.zeros_like(price) if hasattr(market.backend, "zeros_like") else 0.0
        greeks = Greeks(delta=zero, gamma=zero, vega=zero, theta=zero, rho=zero)
        return price, greeks


# Create all formula instances
UpAndOutCallFormula = BarrierFormula("UpAndOutCall", UpAndOutCall)
UpAndOutPutFormula = BarrierFormula("UpAndOutPut", UpAndOutPut)
DownAndOutCallFormula = BarrierFormula("DownAndOutCall", DownAndOutCall)
DownAndOutPutFormula = BarrierFormula("DownAndOutPut", DownAndOutPut)
UpAndInCallFormula = BarrierFormula("UpAndInCall", UpAndInCall)
UpAndInPutFormula = BarrierFormula("UpAndInPut", UpAndInPut)
DownAndInCallFormula = BarrierFormula("DownAndInCall", DownAndInCall)
DownAndInPutFormula = BarrierFormula("DownAndInPut", DownAndInPut)
