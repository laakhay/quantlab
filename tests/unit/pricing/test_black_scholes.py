"""
Comprehensive tests for Black-Scholes pricing engine.
Tests analytical formulas, Greeks calculations, and numerical stability.
"""

import numpy as np
import pytest

from laakhay.quantlab.backend import get_backend
from laakhay.quantlab.pricing import (
    BlackScholesPricer,
    DigitalCall,
    DigitalPut,
    EuropeanCall,
    EuropeanPut,
    GeometricAsianCall,
    GreeksEngine,
    MarketData,
)


class TestBlackScholesCore:
    """Test core Black-Scholes pricing functionality."""

    @pytest.mark.parametrize("backend_name", ["numpy"])
    def test_european_option_pricing(self, backend_name):
        """Test European option pricing accuracy."""
        backend = get_backend(backend_name)
        pricer = BlackScholesPricer(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        # Test call option
        call = EuropeanCall(strike=100.0, expiry=1.0)
        call_price = float(pricer.price(call, market=market))

        # BS formula for ATM call: roughly 0.4 * S * vol * sqrt(T)
        expected_range = (5.0, 15.0)  # Reasonable range for these parameters
        assert expected_range[0] < call_price < expected_range[1]

        # Test put option
        put = EuropeanPut(strike=100.0, expiry=1.0)
        put_price = float(pricer.price(put, market=market))

        # Put-call parity: C - P = S - K*exp(-r*T)
        parity_diff = call_price - put_price
        expected_diff = 100.0 - 100.0 * np.exp(-0.05 * 1.0)
        assert abs(parity_diff - expected_diff) < 1e-10

    def test_moneyness_sensitivity(self):
        """Test pricing sensitivity to moneyness."""
        backend = get_backend("numpy")
        pricer = BlackScholesPricer(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)

        # Test different spot prices
        spots = [80.0, 90.0, 100.0, 110.0, 120.0]
        prices = []

        for spot in spots:
            market_spot = market.with_spot(spot)
            price = float(pricer.price(call, market=market_spot))
            prices.append(price)

        # Prices should be monotonically increasing with spot
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1]

        # Deep OTM should be relatively small
        assert prices[0] < 3.0

        # Deep ITM should be close to intrinsic + small time value
        intrinsic = max(120.0 - 100.0, 0)
        assert prices[-1] > intrinsic
        assert prices[-1] < intrinsic + 10.0  # Reasonable time value bound

    def test_time_decay(self):
        """Test time decay (theta) behavior."""
        backend = get_backend("numpy")
        pricer = BlackScholesPricer(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        # Test different times to expiry
        expiries = [0.01, 0.1, 0.25, 0.5, 1.0]
        prices = []

        for expiry in expiries:
            call_t = EuropeanCall(strike=100.0, expiry=expiry)
            price = float(pricer.price(call_t, market=market))
            prices.append(price)

        # Prices should increase with time (more time value)
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1]

    def test_volatility_sensitivity(self):
        """Test volatility sensitivity (vega)."""
        backend = get_backend("numpy")
        pricer = BlackScholesPricer(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.1, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)

        # Test different volatilities
        vols = [0.1, 0.2, 0.3, 0.4, 0.5]
        prices = []

        for vol in vols:
            market_vol = market.with_vol(vol)
            price = float(pricer.price(call, market=market_vol))
            prices.append(price)

        # Prices should increase monotonically with volatility
        for i in range(len(prices) - 1):
            assert prices[i] < prices[i + 1]


class TestBlackScholesGreeks:
    """Test Greeks calculations."""

    def test_delta_properties(self):
        """Test delta calculation and properties."""
        backend = get_backend("numpy")
        engine = GreeksEngine(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        # Test call delta
        call = EuropeanCall(strike=100.0, expiry=1.0)
        call_greeks = engine.calculate(call, market=market, method="analytical")

        # Call delta should be between 0 and 1
        assert 0 < call_greeks.delta < 1

        # ATM call delta should be around 0.5-0.6
        assert 0.4 < call_greeks.delta < 0.7

        # Test put delta
        put = EuropeanPut(strike=100.0, expiry=1.0)
        put_greeks = engine.calculate(put, market=market, method="analytical")

        # Put delta should be negative
        assert put_greeks.delta < 0

        # Put-call delta relationship: C_delta - P_delta = 1
        delta_diff = call_greeks.delta - put_greeks.delta
        assert abs(delta_diff - 1.0) < 1e-10

    def test_gamma_properties(self):
        """Test gamma calculation and properties."""
        backend = get_backend("numpy")
        engine = GreeksEngine(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        put = EuropeanPut(strike=100.0, expiry=1.0)

        call_greeks = engine.calculate(call, market=market, method="analytical")
        put_greeks = engine.calculate(put, market=market, method="analytical")

        # Gamma should be positive for both calls and puts
        assert call_greeks.gamma > 0
        assert put_greeks.gamma > 0

        # Call and put gamma should be equal
        assert abs(call_greeks.gamma - put_greeks.gamma) < 1e-10

    def test_vega_properties(self):
        """Test vega calculation and properties."""
        backend = get_backend("numpy")
        engine = GreeksEngine(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        put = EuropeanPut(strike=100.0, expiry=1.0)

        call_greeks = engine.calculate(call, market=market, method="analytical")
        put_greeks = engine.calculate(put, market=market, method="analytical")

        # Vega should be positive for both calls and puts
        assert call_greeks.vega > 0
        assert put_greeks.vega > 0

    def test_theta_properties(self):
        """Test theta calculation and properties."""
        backend = get_backend("numpy")
        engine = GreeksEngine(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        put = EuropeanPut(strike=100.0, expiry=1.0)

        call_greeks = engine.calculate(call, market=market, method="analytical")
        put_greeks = engine.calculate(put, market=market, method="analytical")

        # Theta should generally be negative
        assert call_greeks.theta < 0
        assert put_greeks.theta < 0

    def test_rho_properties(self):
        """Test rho calculation and properties."""
        backend = get_backend("numpy")
        engine = GreeksEngine(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        put = EuropeanPut(strike=100.0, expiry=1.0)

        call_greeks = engine.calculate(call, market=market, method="analytical")
        put_greeks = engine.calculate(put, market=market, method="analytical")

        # Call rho should be positive
        assert call_greeks.rho > 0

        # Put rho should be negative
        assert put_greeks.rho < 0


class TestDigitalOptions:
    """Test digital/binary option pricing."""

    def test_digital_call_pricing(self):
        """Test digital call option pricing."""
        backend = get_backend("numpy")
        pricer = BlackScholesPricer(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        digital_call = DigitalCall(strike=100.0, expiry=1.0, payout=1.0)
        price = float(pricer.price(digital_call, market=market))

        assert 0 < price < 1.0
        assert 0.3 < price < 0.7

    def test_digital_parity(self):
        """Test digital call + put parity."""
        backend = get_backend("numpy")
        pricer = BlackScholesPricer(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        digital_call = DigitalCall(strike=100.0, expiry=1.0, payout=1.0)
        digital_put = DigitalPut(strike=100.0, expiry=1.0, payout=1.0)

        call_price = float(pricer.price(digital_call, market=market))
        put_price = float(pricer.price(digital_put, market=market))

        total_price = call_price + put_price
        discounted_payout = 1.0 * np.exp(-0.05 * 1.0)
        assert abs(total_price - discounted_payout) < 1e-10


class TestAsianOptions:
    """Test Asian option pricing."""

    def test_geometric_asian_pricing(self):
        """Test geometric Asian option pricing."""
        backend = get_backend("numpy")
        pricer = BlackScholesPricer(backend=backend)

        market = MarketData.create(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        asian_call = GeometricAsianCall(strike=100.0, expiry=1.0)
        price = float(pricer.price(asian_call, market=market))

        european_call = EuropeanCall(strike=100.0, expiry=1.0)
        european_price = float(pricer.price(european_call, market=market))

        assert price < european_price
        assert price > 0


class TestImpliedVolatility:
    """Test implied volatility calculations."""

    def test_iv_recovery_european(self):
        """Test implied volatility recovery for European options."""
        backend = get_backend("numpy")
        pricer = BlackScholesPricer(backend=backend)

        true_vol = 0.25
        market = MarketData.create(spot=100.0, rate=0.05, vol=true_vol, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        true_price = pricer.price(call, market=market)

        initial_market = market.with_vol(0.2)
        recovered_vol = float(pricer.implied_vol(call, true_price, market=initial_market))

        assert abs(recovered_vol - true_vol) < 1e-6


class TestNumericalStability:
    """Test numerical stability and edge cases."""

    def test_extreme_parameters(self):
        """Test pricing with extreme parameters."""
        backend = get_backend("numpy")
        pricer = BlackScholesPricer(backend=backend)

        market_high_vol = MarketData.create(spot=100.0, rate=0.05, vol=2.0, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        price_high_vol = pricer.price(call, market=market_high_vol)

        assert np.isfinite(float(price_high_vol))
        assert float(price_high_vol) > 0

    def test_vectorized_pricing(self):
        """Test vectorized pricing robustness."""
        backend = get_backend("numpy")
        pricer = BlackScholesPricer(backend=backend)

        spots = backend.array([50.0, 75.0, 100.0, 125.0, 150.0])
        market = MarketData.create(spot=spots, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        prices = pricer.price(call, market=market)

        assert backend.all(backend.isfinite(prices))
        assert backend.all(backend.ge(prices, 0))
