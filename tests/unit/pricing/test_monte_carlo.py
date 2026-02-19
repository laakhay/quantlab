"""
Comprehensive tests for Monte Carlo pricing engine.
Tests simulation paths, shock caching, convergence, and numerical stability.

Ported to quantlab from vol-project.
"""

import numpy as np
import pytest

from laakhay.quantlab.backend import get_backend
from laakhay.quantlab.pricing import (
    DigitalCall,
    DigitalPut,
    EuropeanCall,
    EuropeanPut,
    GeometricAsianCall,
    GreeksEngine,
    MarketData,
    MonteCarloPricer,
)


class TestMonteCarloCore:
    """Test core Monte Carlo pricing functionality."""

    @pytest.mark.parametrize("backend_name", ["numpy"])
    def test_basic_european_pricing(self, backend_name):
        """Test basic European option pricing with Monte Carlo."""
        backend = get_backend(backend_name)
        mc_engine = MonteCarloPricer(backend=backend, n_paths=50000, seed=42)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        mc_price = float(mc_engine.price(call, market=market))

        # Should be close to Black-Scholes theoretical price (~10.45)
        expected_range = (9.5, 11.5)
        assert expected_range[0] < mc_price < expected_range[1]

        # Test put as well
        put = EuropeanPut(strike=100.0, expiry=1.0)
        put_price = float(mc_engine.price(put, market=market))

        # Put-call parity check (with tolerance for MC error)
        parity_diff = mc_price - put_price
        expected_diff = 100.0 - 100.0 * np.exp(-0.05 * 1.0)
        assert abs(parity_diff - expected_diff) < 0.5  # Monte Carlo tolerance

    def test_path_generation_consistency(self):
        """Test path generation is consistent with same seed."""
        backend = get_backend("numpy")

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        # Generate paths with same seed twice
        mc1 = MonteCarloPricer(backend=backend, n_paths=1000, seed=123)
        mc2 = MonteCarloPricer(backend=backend, n_paths=1000, seed=123)

        call = EuropeanCall(strike=100.0, expiry=1.0)

        price1 = float(mc1.price(call, market=market))
        price2 = float(mc2.price(call, market=market))

        # Should be identical with same seed and same implementation
        assert abs(price1 - price2) < 1e-10

    def test_convergence_with_path_count(self):
        """Test pricing convergence as path count increases."""
        backend = get_backend("numpy")

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)

        path_counts = [1000, 5000, 25000]
        prices = []

        for n_paths in path_counts:
            mc_engine = MonteCarloPricer(backend=backend, n_paths=n_paths, seed=42)
            price = float(mc_engine.price(call, market=market))
            prices.append(price)

        # Variance should decrease (prices more stable) with more paths
        price_std = np.std(prices)
        assert price_std < 1.0  # Should be reasonably stable

        # Final price should be in reasonable range
        assert 9.5 < prices[-1] < 11.5

    def test_multiple_expiry_pricing(self):
        """Test pricing options with different expiries."""
        backend = get_backend("numpy")
        mc_engine = MonteCarloPricer(backend=backend, n_paths=20000, seed=42)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        expiries = [0.25, 0.5, 1.0, 2.0]
        prices = []

        for expiry in expiries:
            call = EuropeanCall(strike=100.0, expiry=expiry)
            price = float(mc_engine.price(call, market=market))
            prices.append(price)

        # Prices should increase with time (more time value)
        for i in range(len(prices) - 1):
            assert prices[i] <= prices[i + 1]

        # Short expiry should be smaller than long expiry
        assert prices[0] < prices[-1]

        # Long expiry should be substantial
        assert prices[-1] > 8.0


class TestMonteCarloGreeks:
    """Test Greeks calculation via finite differences."""

    def test_delta_calculation(self):
        """Test delta calculation via finite differences."""
        backend = get_backend("numpy")
        engine = GreeksEngine(backend=backend)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        greeks = engine.calculate(call, market=market, method="finite_diff")

        # Delta should be between 0 and 1 for calls
        assert 0 < greeks.delta < 1

        # ATM call delta should be around 0.5-0.6
        assert 0.4 < greeks.delta < 0.7

    def test_gamma_calculation(self):
        """Test gamma calculation via finite differences."""
        backend = get_backend("numpy")
        engine = GreeksEngine(backend=backend)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        greeks = engine.calculate(call, market=market, method="finite_diff")

        # Gamma should be positive
        assert greeks.gamma > 0

        # Should be reasonably sized for ATM option
        assert 0.005 < greeks.gamma < 0.05

    def test_vega_calculation(self):
        """Test vega calculation via finite differences."""
        backend = get_backend("numpy")
        engine = GreeksEngine(backend=backend)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        greeks = engine.calculate(call, market=market, method="finite_diff")

        # Vega should be positive
        assert greeks.vega > 0

        # Should be positive for ATM option with 1 year to expiry
        assert greeks.vega > 0.1


class TestDigitalOptionsMonteCarlo:
    """Test digital option pricing via Monte Carlo."""

    def test_digital_call_pricing(self):
        """Test digital call pricing convergence."""
        backend = get_backend("numpy")
        mc_engine = MonteCarloPricer(backend=backend, n_paths=100000, seed=42)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        digital_call = DigitalCall(strike=100.0, expiry=1.0, payout=1.0)
        price = float(mc_engine.price(digital_call, market=market))

        # Should be between 0 and 1
        assert 0 < price < 1.0

        # Should be close to analytical result (~0.54 for these parameters)
        assert 0.45 < price < 0.65

    def test_digital_put_pricing(self):
        """Test digital put pricing."""
        backend = get_backend("numpy")
        mc_engine = MonteCarloPricer(backend=backend, n_paths=100000, seed=42)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        digital_put = DigitalPut(strike=100.0, expiry=1.0, payout=1.0)
        price = float(mc_engine.price(digital_put, market=market))

        assert 0 < price < 1.0

    def test_digital_parity(self):
        """Test digital call + put parity."""
        backend = get_backend("numpy")
        mc_engine = MonteCarloPricer(backend=backend, n_paths=100000, seed=42)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        digital_call = DigitalCall(strike=100.0, expiry=1.0, payout=1.0)
        digital_put = DigitalPut(strike=100.0, expiry=1.0, payout=1.0)

        call_price = float(mc_engine.price(digital_call, market=market))
        put_price = float(mc_engine.price(digital_put, market=market))

        # Should sum to discounted payout (with MC tolerance)
        total_price = call_price + put_price
        expected_total = np.exp(-0.05 * 1.0)
        assert abs(total_price - expected_total) < 0.05  # MC tolerance


class TestAsianOptionsMonteCarlo:
    """Test Asian option pricing via Monte Carlo."""

    def test_geometric_asian_pricing(self):
        """Test geometric Asian option pricing."""
        backend = get_backend("numpy")
        mc_engine = MonteCarloPricer(backend=backend, n_paths=50000, seed=42)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        asian_call = GeometricAsianCall(strike=100.0, expiry=1.0)
        mc_price = float(mc_engine.price(asian_call, market=market))

        # Should be cheaper than European due to volatility reduction
        european_call = EuropeanCall(strike=100.0, expiry=1.0)
        european_price = float(mc_engine.price(european_call, market=market))

        assert mc_price < european_price
        assert mc_price > 0

        # Should be in reasonable range
        assert 5.0 < mc_price < 12.0


class TestMonteCarloStability:
    """Test Monte Carlo numerical stability."""

    def test_extreme_volatility(self):
        """Test pricing with extreme volatility."""
        backend = get_backend("numpy")
        mc_engine = MonteCarloPricer(backend=backend, n_paths=10000, seed=42)

        # Very high volatility
        market_high_vol = MarketData(spot=100.0, rate=0.05, vol=2.0, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        price_high_vol = mc_engine.price(call, market=market_high_vol)

        assert backend.all(backend.isfinite(price_high_vol))
        assert float(price_high_vol) > 0

        # Very low volatility
        market_low_vol = MarketData(spot=100.0, rate=0.05, vol=0.001, backend=backend)

        price_low_vol = mc_engine.price(call, market=market_low_vol)

        assert backend.all(backend.isfinite(price_low_vol))
        assert float(price_low_vol) >= 0

    def test_extreme_moneyness(self):
        """Test pricing with extreme moneyness."""
        backend = get_backend("numpy")
        mc_engine = MonteCarloPricer(backend=backend, n_paths=10000, seed=42)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        # Deep ITM call
        deep_itm = EuropeanCall(strike=10.0, expiry=1.0)
        itm_price = mc_engine.price(deep_itm, market=market)

        assert backend.all(backend.isfinite(itm_price))

        # Should be close to intrinsic value
        intrinsic = 100.0 - 10.0
        assert float(itm_price) > intrinsic * 0.9

        # Deep OTM call
        deep_otm = EuropeanCall(strike=1000.0, expiry=1.0)
        otm_price = mc_engine.price(deep_otm, market=market)

        assert backend.all(backend.isfinite(otm_price))
        assert float(otm_price) >= 0
        assert float(otm_price) < 1.0  # Should be very small

    def test_vectorized_pricing_stability(self):
        """Test vectorized pricing robustness."""
        backend = get_backend("numpy")
        mc_engine = MonteCarloPricer(backend=backend, n_paths=5000, seed=42)

        # Vector of spot prices
        spots = backend.convert([50.0, 75.0, 100.0, 125.0, 150.0])
        market = MarketData(spot=spots, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        prices = mc_engine.price(call, market=market)

        # All prices should be finite and non-negative
        assert backend.all(backend.isfinite(prices))
        assert backend.all(backend.ge(prices, 0))

        # Prices should generally increase with spot
        prices_np = backend.to_numpy(prices)

        # Allow for some MC noise, but general trend should be upward
        assert prices_np[-1] > prices_np[0]  # Highest > Lowest
        assert prices_np[2] > prices_np[0]  # ATM > Deep OTM


class TestShockCaching:
    """Test shock caching functionality."""

    def test_shock_cache_consistency(self):
        """Test that shock caching produces consistent results."""
        backend = get_backend("numpy")

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)

        # Price with caching
        mc_cached = MonteCarloPricer(backend=backend, n_paths=10000, seed=42)
        price_cached = float(mc_cached.price(call, market=market))

        # Price same option again (should use cache)
        price_cached_2 = float(mc_cached.price(call, market=market))

        # Should be identical due to caching
        assert abs(price_cached - price_cached_2) < 1e-10
