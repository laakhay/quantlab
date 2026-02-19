"""
Comprehensive tests for unified Pricer interface.
Tests automatic method selection, backend integration, and pricing workflows.

Ported to quantlab from vol-project.
"""

import numpy as np
import pytest

from laakhay.quantlab.backend import get_backend
from laakhay.quantlab.exceptions import BackendNotFoundError
from laakhay.quantlab.pricing import (
    EuropeanCall,
    GeometricAsianCall,
    MarketData,
    Pricer,
    UpAndOutCall,
)
from laakhay.quantlab.pricing.pricers.pricer import PricingMethod


class TestPricerInterface:
    """Test unified Pricer interface functionality."""

    @pytest.mark.parametrize("backend_name", ["numpy"])
    def test_pricer_initialization(self, backend_name):
        """Test Pricer initialization with different backends."""
        backend = get_backend(backend_name)
        pricer = Pricer(backend=backend)

        # In quantlab, backend might not have .name attribute, using type or repr
        assert backend_name in str(pricer.backend).lower()
        assert hasattr(pricer, "_bs_pricer")
        assert hasattr(pricer, "mc_pricer")

    def test_pricer_from_string(self):
        """Test Pricer initialization from backend string."""
        pricer = Pricer(backend="numpy")
        assert "numpy" in str(pricer.backend).lower()

        # Test invalid backend
        with pytest.raises(BackendNotFoundError):
            get_backend("invalid_backend")

    def test_pricer_configuration(self):
        """Test Pricer configuration options."""
        backend = get_backend("numpy")

        pricer = Pricer(backend=backend, market=MarketData(spot=110.0, backend=backend))

        # Check that configurations are accessible
        assert hasattr(pricer, "backend")
        assert pricer.market.spot == 110.0


class TestAutomaticMethodSelection:
    """Test automatic method selection logic."""

    def test_european_option_method_selection(self):
        """Test that European options use Black-Scholes by default."""
        backend = get_backend("numpy")
        pricer = Pricer(backend=backend)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)

        # Should automatically select Black-Scholes
        price = pricer.price(call, market=market)

        # Verify it's using analytical method (should be very precise)
        # We can check by comparing with explicit BS method
        pricer.method = PricingMethod.BLACK_SCHOLES
        bs_price = pricer.price(call, market=market)
        assert abs(float(price) - float(bs_price)) < 1e-10

    def test_barrier_option_method_selection(self):
        """Test that barrier options use Monte Carlo by default."""
        backend = get_backend("numpy")
        # Initialize with enough paths for stability
        pricer = Pricer(backend=backend)
        pricer.mc_pricer.n_paths = 10000
        pricer.mc_pricer.seed = 42

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        barrier_call = UpAndOutCall(strike=100.0, expiry=1.0, barrier=120.0)

        # Should automatically select Monte Carlo (BS doesn't support UpAndOutCall yet in registry)
        price = pricer.price(barrier_call, market=market)

        # Verify it's using Monte Carlo
        pricer.method = PricingMethod.MONTE_CARLO
        mc_price = pricer.price(barrier_call, market=market)
        assert abs(float(price) - float(mc_price)) < 1e-10

    def test_asian_option_method_selection(self):
        """Test Asian option method selection."""
        backend = get_backend("numpy")
        pricer = Pricer(backend=backend)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        # Geometric Asian should use Black-Scholes (analytical available)
        geo_asian = GeometricAsianCall(strike=100.0, expiry=1.0)
        price = pricer.price(geo_asian, market=market)

        # Should be analytical
        pricer.method = PricingMethod.BLACK_SCHOLES
        bs_price = pricer.price(geo_asian, market=market)
        assert abs(float(price) - float(bs_price)) < 1e-10


class TestMethodForcing:
    """Test explicit method specification."""

    def test_force_black_scholes(self):
        """Test forcing Black-Scholes method."""
        backend = get_backend("numpy")
        pricer = Pricer(backend=backend, method=PricingMethod.BLACK_SCHOLES)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)

        bs_price = pricer.price(call, market=market)
        assert float(bs_price) > 0

    def test_force_monte_carlo(self):
        """Test forcing Monte Carlo method."""
        backend = get_backend("numpy")
        pricer = Pricer(backend=backend, method=PricingMethod.MONTE_CARLO)
        pricer.mc_pricer.n_paths = 50000
        pricer.mc_pricer.seed = 42

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)

        mc_price = pricer.price(call, market=market)
        assert float(mc_price) > 0

        # Should be close to analytical but not exact
        pricer.method = PricingMethod.BLACK_SCHOLES
        bs_price = pricer.price(call, market=market)
        assert abs(float(mc_price) - float(bs_price)) < 0.5  # MC tolerance


class TestGreeksIntegration:
    """Test Greeks calculation through Pricer interface."""

    def test_greeks_calculation(self):
        """Test Greeks calculation via Pricer."""
        backend = get_backend("numpy")
        pricer = Pricer(backend=backend)

        market = MarketData(spot=100.0, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        # Using price_with_greeks method in quantlab
        _, greeks = pricer.price_with_greeks(call, market=market)

        # Should have all Greeks
        assert hasattr(greeks, "delta")
        assert hasattr(greeks, "gamma")
        assert hasattr(greeks, "vega")
        assert hasattr(greeks, "theta")
        assert hasattr(greeks, "rho")

        # All should be finite
        assert np.isfinite(float(greeks.delta))
        assert np.isfinite(float(greeks.gamma))
        assert np.isfinite(float(greeks.vega))
        assert np.isfinite(float(greeks.theta))
        assert np.isfinite(float(greeks.rho))


class TestImpliedVolatility:
    """Test implied volatility calculation through Pricer."""

    def test_implied_vol_european(self):
        """Test implied volatility for European options."""
        backend = get_backend("numpy")
        pricer = Pricer(backend=backend)

        true_vol = 0.25
        market = MarketData(spot=100.0, rate=0.05, vol=true_vol, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)

        # Get theoretical price
        theoretical_price = pricer.price(call, market=market)

        # Recover implied vol
        # In quantlab, implied_vol is on the BS pricer directly or through top level pricer if we add it
        # Top level pricer doesn't have implied_vol yet, adding it to BS pricer for now
        implied_vol = pricer._bs_pricer.implied_vol(call, theoretical_price, market=market)

        assert abs(float(implied_vol) - true_vol) < 1e-6


class TestVectorizedPricing:
    """Test vectorized pricing capabilities."""

    def test_vectorized_market_data(self):
        """Test pricing with vectorized market data."""
        backend = get_backend("numpy")
        pricer = Pricer(backend=backend)

        # Vector of spot prices
        spots = backend.convert([80.0, 90.0, 100.0, 110.0, 120.0])
        market = MarketData(spot=spots, rate=0.05, vol=0.2, backend=backend)

        call = EuropeanCall(strike=100.0, expiry=1.0)
        prices = pricer.price(call, market=market)

        # Should return vector of prices
        prices_np = backend.to_numpy(prices)
        assert len(prices_np) == 5

        # Prices should increase with spot price
        for i in range(len(prices_np) - 1):
            assert prices_np[i] <= prices_np[i + 1]
