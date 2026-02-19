"""
Comprehensive tests for option contracts in QuantLab.
Tests all contract types, payoffs, properties, and edge cases.
"""

import pytest
import numpy as np

from laakhay.quantlab.backend import get_backend
from laakhay.quantlab.pricing.options import (
    EuropeanCall,
    EuropeanPut,
    DigitalCall,
    DigitalPut,
    GeometricAsianCall,
    GeometricAsianPut,
    UpAndOutCall,
    UpAndInCall,
    DownAndOutCall,
    DownAndInCall,
    UpAndOutPut,
    UpAndInPut,
    DownAndOutPut,
    DownAndInPut,
    Side,
    PayoffType,
)


class TestEuropeanOptions:
    """Test European vanilla options."""

    @pytest.mark.parametrize("backend_name", ["numpy"])
    def test_european_call_payoffs(self, backend_name):
        """Test European call payoffs across scenarios."""
        backend = get_backend(backend_name)
        strike = 100.0
        call = EuropeanCall(strike=strike, expiry=1.0)

        scenarios = [
            (80.0, 0.0),  # OTM
            (100.0, 0.0),  # ATM
            (120.0, 20.0),  # ITM
        ]

        for spot, expected in scenarios:
            path = backend.convert([[spot]])
            payoff = backend.item(call(path, backend=backend))
            assert abs(payoff - expected) < 1e-10

    @pytest.mark.parametrize("backend_name", ["numpy"])
    def test_european_put_payoffs(self, backend_name):
        """Test European put payoffs across scenarios."""
        backend = get_backend(backend_name)
        strike = 100.0
        put = EuropeanPut(strike=strike, expiry=1.0)

        scenarios = [
            (80.0, 20.0),  # ITM
            (100.0, 0.0),  # ATM
            (120.0, 0.0),  # OTM
        ]

        for spot, expected in scenarios:
            path = backend.convert([[spot]])
            payoff = backend.item(put(path, backend=backend))
            assert abs(payoff - expected) < 1e-10

    def test_put_call_parity(self):
        """Test put-call parity holds at expiry."""
        backend = get_backend("numpy")
        strike = 100.0
        expiry = 1.0

        call = EuropeanCall(strike=strike, expiry=expiry)
        put = EuropeanPut(strike=strike, expiry=expiry)

        spots = [80.0, 90.0, 100.0, 110.0, 120.0]

        for spot in spots:
            path = backend.convert([[spot]])
            call_payoff = backend.item(call(path, backend=backend))
            put_payoff = backend.item(put(path, backend=backend))

            # At expiry: C - P = S - K
            parity_diff = call_payoff - put_payoff
            expected_diff = spot - strike
            assert abs(parity_diff - expected_diff) < 1e-10

    def test_vectorized_pricing(self):
        """Test vectorized pricing with multiple spots."""
        backend = get_backend("numpy")
        call = EuropeanCall(strike=100.0, expiry=1.0)

        spots = backend.convert([[80.0], [90.0], [100.0], [110.0], [120.0]])
        payoffs = call(spots, backend=backend)

        assert backend.shape(payoffs) == (5,)
        payoffs_np = backend.to_numpy(payoffs)
        expected = [0.0, 0.0, 0.0, 10.0, 20.0]

        for i, exp in enumerate(expected):
            assert abs(payoffs_np[i] - exp) < 1e-10


class TestDigitalOptions:
    """Test digital/binary options."""

    @pytest.mark.parametrize("backend_name", ["numpy"])
    def test_digital_call_payoffs(self, backend_name):
        """Test digital call payoffs."""
        backend = get_backend(backend_name)
        strike = 100.0
        payout = 10.0

        digital_call = DigitalCall(strike=strike, expiry=1.0, payout=payout)

        scenarios = [
            (95.0, 0.0),  # Below strike
            (100.0, 0.0),  # At strike (boundary case - not triggered)
            (105.0, 10.0),  # Above strike
        ]

        for spot, expected in scenarios:
            path = backend.convert([[spot]])
            payoff = backend.item(digital_call(path, backend=backend))
            assert abs(payoff - expected) < 1e-10

    @pytest.mark.parametrize("backend_name", ["numpy"])
    def test_digital_put_payoffs(self, backend_name):
        """Test digital put payoffs."""
        backend = get_backend(backend_name)
        strike = 100.0
        payout = 5.0

        digital_put = DigitalPut(strike=strike, expiry=1.0, payout=payout)

        scenarios = [
            (95.0, 5.0),  # Below strike
            (100.0, 0.0),  # At strike (boundary case - not triggered)
            (105.0, 0.0),  # Above strike
        ]

        for spot, expected in scenarios:
            path = backend.convert([[spot]])
            payoff = backend.item(digital_put(path, backend=backend))
            assert abs(payoff - expected) < 1e-10


class TestAsianOptions:
    """Test Asian (path-dependent) options."""

    def test_geometric_asian_call(self):
        """Test geometric Asian call with known path."""
        backend = get_backend("numpy")
        strike = 100.0

        # Path: [100, 105, 110]
        # Geometric mean = (100 * 105 * 110)^(1/3) ≈ 104.915
        path = backend.convert([[100.0, 105.0, 110.0]])

        asian_call = GeometricAsianCall(strike=strike, expiry=1.0)
        payoff = backend.item(asian_call(path, backend=backend))

        geom_mean = (100.0 * 105.0 * 110.0) ** (1 / 3)
        expected = max(geom_mean - strike, 0)
        assert abs(payoff - expected) < 1e-2

    def test_geometric_asian_put(self):
        """Test geometric Asian put with known path."""
        backend = get_backend("numpy")
        strike = 110.0

        # Path where geometric mean < strike
        path = backend.convert([[100.0, 105.0, 108.0]])

        asian_put = GeometricAsianPut(strike=strike, expiry=1.0)
        payoff = backend.item(asian_put(path, backend=backend))

        geom_mean = (100.0 * 105.0 * 108.0) ** (1 / 3)
        expected = max(strike - geom_mean, 0)
        assert abs(payoff - expected) < 1e-2
        assert payoff > 0  # Should be ITM

    def test_single_observation_asian(self):
        """Test Asian option with single observation point."""
        backend = get_backend("numpy")
        strike = 100.0

        # Single observation should equal European payoff
        path = backend.convert([[105.0]])

        asian_call = GeometricAsianCall(strike=strike, expiry=1.0)
        european_call = EuropeanCall(strike=strike, expiry=1.0)

        asian_payoff = backend.item(asian_call(path, backend=backend))
        european_payoff = backend.item(european_call(path, backend=backend))

        assert abs(asian_payoff - european_payoff) < 1e-10


class TestBarrierOptions:
    """Test barrier options."""

    def test_up_and_out_call_not_breached(self):
        """Test up-and-out call when barrier not breached."""
        backend = get_backend("numpy")
        strike = 100.0
        barrier = 120.0

        # Path stays below barrier
        path = backend.convert([[100.0, 105.0, 115.0]])

        barrier_call = UpAndOutCall(strike=strike, expiry=1.0, barrier=barrier)
        payoff = backend.item(barrier_call(path, backend=backend))

        # Should equal European call payoff (final value - strike)
        expected = max(115.0 - strike, 0)
        assert abs(payoff - expected) < 1e-10

    def test_up_and_out_call_breached(self):
        """Test up-and-out call when barrier is breached."""
        backend = get_backend("numpy")
        strike = 100.0
        barrier = 115.0

        # Path breaches barrier
        path = backend.convert([[100.0, 120.0, 110.0]])  # Breaches at second step

        barrier_call = UpAndOutCall(strike=strike, expiry=1.0, barrier=barrier)
        payoff = backend.item(barrier_call(path, backend=backend))

        # Should be zero (knocked out)
        assert abs(payoff - 0.0) < 1e-10

    def test_down_and_in_put_not_breached(self):
        """Test down-and-in put when barrier not breached."""
        backend = get_backend("numpy")
        strike = 100.0
        barrier = 90.0

        # Path stays above barrier
        path = backend.convert([[100.0, 95.0, 92.0]])

        barrier_put = DownAndInPut(strike=strike, expiry=1.0, barrier=barrier)
        payoff = backend.item(barrier_put(path, backend=backend))

        # Should be zero (not knocked in)
        assert abs(payoff - 0.0) < 1e-10

    def test_down_and_in_put_breached(self):
        """Test down-and-in put when barrier is breached."""
        backend = get_backend("numpy")
        strike = 100.0
        barrier = 95.0

        # Path breaches barrier
        path = backend.convert([[100.0, 90.0, 85.0]])  # Breaches at second step

        barrier_put = DownAndInPut(strike=strike, expiry=1.0, barrier=barrier)
        payoff = backend.item(barrier_put(path, backend=backend))

        # Should equal European put payoff (strike - final value)
        expected = max(strike - 85.0, 0)
        assert abs(payoff - expected) < 1e-10


class TestOptionProperties:
    """Test option contract properties and metadata."""

    def test_option_attributes(self):
        """Test all options have required attributes."""
        options = [
            EuropeanCall(strike=100.0, expiry=1.0),
            EuropeanPut(strike=110.0, expiry=0.5),
            DigitalCall(strike=90.0, expiry=2.0, payout=5.0),
            GeometricAsianCall(strike=95.0, expiry=1.5),
            UpAndOutCall(strike=100.0, expiry=1.0, barrier=120.0),
        ]

        for option in options:
            assert hasattr(option, "strike")
            assert hasattr(option, "expiry")
            assert hasattr(option, "side")
            assert hasattr(option, "payoff_type")
            assert hasattr(option, "is_analytic")

            assert isinstance(option.strike, (int, float))
            assert isinstance(option.expiry, (int, float))
            assert option.strike > 0
            assert option.expiry > 0

    def test_option_side_classification(self):
        """Test options are correctly classified by side."""
        call_options = [
            EuropeanCall(strike=100.0, expiry=1.0),
            DigitalCall(strike=100.0, expiry=1.0),
            GeometricAsianCall(strike=100.0, expiry=1.0),
            UpAndOutCall(strike=100.0, expiry=1.0, barrier=120.0),
        ]

        put_options = [
            EuropeanPut(strike=100.0, expiry=1.0),
            DigitalPut(strike=100.0, expiry=1.0),
            GeometricAsianPut(strike=100.0, expiry=1.0),
            DownAndInPut(strike=100.0, expiry=1.0, barrier=80.0),
        ]

        for call in call_options:
            assert call.side == Side.CALL

        for put in put_options:
            assert put.side == Side.PUT

    def test_option_payoff_types(self):
        """Test options have correct payoff type classification."""
        european_options = [
            EuropeanCall(strike=100.0, expiry=1.0),
            EuropeanPut(strike=100.0, expiry=1.0),
        ]

        digital_options = [
            DigitalCall(strike=100.0, expiry=1.0),
            DigitalPut(strike=100.0, expiry=1.0),
        ]

        asian_options = [
            GeometricAsianCall(strike=100.0, expiry=1.0),
            GeometricAsianPut(strike=100.0, expiry=1.0),
        ]

        barrier_options = [
            UpAndOutCall(strike=100.0, expiry=1.0, barrier=120.0),
            DownAndInPut(strike=100.0, expiry=1.0, barrier=80.0),
        ]

        for option in european_options:
            assert option.payoff_type == PayoffType.EUROPEAN

        for option in digital_options:
            assert option.payoff_type == PayoffType.DIGITAL

        for option in asian_options:
            assert option.payoff_type == PayoffType.GEOMETRIC_ASIAN

        for option in barrier_options:
            assert option.payoff_type == PayoffType.BARRIER

    def test_analytic_capability(self):
        """Test which options support analytical pricing."""
        analytic_options = [
            EuropeanCall(strike=100.0, expiry=1.0),
            EuropeanPut(strike=100.0, expiry=1.0),
            DigitalCall(strike=100.0, expiry=1.0),
            DigitalPut(strike=100.0, expiry=1.0),
            GeometricAsianCall(strike=100.0, expiry=1.0),
            GeometricAsianPut(strike=100.0, expiry=1.0),
        ]

        # Barrier options in this implementation are not marked as analytic
        # because they typically use numeric or complex analytic formulas
        # that were separated in the original codebase.
        numeric_options = [
            UpAndOutCall(strike=100.0, expiry=1.0, barrier=120.0),
            DownAndInPut(strike=100.0, expiry=1.0, barrier=80.0),
        ]

        for option in analytic_options:
            assert option.is_analytic

        for option in numeric_options:
            assert not option.is_analytic
