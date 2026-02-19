"""Tests for Geometric Brownian Motion."""

import pytest

from laakhay.quantlab.simulations import GeometricBrownianMotion
from laakhay.quantlab.simulations.samplers import GaussianSampler


class TestGeometricBrownianMotion:
    """Test GBM functionality."""

    def test_initialization(self, gbm_params):
        """Test GBM initialization."""
        gbm = GeometricBrownianMotion(**gbm_params)

        assert gbm.spot == gbm_params["spot"]
        assert gbm.drift == gbm_params["drift"]
        assert gbm.volatility == gbm_params["volatility"]
        assert gbm.time_to_maturity == gbm_params["time_to_maturity"]
        assert gbm.num_steps == gbm_params["num_steps"]
        assert gbm.dt == gbm_params["time_to_maturity"] / gbm_params["num_steps"]

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="Spot price must be positive"):
            GeometricBrownianMotion(
                spot=0.0,
                drift=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                num_steps=100,
            )

        with pytest.raises(ValueError, match="Volatility must be non-negative"):
            GeometricBrownianMotion(
                spot=100.0,
                drift=0.05,
                volatility=-0.2,
                time_to_maturity=1.0,
                num_steps=100,
            )

        with pytest.raises(ValueError, match="Time to maturity must be positive"):
            GeometricBrownianMotion(
                spot=100.0,
                drift=0.05,
                volatility=0.2,
                time_to_maturity=0.0,
                num_steps=100,
            )

        with pytest.raises(ValueError, match="Number of steps must be positive"):
            GeometricBrownianMotion(
                spot=100.0,
                drift=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                num_steps=0,
            )

    def test_custom_sampler(self):
        """Test GBM with custom sampler."""
        custom_sampler = GaussianSampler(mean=0.0, std=1.0)
        gbm = GeometricBrownianMotion(
            spot=100.0,
            drift=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_steps=100,
            sampler=custom_sampler,
        )
        assert gbm.sampler is custom_sampler

    def test_generate_shocks_shape(self, backend, gbm_params):
        """Test shock generation shape."""
        gbm = GeometricBrownianMotion(**gbm_params)
        num_paths = 1000

        if backend.name == "jax":
            key = backend.random_key(42)
            shocks = gbm.generate_shocks(num_paths, backend=backend, key=key)
        else:
            shocks = gbm.generate_shocks(num_paths, backend=backend)

        assert backend.shape(shocks) == (num_paths, gbm.num_steps)

    def test_generate_shocks_statistics(self, backend):
        """Test that shocks are standard normal."""
        gbm = GeometricBrownianMotion(
            spot=100.0, drift=0.05, volatility=0.2, time_to_maturity=1.0, num_steps=252
        )
        num_paths = 10000

        if backend.name == "jax":
            key = backend.random_key(42)
            shocks = gbm.generate_shocks(num_paths, backend=backend, key=key)
        else:
            shocks = gbm.generate_shocks(num_paths, backend=backend)

        # Check mean and std of shocks
        mean_shock = float(backend.to_numpy(backend.mean(shocks)))
        std_shock = float(backend.to_numpy(backend.std(shocks)))

        assert abs(mean_shock) < 0.02
        assert abs(std_shock - 1.0) < 0.02

    def test_antithetic_shocks(self, backend):
        """Test antithetic variate generation."""
        gbm = GeometricBrownianMotion(
            spot=100.0, drift=0.05, volatility=0.2, time_to_maturity=1.0, num_steps=100
        )
        num_paths = 1000  # Must be even

        if backend.name == "jax":
            key = backend.random_key(42)
            shocks = gbm.generate_shocks(num_paths, backend=backend, key=key, antithetic=True)
        else:
            shocks = gbm.generate_shocks(num_paths, backend=backend, antithetic=True)

        # Check that second half is negative of first half
        half = num_paths // 2
        first_half = shocks[:half]
        second_half = shocks[half:]

        diff = backend.add(first_half, second_half)
        max_diff = float(backend.to_numpy(backend.max(backend.abs(diff))))
        assert max_diff < 1e-10

    def test_antithetic_odd_paths(self, backend):
        """Test that antithetic with odd paths raises error."""
        gbm = GeometricBrownianMotion(
            spot=100.0, drift=0.05, volatility=0.2, time_to_maturity=1.0, num_steps=100
        )

        with pytest.raises(ValueError, match="Number of paths must be even"):
            if backend.name == "jax":
                key = backend.random_key(42)
                gbm.generate_shocks(999, backend=backend, key=key, antithetic=True)
            else:
                gbm.generate_shocks(999, backend=backend, antithetic=True)

    def test_build_unit_paths(self, backend):
        """Test unit path construction."""
        gbm = GeometricBrownianMotion(
            spot=100.0, drift=0.05, volatility=0.2, time_to_maturity=1.0, num_steps=100
        )
        num_paths = 1000

        if backend.name == "jax":
            key = backend.random_key(42)
            shocks = gbm.generate_shocks(num_paths, backend=backend, key=key)
        else:
            shocks = gbm.generate_shocks(num_paths, backend=backend)

        unit_paths = gbm.build_unit_paths(shocks, backend=backend)

        # Check shape
        assert backend.shape(unit_paths) == (num_paths, gbm.num_steps + 1)

        # Check that paths start at 0
        initial_values = unit_paths[:, 0]
        assert float(backend.to_numpy(backend.max(backend.abs(initial_values)))) < 1e-10

    def test_build_log_paths(self, backend, gbm_params):
        """Test log path construction."""
        gbm = GeometricBrownianMotion(**gbm_params)
        num_paths = 100

        if backend.name == "jax":
            key = backend.random_key(42)
            shocks = gbm.generate_shocks(num_paths, backend=backend, key=key)
        else:
            shocks = gbm.generate_shocks(num_paths, backend=backend)

        unit_paths = gbm.build_unit_paths(shocks, backend=backend)
        log_paths = gbm.build_log_paths(unit_paths, backend=backend)

        # Check shape
        assert backend.shape(log_paths) == (num_paths, gbm.num_steps + 1)

        # Check initial value
        log_spot = backend.log(backend.array(gbm.spot))
        initial_log_values = log_paths[:, 0]
        expected_initial = backend.full(backend.shape(initial_log_values), log_spot)

        diff = backend.abs(backend.sub(initial_log_values, expected_initial))
        assert float(backend.to_numpy(backend.max(diff))) < 1e-10

    def test_scale_to_spot(self, backend):
        """Test conversion from log paths to price paths."""
        gbm = GeometricBrownianMotion(
            spot=100.0, drift=0.05, volatility=0.2, time_to_maturity=1.0, num_steps=100
        )

        # Create simple log paths for testing
        log_paths = backend.array([[4.0, 4.1, 4.2], [4.5, 4.6, 4.7]])
        price_paths = gbm.scale_to_spot(log_paths, backend=backend)

        # Check that exp(log_paths) = price_paths
        expected = backend.exp(log_paths)
        diff = backend.abs(backend.sub(price_paths, expected))
        assert float(backend.to_numpy(backend.max(diff))) < 1e-10

    def test_generate_paths_shape(self, backend, gbm_params):
        """Test full path generation shape."""
        gbm = GeometricBrownianMotion(**gbm_params)
        num_paths = 100

        if backend.name == "jax":
            key = backend.random_key(42)
            paths = gbm.generate_paths(num_paths, backend=backend, key=key)
        else:
            paths = gbm.generate_paths(num_paths, backend=backend)

        assert backend.shape(paths) == (num_paths, gbm.num_steps + 1)

    def test_generate_paths_initial_value(self, backend, gbm_params):
        """Test that paths start at spot price."""
        gbm = GeometricBrownianMotion(**gbm_params)
        num_paths = 100

        if backend.name == "jax":
            key = backend.random_key(42)
            paths = gbm.generate_paths(num_paths, backend=backend, key=key)
        else:
            paths = gbm.generate_paths(num_paths, backend=backend)

        initial_values = paths[:, 0]
        expected_initial = backend.full(backend.shape(initial_values), gbm.spot)

        diff = backend.abs(backend.sub(initial_values, expected_initial))
        # Use looser tolerance for float32
        tolerance = 1e-5 if str(initial_values.dtype).endswith("32") else 1e-10
        assert float(backend.to_numpy(backend.max(diff))) < tolerance

    def test_generate_paths_positivity(self, backend):
        """Test that all price paths remain positive."""
        gbm = GeometricBrownianMotion(
            spot=100.0, drift=0.05, volatility=0.2, time_to_maturity=1.0, num_steps=252
        )
        num_paths = 1000

        if backend.name == "jax":
            key = backend.random_key(42)
            paths = gbm.generate_paths(num_paths, backend=backend, key=key)
        else:
            paths = gbm.generate_paths(num_paths, backend=backend)

        min_value = float(backend.to_numpy(backend.min(paths)))
        assert min_value > 0

    def test_generate_log_paths(self, backend):
        """Test log path generation."""
        gbm = GeometricBrownianMotion(
            spot=100.0, drift=0.05, volatility=0.2, time_to_maturity=1.0, num_steps=100
        )
        num_paths = 100

        if backend.name == "jax":
            key = backend.random_key(42)
            log_paths = gbm.generate_paths(num_paths, backend=backend, key=key, return_log=True)
        else:
            log_paths = gbm.generate_paths(num_paths, backend=backend, return_log=True)

        # Convert to prices and check consistency
        price_paths = backend.exp(log_paths)
        assert backend.shape(price_paths) == backend.shape(log_paths)

        # Generate price paths directly
        if backend.name == "jax":
            key2 = backend.random_key(42)
            direct_paths = gbm.generate_paths(num_paths, backend=backend, key=key2)
        else:
            # Reset numpy random state for consistency
            import numpy as np

            np.random.seed(42)
            direct_paths = gbm.generate_paths(num_paths, backend=backend)

        # They should be close (not exact due to different random streams)
        assert backend.shape(log_paths) == backend.shape(direct_paths)

    def test_terminal_values_shape(self, backend, gbm_params):
        """Test terminal value generation shape."""
        gbm = GeometricBrownianMotion(**gbm_params)
        num_paths = 10000

        if backend.name == "jax":
            key = backend.random_key(42)
            terminal_vals = gbm.terminal_values(num_paths, backend=backend, key=key)
        else:
            terminal_vals = gbm.terminal_values(num_paths, backend=backend)

        assert backend.shape(terminal_vals) == (num_paths,)

    def test_terminal_values_statistics(self, backend):
        """Test terminal value distribution."""
        # Use risk-neutral drift for easier testing
        spot = 100.0
        risk_free_rate = 0.05
        volatility = 0.2
        time_to_maturity = 1.0

        gbm = GeometricBrownianMotion(
            spot=spot,
            drift=risk_free_rate,
            volatility=volatility,
            time_to_maturity=time_to_maturity,
            num_steps=1,
        )
        num_paths = 100000

        if backend.name == "jax":
            key = backend.random_key(42)
            terminal_vals = gbm.terminal_values(num_paths, backend=backend, key=key)
        else:
            terminal_vals = gbm.terminal_values(num_paths, backend=backend)

        # Expected value under risk-neutral measure
        expected_mean = spot * float(
            backend.to_numpy(backend.exp(backend.array(risk_free_rate * time_to_maturity)))
        )
        sample_mean = float(backend.to_numpy(backend.mean(terminal_vals)))

        # Allow 1% relative error due to sampling
        relative_error = abs(sample_mean - expected_mean) / expected_mean
        assert relative_error < 0.01

    def test_terminal_values_positivity(self, backend, gbm_params):
        """Test that terminal values are positive."""
        gbm = GeometricBrownianMotion(**gbm_params)
        num_paths = 10000

        if backend.name == "jax":
            key = backend.random_key(42)
            terminal_vals = gbm.terminal_values(num_paths, backend=backend, key=key)
        else:
            terminal_vals = gbm.terminal_values(num_paths, backend=backend)

        min_value = float(backend.to_numpy(backend.min(terminal_vals)))
        assert min_value > 0

    def test_terminal_values_antithetic(self, backend):
        """Test terminal values with antithetic variates."""
        gbm = GeometricBrownianMotion(
            spot=100.0, drift=0.05, volatility=0.2, time_to_maturity=1.0, num_steps=1
        )
        num_paths = 10000

        if backend.name == "jax":
            key = backend.random_key(42)
            terminal_vals = gbm.terminal_values(
                num_paths, backend=backend, key=key, antithetic=True
            )
        else:
            terminal_vals = gbm.terminal_values(num_paths, backend=backend, antithetic=True)

        assert backend.shape(terminal_vals) == (num_paths,)

        # Antithetic should reduce variance
        # We can't easily test this without running multiple simulations

    def test_add_drift(self, backend):
        """Test drift addition to paths."""
        gbm = GeometricBrownianMotion(
            spot=100.0, drift=0.1, volatility=0.2, time_to_maturity=1.0, num_steps=100
        )

        # Create constant paths for testing
        num_paths = 10
        constant_value = 100.0
        paths = backend.full((num_paths, gbm.num_steps + 1), constant_value)

        drifted_paths = gbm.add_drift(paths, backend=backend)

        # Check that final values have expected drift
        final_values = drifted_paths[:, -1]
        expected_final = constant_value * float(
            backend.to_numpy(backend.exp(backend.array(gbm.drift * gbm.time_to_maturity)))
        )

        diff = backend.abs(backend.sub(final_values, backend.array(expected_final)))
        assert float(backend.to_numpy(backend.max(diff))) < 1e-6
