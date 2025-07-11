"""Comprehensive tests for simulations module coverage."""

import pytest

from laakhay.quantlab.backend import get_backend, has_backend
from laakhay.quantlab.simulations import GeometricBrownianMotion
from laakhay.quantlab.simulations.samplers import GaussianSampler, BaseSampler


class TestSimulationsCoverage:
    """Comprehensive tests for simulations module."""

    def test_gaussian_sampler_all_methods(self):
        """Test all GaussianSampler methods."""
        backend = get_backend("numpy")
        sampler = GaussianSampler(mean=2.0, std=1.5)
        
        # Test properties
        assert sampler.mean == 2.0
        assert sampler.std == 1.5
        assert sampler.variance == 2.25
        
        # Test sampling with different options
        samples = sampler.sample((1000,), backend=backend)
        assert backend.shape(samples) == (1000,)
        
        # Test standardized sampling
        std_samples = sampler.sample((1000,), backend=backend, standardize=True)
        assert abs(float(backend.to_numpy(backend.mean(std_samples)))) < 0.1
        assert abs(float(backend.to_numpy(backend.std(std_samples))) - 1.0) < 0.1
        
        # Test PDF
        x = backend.array([1.0, 2.0, 3.0])
        pdf_vals = sampler.pdf(x, backend=backend)
        assert backend.shape(pdf_vals) == (3,)
        
        # Test standardized PDF
        pdf_std = sampler.pdf(x, backend=backend, standardize=True)
        
        # Test CDF
        cdf_vals = sampler.cdf(x, backend=backend)
        assert all(0 <= float(backend.to_numpy(v)) <= 1 for v in cdf_vals)
        
        # Test standardized CDF
        cdf_std = sampler.cdf(x, backend=backend, standardize=True)
        
        # Test PPF
        q = backend.array([0.1, 0.5, 0.9])
        ppf_vals = sampler.ppf(q, backend=backend)
        
        # Test standardized PPF
        ppf_std = sampler.ppf(q, backend=backend, standardize=True)
        
        # Test that moments properties exist (they are computed via sampling)
        skew = sampler.skew  # Just access the property
        ex_kurtosis = sampler.ex_kurtosis  # Just access the property
        
        # Test MGF
        mgf_val = sampler.mgf(0.5, backend=backend)
        assert mgf_val > 0
        
        # Test string representation
        assert str(sampler) == "GaussianSampler(mean=2.0, std=1.5)"

    def test_gbm_all_methods(self):
        """Test all GBM methods comprehensively."""
        backend = get_backend("numpy")
        
        gbm = GeometricBrownianMotion(
            spot=100.0,
            drift=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_steps=252
        )
        
        # Test properties
        assert gbm.spot == 100.0
        assert gbm.drift == 0.05
        assert gbm.volatility == 0.2
        assert gbm.time_to_maturity == 1.0
        assert gbm.num_steps == 252
        assert abs(gbm.dt - 1.0/252) < 1e-10
        
        # Test shock generation
        shocks = gbm.generate_shocks(1000, backend=backend)
        assert backend.shape(shocks) == (1000, 252)
        
        # Test antithetic shocks
        anti_shocks = gbm.generate_shocks(1000, backend=backend, antithetic=True)
        assert backend.shape(anti_shocks) == (1000, 252)
        
        # Test unit paths
        unit_paths = gbm.build_unit_paths(shocks, backend=backend)
        assert backend.shape(unit_paths) == (1000, 253)
        
        # Test log paths
        log_paths = gbm.build_log_paths(unit_paths, backend=backend)
        assert backend.shape(log_paths) == (1000, 253)
        
        # Test scale to spot
        price_paths = gbm.scale_to_spot(log_paths, backend=backend)
        assert backend.shape(price_paths) == (1000, 253)
        
        # Test generate paths
        paths = gbm.generate_paths(100, backend=backend)
        assert backend.shape(paths) == (100, 253)
        
        # Test log paths generation
        log_paths_direct = gbm.generate_paths(100, backend=backend, return_log=True)
        assert backend.shape(log_paths_direct) == (100, 253)
        
        # Test terminal values
        terminal = gbm.terminal_values(10000, backend=backend)
        assert backend.shape(terminal) == (10000,)
        assert float(backend.to_numpy(backend.min(terminal))) > 0
        
        # Test terminal values with antithetic
        terminal_anti = gbm.terminal_values(10000, backend=backend, antithetic=True)
        assert backend.shape(terminal_anti) == (10000,)
        
        # Test add drift
        const_paths = backend.full((100, 253), 100.0)
        drifted = gbm.add_drift(const_paths, backend=backend)
        
        # Test with custom sampler
        custom_sampler = GaussianSampler(mean=0.0, std=1.0)
        gbm_custom = GeometricBrownianMotion(
            spot=100.0,
            drift=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_steps=100,
            sampler=custom_sampler
        )
        assert gbm_custom.sampler == custom_sampler

    def test_base_sampler_abstract_methods(self):
        """Test BaseSampler abstract methods."""
        backend = get_backend("numpy")
        
        class SimpleSampler(BaseSampler):
            """Simple sampler for testing."""
            def __init__(self):
                super().__init__(mean=0.0, std=1.0)
            
            def _raw_sample(self, shape, backend, key=None):
                return backend.random_normal(key, shape)
            
            def pdf(self, x, backend=None, **kwargs):
                backend = backend or get_backend()
                return backend.exp(-0.5 * backend.pow(x, 2)) / backend.sqrt(2 * 3.14159)
            
            def cdf(self, x, backend=None, **kwargs):
                backend = backend or get_backend()
                return 0.5 * (1 + backend.erf(x / backend.sqrt(2)))
            
            def ppf(self, q, backend=None, **kwargs):
                backend = backend or get_backend()
                # Simple PPF implementation
                return q  # Placeholder
                
            def sample(self, shape, backend=None, key=None, **kwargs):
                backend = backend or get_backend()
                return self._raw_sample(shape, backend, key)
                
            def __repr__(self):
                return "SimpleSampler()"
        
        sampler = SimpleSampler()
        
        # Test sampling
        samples = sampler.sample((100,), backend=backend)
        assert backend.shape(samples) == (100,)
        
        # Test PDF
        x = backend.array([0.0, 1.0])
        pdf = sampler.pdf(x, backend=backend)
        
        # Test CDF
        cdf = sampler.cdf(x, backend=backend)
        
        # Test PPF
        q = backend.array([0.5, 0.9])
        ppf = sampler.ppf(q, backend=backend)
        
        # Test MGF calculation
        mgf = sampler.mgf(0.5, backend=backend)
        assert mgf > 0

    def test_error_conditions(self):
        """Test error conditions for coverage."""
        backend = get_backend("numpy")
        
        # Test invalid sampler initialization
        with pytest.raises(ValueError):
            GaussianSampler(mean=0.0, std=-1.0)
        
        # Test invalid GBM initialization
        with pytest.raises(ValueError):
            GeometricBrownianMotion(
                spot=-100.0,  # negative spot
                drift=0.05,
                volatility=0.2,
                time_to_maturity=1.0,
                num_steps=100
            )
        
        # Test invalid quantiles for PPF
        sampler = GaussianSampler(mean=0.0, std=1.0)
        with pytest.raises(ValueError):
            sampler.ppf(backend.array([1.5]), backend=backend)
        
        # Test antithetic with odd number of paths
        gbm = GeometricBrownianMotion(
            spot=100.0,
            drift=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_steps=100
        )
        with pytest.raises(ValueError):
            gbm.generate_shocks(101, backend=backend, antithetic=True)

    def test_jax_backend_sampling(self):
        """Test sampling with JAX backend."""
        if not has_backend("jax"):
            pytest.skip("JAX not available")
        
        backend = get_backend("jax")
        sampler = GaussianSampler(mean=0.0, std=1.0)
        
        # JAX requires key
        key = backend.random_key(42)
        samples = sampler.sample((100,), backend=backend, key=key)
        assert backend.shape(samples) == (100,)
        
        # Test GBM with JAX
        gbm = GeometricBrownianMotion(
            spot=100.0,
            drift=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_steps=100
        )
        
        paths = gbm.generate_paths(50, backend=backend, key=key)
        assert backend.shape(paths) == (50, 101)

    def test_torch_backend_sampling(self):
        """Test sampling with PyTorch backend."""
        if not has_backend("torch"):
            pytest.skip("PyTorch not available")
        
        backend = get_backend("torch")
        sampler = GaussianSampler(mean=0.0, std=1.0)
        
        # PyTorch doesn't require key but accepts it
        samples = sampler.sample((100,), backend=backend)
        assert backend.shape(samples) == (100,)
        
        # Test GBM with PyTorch
        gbm = GeometricBrownianMotion(
            spot=100.0,
            drift=0.05,
            volatility=0.2,
            time_to_maturity=1.0,
            num_steps=100
        )
        
        paths = gbm.generate_paths(50, backend=backend)
        assert backend.shape(paths) == (50, 101)