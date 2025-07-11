"""Tests for Gaussian sampler."""

import pytest

from laakhay.quantlab.simulations.samplers import GaussianSampler


class TestGaussianSampler:
    """Test Gaussian sampler functionality."""

    def test_initialization(self):
        """Test sampler initialization."""
        sampler = GaussianSampler(mean=0.0, std=1.0)
        assert sampler.mean == 0.0
        assert sampler.std == 1.0
        assert sampler.variance == 1.0

    def test_invalid_std(self):
        """Test that negative or zero std raises error."""
        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            GaussianSampler(mean=0.0, std=0.0)
        
        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            GaussianSampler(mean=0.0, std=-1.0)

    def test_sampling_shape(self, backend, gaussian_params):
        """Test that sampling returns correct shape."""
        mean, std = gaussian_params
        sampler = GaussianSampler(mean=mean, std=std)
        
        shapes = [(100,), (50, 20), (10, 10, 10)]
        for shape in shapes:
            if backend.name == "jax":
                if not hasattr(backend, "random_normal"):
                    pytest.skip(f"{backend.name} backend doesn't support random generation")
                import jax
                key = jax.random.PRNGKey(42)
                samples = sampler.sample(shape, backend=backend, key=key)
            else:
                samples = sampler.sample(shape, backend=backend)
            
            assert backend.shape(samples) == shape

    def test_sampling_statistics(self, backend, sample_size):
        """Test that sample statistics match distribution parameters."""
        sampler = GaussianSampler(mean=5.0, std=2.0)
        
        if backend.name == "jax":
            key = backend.random_key(42)
            samples = sampler.sample((sample_size,), backend=backend, key=key)
        else:
            samples = sampler.sample((sample_size,), backend=backend)
        
        sample_mean = float(backend.to_numpy(backend.mean(samples)))
        sample_std = float(backend.to_numpy(backend.std(samples)))
        
        # Check statistics with appropriate tolerance
        tolerance = 10.0 / (sample_size ** 0.5)  # Scales with sample size
        assert abs(sample_mean - 5.0) < tolerance
        assert abs(sample_std - 2.0) < tolerance

    def test_standardized_sampling(self, backend):
        """Test standardized sampling returns N(0,1)."""
        sampler = GaussianSampler(mean=5.0, std=2.0)
        
        if backend.name == "jax":
            key = backend.random_key(42)
            samples = sampler.sample((10000,), backend=backend, key=key, standardize=True)
        else:
            samples = sampler.sample((10000,), backend=backend, standardize=True)
        
        sample_mean = float(backend.to_numpy(backend.mean(samples)))
        sample_std = float(backend.to_numpy(backend.std(samples)))
        
        assert abs(sample_mean) < 0.05
        assert abs(sample_std - 1.0) < 0.05

    def test_pdf_values(self, backend, gaussian_params):
        """Test PDF calculation at specific points."""
        mean, std = gaussian_params
        sampler = GaussianSampler(mean=mean, std=std)
        
        # Test at mean (should be maximum)
        x_mean = backend.array(mean)
        pdf_at_mean = sampler.pdf(x_mean, backend=backend)
        expected_max = 1.0 / (std * (2 * 3.141592653589793) ** 0.5)
        
        assert abs(float(backend.to_numpy(pdf_at_mean)) - expected_max) < 1e-6
        
        # Test symmetry
        x_left = backend.array(mean - std)
        x_right = backend.array(mean + std)
        pdf_left = sampler.pdf(x_left, backend=backend)
        pdf_right = sampler.pdf(x_right, backend=backend)
        
        assert abs(float(backend.to_numpy(pdf_left - pdf_right))) < 1e-10

    def test_pdf_standardized(self, backend):
        """Test standardized PDF calculation."""
        sampler = GaussianSampler(mean=5.0, std=2.0)
        
        x = backend.array([0.0, 1.0, -1.0])
        pdf_vals = sampler.pdf(x, backend=backend, standardize=True)
        
        # Standard normal PDF at 0 should be 1/sqrt(2π)
        expected_at_0 = 1.0 / (2 * 3.141592653589793) ** 0.5
        assert abs(float(backend.to_numpy(pdf_vals[0])) - expected_at_0) < 1e-6

    def test_cdf_values(self, backend, gaussian_params):
        """Test CDF calculation."""
        mean, std = gaussian_params
        sampler = GaussianSampler(mean=mean, std=std)
        
        # CDF at mean should be 0.5
        x_mean = backend.array(mean)
        cdf_at_mean = sampler.cdf(x_mean, backend=backend)
        assert abs(float(backend.to_numpy(cdf_at_mean)) - 0.5) < 1e-6
        
        # Test monotonicity
        x_vals = backend.linspace(mean - 3*std, mean + 3*std, 10)
        cdf_vals = sampler.cdf(x_vals, backend=backend)
        cdf_numpy = backend.to_numpy(cdf_vals)
        
        for i in range(len(cdf_numpy) - 1):
            assert cdf_numpy[i] <= cdf_numpy[i + 1]

    def test_cdf_bounds(self, backend):
        """Test CDF bounds."""
        sampler = GaussianSampler(mean=0.0, std=1.0)
        
        # Far left tail
        x_left = backend.array(-10.0)
        cdf_left = sampler.cdf(x_left, backend=backend)
        assert float(backend.to_numpy(cdf_left)) < 1e-6
        
        # Far right tail
        x_right = backend.array(10.0)
        cdf_right = sampler.cdf(x_right, backend=backend)
        assert float(backend.to_numpy(cdf_right)) > 1 - 1e-6

    def test_ppf_values(self, backend, gaussian_params):
        """Test PPF (quantile) calculation."""
        mean, std = gaussian_params
        sampler = GaussianSampler(mean=mean, std=std)
        
        # PPF at 0.5 should be mean
        q_median = backend.array(0.5)
        ppf_median = sampler.ppf(q_median, backend=backend)
        assert abs(float(backend.to_numpy(ppf_median)) - mean) < 1e-6
        
        # Test symmetry
        q_low = backend.array(0.25)
        q_high = backend.array(0.75)
        ppf_low = sampler.ppf(q_low, backend=backend)
        ppf_high = sampler.ppf(q_high, backend=backend)
        
        dist_low = abs(float(backend.to_numpy(ppf_low)) - mean)
        dist_high = abs(float(backend.to_numpy(ppf_high)) - mean)
        assert abs(dist_low - dist_high) < 1e-6

    def test_ppf_bounds(self, backend):
        """Test PPF with invalid quantiles."""
        sampler = GaussianSampler(mean=0.0, std=1.0)
        
        with pytest.raises(ValueError, match="Quantiles must be in"):
            sampler.ppf(backend.array(-0.1), backend=backend)
        
        with pytest.raises(ValueError, match="Quantiles must be in"):
            sampler.ppf(backend.array(1.1), backend=backend)

    def test_cdf_ppf_inverse(self, backend):
        """Test that CDF and PPF are inverses."""
        sampler = GaussianSampler(mean=2.0, std=1.5)
        
        # Test CDF(PPF(q)) = q
        q_vals = backend.array([0.1, 0.25, 0.5, 0.75, 0.9])
        x_vals = sampler.ppf(q_vals, backend=backend)
        q_recovered = sampler.cdf(x_vals, backend=backend)
        
        # Use looser tolerance for float32
        tolerance = 1e-5 if str(q_vals.dtype).endswith('32') else 1e-6
        assert backend.to_numpy(backend.max(backend.abs(q_vals - q_recovered))) < tolerance
        
        # Test PPF(CDF(x)) = x
        x_test = backend.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        q_test = sampler.cdf(x_test, backend=backend)
        x_recovered = sampler.ppf(q_test, backend=backend)
        
        assert backend.to_numpy(backend.max(backend.abs(x_test - x_recovered))) < tolerance

    def test_moments(self, backend):
        """Test skewness and excess kurtosis."""
        sampler = GaussianSampler(mean=0.0, std=1.0)
        
        # Gaussian should have skew = 0, excess kurtosis = 0
        # Note: These are computed via sampling, so we allow some tolerance
        assert abs(sampler.skew) < 0.1
        assert abs(sampler.ex_kurtosis) < 0.1

    def test_mgf(self, backend):
        """Test moment generating function."""
        sampler = GaussianSampler(mean=1.0, std=0.5)
        
        # For Gaussian, MGF(t) = exp(μt + σ²t²/2)
        t_vals = [0.1, 0.5, 1.0]
        for t in t_vals:
            mgf_computed = sampler.mgf(t, backend=backend)
            mgf_expected = backend.to_numpy(
                backend.exp(backend.array(1.0 * t + 0.5**2 * t**2 / 2))
            )
            
            # MGF is computed via sampling, so allow some tolerance
            relative_error = abs(mgf_computed - mgf_expected) / mgf_expected
            assert relative_error < 0.05

    def test_repr(self):
        """Test string representation."""
        sampler = GaussianSampler(mean=2.5, std=1.5)
        assert repr(sampler) == "GaussianSampler(mean=2.5, std=1.5)"