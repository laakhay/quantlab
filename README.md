# Laakhay Quantlab

`laakhay-quantlab` is a high-performance, backend-agnostic quantitative computation layer designed for simulation-heavy research and production analytics. It provides a unified interface over **NumPy**, **JAX**, and **PyTorch**, enabling seamless switching between CPU and GPU backends without code changes.

## Key Features

- **Backend Agnostic**: Write once, run on NumPy, JAX, or PyTorch.
- **Hardware Acceleration**: Transparent GPU/TPU support via JAX/Torch backends.
- **Vectorized Operations**: Optimized `ArrayBackend` with JIT compilation support.
- **Simulation Primitives**: Fast Gaussian sampling, Geometric Brownian Motion (GBM), and more.
- **Options & Pricing (New)**: Comprehensive verification and pricing of derivatives using analytical (Black-Scholes) and numerical (Monte Carlo) methods.

## Ecosystem

`laakhay-quantlab` fits into the broader Laakhay quantitative ecosystem:
1. **`laakhay-data`**: Market data acquisition and normalization.
2. **`laakhay-ta`**: Technical analysis indicators and strategy engine.
3. **`laakhay-quantlab`**: Numerical simulation, pricing, and risk modeling.

## Installation

```bash
pip install laakhay-quantlab
# extensions: [jax, jax-gpu, torch, all]
pip install "laakhay-quantlab[all]"
```

## Quick Start: Options Pricing

The `pricing` module supports a wide range of exotic and vanilla options, along with Greeks calculation.

```python
from laakhay.quantlab.pricing import (
    EuropeanCall, 
    MarketData, 
    Pricer, 
    PricingMethod
)

# 1. Define Market Conditions
market = MarketData(spot=100.0, rate=0.05, vol=0.2)

# 2. Define Instrument
option = EuropeanCall(strike=100.0, expiry=1.0)

# 3. Price using Black-Scholes (Analytical)
bs_pricer = Pricer(method=PricingMethod.BLACK_SCHOLES)
price, greeks = bs_pricer.price_with_greeks(option, market)

print(f"Price: {price:.4f}")
print(f"Delta: {greeks.delta:.4f}")

# 4. Price using Monte Carlo (Numerical)
mc_pricer = Pricer(method=PricingMethod.MONTE_CARLO)
mc_price = mc_pricer.price(option, market)

print(f"MC Price: {mc_price:.4f}")
```

## Documentation

See the `docs/` directory for detailed guides:
- **Getting Started**: Installation and first steps.
- **Pricing**: Detailed guide on options, strategies, and pricing models.
- **Backends**: Configuring specific computation backends.