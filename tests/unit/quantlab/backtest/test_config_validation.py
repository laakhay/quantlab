from __future__ import annotations

from decimal import Decimal

import pytest

from laakhay.quantlab.backtest import (
    BacktestConfig,
    LeverageControl,
    MarginMode,
    PortfolioControl,
    PortfolioMode,
    RunMode,
)


def test_backtest_config_defaults_remain_compatible():
    config = BacktestConfig()
    assert config.run_mode == RunMode.SINGLE
    assert config.portfolio.mode == PortfolioMode.INDEPENDENT
    assert config.leverage.enabled is False


def test_portfolio_control_validation():
    with pytest.raises(ValueError, match="max_open_positions must be > 0"):
        PortfolioControl(max_open_positions=0)

    with pytest.raises(ValueError, match="max_symbol_weight_pct must be > 0"):
        PortfolioControl(max_symbol_weight_pct=0)


def test_leverage_control_validation():
    with pytest.raises(ValueError, match="leverage must be > 0"):
        LeverageControl(leverage=0)

    with pytest.raises(ValueError, match="maintenance_margin_ratio must be in"):
        LeverageControl(maintenance_margin_ratio=1)

    control = LeverageControl(
        enabled=True,
        leverage=Decimal("3"),
        margin_mode=MarginMode.CROSS,
        maintenance_margin_ratio=Decimal("0.2"),
    )
    assert control.enabled is True


def test_config_daily_loss_limit_validation():
    with pytest.raises(ValueError, match="daily_loss_limit_pct must be in"):
        BacktestConfig(daily_loss_limit_pct=101)
