from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from ..domain import BacktestConfig


@dataclass(frozen=True)
class PortfolioExposure:
    gross_notional: Decimal
    net_notional: Decimal
    by_symbol_notional: dict[str, Decimal]


class PortfolioRiskChecker:
    """Portfolio-level pre-trade risk checks."""

    def projected_exposure_after_open(
        self,
        *,
        current: PortfolioExposure,
        symbol: str,
        side_sign: int,
        notional: Decimal,
    ) -> PortfolioExposure:
        by_symbol = dict(current.by_symbol_notional)
        by_symbol[symbol] = by_symbol.get(symbol, Decimal("0")) + notional
        return PortfolioExposure(
            gross_notional=current.gross_notional + notional,
            net_notional=current.net_notional + (Decimal(str(side_sign)) * notional),
            by_symbol_notional=by_symbol,
        )

    def check_open(
        self,
        *,
        config: BacktestConfig,
        equity: Decimal,
        current: PortfolioExposure,
        symbol: str,
        side_sign: int,
        notional: Decimal,
    ) -> str | None:
        """Return rejection reason if trade should be blocked, else None."""
        if notional <= 0:
            return "allocation_rejected"
        if equity <= 0:
            return "non_positive_equity"

        projected = self.projected_exposure_after_open(
            current=current,
            symbol=symbol,
            side_sign=side_sign,
            notional=notional,
        )

        max_symbol_weight_pct = config.portfolio.max_symbol_weight_pct
        if max_symbol_weight_pct is not None:
            symbol_cap = equity * (Decimal(str(max_symbol_weight_pct)) / Decimal("100"))
            if projected.by_symbol_notional.get(symbol, Decimal("0")) > symbol_cap:
                return "max_symbol_weight"

        max_gross_exposure_pct = config.portfolio.max_gross_exposure_pct
        if max_gross_exposure_pct is not None:
            gross_cap = equity * (Decimal(str(max_gross_exposure_pct)) / Decimal("100"))
            if projected.gross_notional > gross_cap:
                return "max_gross_exposure"

        max_net_exposure_pct = config.portfolio.max_net_exposure_pct
        if max_net_exposure_pct is not None:
            net_cap = equity * (Decimal(str(max_net_exposure_pct)) / Decimal("100"))
            if abs(projected.net_notional) > net_cap:
                return "max_net_exposure"

        mmr = config.leverage.maintenance_margin_ratio
        if config.leverage.enabled and mmr is not None and projected.gross_notional > 0:
            ratio = equity / projected.gross_notional
            if ratio < Decimal(str(mmr)):
                return "maintenance_margin_breach"

        return None
