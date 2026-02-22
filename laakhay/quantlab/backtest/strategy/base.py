from __future__ import annotations

import re
from collections.abc import Callable

from ..models import OrderSide, OrderType, RiskType, Signal, SLTPType


class Strategy:
    """Base strategy driven by externally supplied entry/exit evaluators."""

    def __init__(
        self,
        entry_signal: str | None = None,
        exit_signal: str | None = None,
        name: str = "Strategy",
        side: str = "long",
        *,
        entry_rule: Callable[[object, str, str], bool] | None = None,
        exit_rule: Callable[[object, str, str], bool] | None = None,
        required_lookback: int = 2,
        risk_size: RiskType | None = None,
        default_sl: SLTPType | None = None,
        default_tp: SLTPType | None = None,
    ) -> None:
        self.name = name
        self.side = side.lower()
        self.entry_rule = entry_rule
        self.exit_rule = exit_rule
        self.default_risk = risk_size
        self.default_sl = default_sl
        self.default_tp = default_tp
        self._required_lookback = max(
            2,
            int(required_lookback),
            self._infer_lookback_from_expression(entry_signal),
            self._infer_lookback_from_expression(exit_signal),
        )

    def required_lookback(self) -> int:
        """Minimum bars required to evaluate this strategy safely."""
        return self._required_lookback

    def prepare(self, symbol: str, timeframe: str) -> None:
        """Prepare strategy for execution."""
        return

    def on_bar(self, dataset: object, symbol: str, timeframe: str) -> list[Signal]:
        """Evaluate strategy for current bar and emit trading signals."""
        signals = {"entry": False, "exit": False}

        if self.entry_rule is not None:
            try:
                signals["entry"] = bool(self.entry_rule(dataset, symbol, timeframe))
            except Exception:
                signals["entry"] = False

        if self.exit_rule is not None:
            try:
                signals["exit"] = bool(self.exit_rule(dataset, symbol, timeframe))
            except Exception:
                signals["exit"] = False

        is_long = self.side == "long"
        entry_side = OrderSide.BUY if is_long else OrderSide.SELL
        exit_side = OrderSide.SELL if is_long else OrderSide.BUY

        result_signals: list[Signal] = []
        if signals["entry"]:
            result_signals.append(
                Signal(
                    symbol=symbol,
                    side=entry_side,
                    type=OrderType.MARKET,
                    sl=self.default_sl,
                    tp=self.default_tp,
                    size=self.default_risk,
                )
            )

        if signals["exit"]:
            result_signals.append(Signal(symbol=symbol, side=exit_side, type=OrderType.MARKET))

        return result_signals

    @staticmethod
    def _infer_lookback_from_expression(expression: str | None) -> int:
        """Best-effort lookback inference without depending on TA parser."""
        if not expression:
            return 0
        nums = [int(token) for token in re.findall(r"\b\d+\b", expression)]
        return max(nums, default=0)
