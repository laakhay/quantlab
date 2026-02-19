from __future__ import annotations

from laakhay.ta.core.dataset import Dataset
from laakhay.ta.expr.dsl.compiler import ExpressionCompiler
from laakhay.ta.expr.dsl.parser import ExpressionParser
from laakhay.ta.expr.planner.planner import plan_expression
from laakhay.ta.expr.planner.types import PlanResult
from laakhay.ta.expr.runtime.evaluator import RuntimeEvaluator

from ..models import OrderSide, OrderType, RiskType, Signal, SLTPType


class Strategy:
    """Base strategy class using TA expressions."""

    def __init__(
        self,
        entry_signal: str | None = None,
        exit_signal: str | None = None,
        name: str = "Strategy",
        side: str = "long",
        # Advanced Parameters
        risk_size: RiskType | None = None,
        default_sl: SLTPType | None = None,
        default_tp: SLTPType | None = None,
    ) -> None:
        self.name = name
        self.side = side.lower()
        self.entry_plan: PlanResult | None = None
        self.exit_plan: PlanResult | None = None
        self.evaluator = RuntimeEvaluator()
        self.parser = ExpressionParser()
        self.compiler = ExpressionCompiler()

        # Advanced Defaults
        self.default_risk = risk_size
        self.default_sl = default_sl
        self.default_tp = default_tp
        self._required_lookback = 2

        if entry_signal:
            entry_node = self.parser.parse_text(entry_signal)
            compiled_entry = self.compiler.compile(entry_node)
            self.entry_plan = plan_expression(compiled_entry._node)
            self._required_lookback = max(
                self._required_lookback, self._plan_required_lookback(self.entry_plan)
            )

        if exit_signal:
            exit_node = self.parser.parse_text(exit_signal)
            compiled_exit = self.compiler.compile(exit_node)
            self.exit_plan = plan_expression(compiled_exit._node)
            self._required_lookback = max(
                self._required_lookback, self._plan_required_lookback(self.exit_plan)
            )

    def _plan_required_lookback(self, plan: PlanResult | None) -> int:
        if plan is None:
            return 1
        if not plan.requirements.data_requirements:
            return 1
        return max(req.min_lookback for req in plan.requirements.data_requirements)

    def required_lookback(self) -> int:
        """Minimum bars required to evaluate this strategy safely."""
        return max(2, self._required_lookback)

    def prepare(self, symbol: str, timeframe: str) -> None:
        """Prepare strategy for execution."""
        # Could pre-compile here if needed
        pass

    def on_bar(self, dataset: Dataset, symbol: str, timeframe: str) -> list[Signal]:
        """Evaluate strategy for the current bar."""
        signals = {"entry": False, "exit": False}

        # Clear previous cache to ensure fresh evaluation
        self.evaluator.clear_cache()

        # Evaluate Entry
        if self.entry_plan:
            try:
                entry_series = self.evaluator.evaluate(
                    self.entry_plan, dataset, symbol=symbol, timeframe=timeframe
                )
                if entry_series and len(entry_series) > 0:
                    signals["entry"] = bool(entry_series.values[-1])
            except ValueError:
                # Expected during warmup
                pass
            except Exception as e:
                print(f"Strategy Entry Error: {e}")
                pass

        # Evaluate Exit
        if self.exit_plan:
            try:
                exit_series = self.evaluator.evaluate(
                    self.exit_plan, dataset, symbol=symbol, timeframe=timeframe
                )
                if exit_series and len(exit_series) > 0:
                    signals["exit"] = bool(exit_series.values[-1])
            except ValueError:
                pass
            except Exception as e:
                print(f"Strategy Exit Error: {e}")
                pass

        # Convert to Signal Objects
        result_signals = []

        is_long = self.side == "long"
        entry_side = OrderSide.BUY if is_long else OrderSide.SELL
        exit_side = OrderSide.SELL if is_long else OrderSide.BUY

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
