from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from ..domain import (
    BacktestConfig,
    ExecutionRejection,
    OrderSide,
    PortfolioAccount,
    Position,
    RunMode,
)
from ..feed import MultiAssetMemFeed
from ..metrics import compute_performance_metrics, infer_periods_per_year
from .allocation import EqualNotionalAllocator
from .risk import PortfolioExposure, PortfolioRiskChecker
from .scheduler import MultiAssetScheduler


@dataclass
class PortfolioPositionState:
    position: Position
    margin_used: Decimal


class PortfolioBacktestEngine:
    """Shared-capital multi-asset backtest engine scaffold."""

    _BPS_SCALE = Decimal("10000")

    def __init__(
        self,
        initial_capital: float | Decimal = 10000.0,
        *,
        commission_bps: float | Decimal = 0,
        config: BacktestConfig | None = None,
    ) -> None:
        self.initial_capital = Decimal(str(initial_capital))
        self.commission_bps = Decimal(str(commission_bps))
        self.config = config or BacktestConfig(run_mode=RunMode.PORTFOLIO_SHARED)
        if self.commission_bps < 0:
            raise ValueError("commission_bps must be >= 0")

        self.account = PortfolioAccount.from_initial_capital(self.initial_capital)
        self.positions: dict[str, PortfolioPositionState] = {}
        self.mark_prices: dict[str, Decimal] = {}
        self.equity_curve: list[tuple[Any, Decimal]] = []
        self.trades: list[dict[str, Any]] = []
        self.rejections: list[ExecutionRejection] = []
        self._allocator = EqualNotionalAllocator()
        self._risk = PortfolioRiskChecker()

    def run(
        self,
        strategy_factory: Any,
        feed: MultiAssetMemFeed,
        *,
        start_dt: Any | None = None,
        end_dt: Any | None = None,
    ) -> dict[str, Any]:
        self.account = PortfolioAccount.from_initial_capital(self.initial_capital)
        self.positions.clear()
        self.mark_prices.clear()
        self.equity_curve.clear()
        self.trades.clear()
        self.rejections.clear()

        strategies: dict[str, Any] = {}
        lookbacks: dict[str, int] = {}
        for symbol in feed.symbols:
            strategy = strategy_factory(symbol)
            strategy.prepare(symbol, feed.timeframe)
            strategies[symbol] = strategy
            required = getattr(strategy, "required_lookback", None)
            lookbacks[symbol] = max(1, int(required() if callable(required) else 1))

        scheduler = MultiAssetScheduler(feed)
        for symbol, bar in scheduler.iter_events(start_dt=start_dt, end_dt=end_dt):
            mark_price = Decimal(str(bar.close))
            self.mark_prices[symbol] = mark_price
            history = feed.get_history(symbol, lookbacks[symbol])
            if len(history) < lookbacks[symbol]:
                self._record_equity(bar.timestamp)
                continue

            signals = strategies[symbol].on_bar(history, symbol, feed.timeframe) or []
            for signal in signals:
                self._process_signal(
                    symbol=symbol, signal=signal, timestamp=bar.timestamp, price=mark_price
                )
            self._record_equity(bar.timestamp)

        return self._build_report(feed.timeframe)

    def _process_signal(self, *, symbol: str, signal: Any, timestamp: Any, price: Decimal) -> None:
        if not hasattr(signal, "side"):
            return

        state = self.positions.get(symbol)
        if state is None:
            if signal.side == OrderSide.BUY:
                self._open_position(
                    symbol=symbol, side=OrderSide.BUY, price=price, timestamp=timestamp
                )
                return
            if signal.side == OrderSide.SELL and self.config.allows_short_entry():
                self._open_position(
                    symbol=symbol, side=OrderSide.SELL, price=price, timestamp=timestamp
                )
            return

        if state.position.is_long and signal.side == OrderSide.SELL:
            self._close_position(
                symbol=symbol, price=price, timestamp=timestamp, reason="EXIT_SIGNAL"
            )
            return
        if state.position.is_short and signal.side == OrderSide.BUY:
            self._close_position(
                symbol=symbol, price=price, timestamp=timestamp, reason="EXIT_SIGNAL"
            )

    def _open_position(
        self, *, symbol: str, side: OrderSide, price: Decimal, timestamp: Any
    ) -> None:
        if side == OrderSide.BUY and not self.config.allows_long_entry():
            self.rejections.append(
                ExecutionRejection(symbol=symbol, timestamp=timestamp, reason="long_not_allowed")
            )
            return
        if side == OrderSide.SELL and not self.config.allows_short_entry():
            self.rejections.append(
                ExecutionRejection(symbol=symbol, timestamp=timestamp, reason="short_not_allowed")
            )
            return
        max_open = self.config.portfolio.max_open_positions
        if max_open is not None and len(self.positions) >= max_open:
            self.rejections.append(
                ExecutionRejection(symbol=symbol, timestamp=timestamp, reason="max_open_positions")
            )
            return

        equity = self._current_equity()
        decision = self._allocator.allocate(
            account=self.account,
            config=self.config,
            open_positions_count=len(self.positions),
            current_equity=equity,
        )
        if decision.notional <= 0:
            self.rejections.append(
                ExecutionRejection(
                    symbol=symbol,
                    timestamp=timestamp,
                    reason=decision.reason or "allocation_rejected",
                )
            )
            return

        side_sign = 1 if side == OrderSide.BUY else -1
        rejection = self._risk.check_open(
            config=self.config,
            equity=equity,
            current=self._current_exposure(),
            symbol=symbol,
            side_sign=side_sign,
            notional=decision.notional,
        )
        if rejection is not None:
            self.rejections.append(
                ExecutionRejection(symbol=symbol, timestamp=timestamp, reason=rejection)
            )
            return

        leverage = Decimal(str(self.config.leverage.leverage))
        margin_required = decision.notional / leverage
        commission = (decision.notional * self.commission_bps) / self._BPS_SCALE
        if not self.account.reserve(margin_required + commission):
            self.rejections.append(
                ExecutionRejection(symbol=symbol, timestamp=timestamp, reason="insufficient_cash")
            )
            return

        qty = decision.notional / price
        if side == OrderSide.SELL:
            qty = -qty
        position = Position(symbol=symbol, qty=qty, avg_entry_price=price)
        self.positions[symbol] = PortfolioPositionState(
            position=position, margin_used=margin_required
        )
        self.trades.append(
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "side": side.value,
                "action": "OPEN",
                "price": price,
                "qty": qty,
                "notional": decision.notional,
                "commission": commission,
            }
        )

    def _close_position(self, *, symbol: str, price: Decimal, timestamp: Any, reason: str) -> None:
        state = self.positions.get(symbol)
        if state is None:
            return
        qty = state.position.qty
        notional = abs(qty) * price
        commission = (notional * self.commission_bps) / self._BPS_SCALE
        pnl = (price - state.position.avg_entry_price) * qty
        self.account.release(state.margin_used)
        self.account.apply_realized_pnl(pnl - commission)
        del self.positions[symbol]
        self.trades.append(
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "side": "SELL" if qty > 0 else "BUY",
                "action": "CLOSE",
                "reason": reason,
                "price": price,
                "qty": abs(qty),
                "pnl": pnl - commission,
                "commission": commission,
            }
        )

    def _current_equity(self) -> Decimal:
        unrealized = Decimal("0")
        used_margin = Decimal("0")
        for symbol, state in self.positions.items():
            mark = self.mark_prices.get(symbol, state.position.avg_entry_price)
            unrealized += (mark - state.position.avg_entry_price) * state.position.qty
            used_margin += state.margin_used
        return self.account.cash + used_margin + unrealized

    def _current_exposure(self) -> PortfolioExposure:
        gross = Decimal("0")
        net = Decimal("0")
        by_symbol: dict[str, Decimal] = {}
        for symbol, state in self.positions.items():
            mark = self.mark_prices.get(symbol, state.position.avg_entry_price)
            notional = abs(state.position.qty) * mark
            gross += notional
            if state.position.is_long:
                net += notional
            elif state.position.is_short:
                net -= notional
            by_symbol[symbol] = notional
        return PortfolioExposure(
            gross_notional=gross, net_notional=net, by_symbol_notional=by_symbol
        )

    def _record_equity(self, timestamp: Any) -> None:
        self.equity_curve.append((timestamp, self._current_equity()))

    def _build_report(self, timeframe: str) -> dict[str, Any]:
        final_equity = self._current_equity()
        equity_values = [point[1] for point in self.equity_curve] or [
            self.initial_capital,
            final_equity,
        ]
        perf = compute_performance_metrics(
            equity_values,
            periods_per_year=infer_periods_per_year(timeframe),
            risk_free_rate=self.config.risk_free_rate,
        ).to_dict()
        closed_trades = [t for t in self.trades if t.get("action") == "CLOSE"]
        winning = sum(1 for t in closed_trades if Decimal(str(t.get("pnl", 0))) > 0)
        total = len(closed_trades)
        win_rate = (winning / total) if total else 0.0
        return {
            "run_mode": RunMode.PORTFOLIO_SHARED.value,
            "initial_capital": self.initial_capital,
            "final_capital": self.account.cash,
            "final_equity": final_equity,
            "pnl": final_equity - self.initial_capital,
            "performance": perf,
            "total_trades": len(self.trades),
            "trade_metrics": {"closed_trades": total, "win_rate": win_rate},
            "portfolio": {
                "open_positions": len(self.positions),
                "used_margin": sum(
                    (state.margin_used for state in self.positions.values()), start=Decimal("0")
                ),
                "rejections": [
                    {"symbol": item.symbol, "timestamp": item.timestamp, "reason": item.reason}
                    for item in self.rejections
                ],
            },
            "trades": self.trades,
            "equity_curve": [{"timestamp": ts, "equity": eq} for ts, eq in self.equity_curve],
        }
