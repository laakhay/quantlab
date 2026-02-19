from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from laakhay.ta.core.dataset import Dataset

from ..config import BacktestConfig
from ..feed import DataFeed
from ..metrics import (
    RoundTripTrade,
    compute_performance_metrics,
    compute_trade_metrics,
    infer_periods_per_year,
)
from ..models import OrderSide, OrderType, Position
from .oms import OrderManager
from .sizer import PositionSizer


@dataclass
class PortfolioPoint:
    """Historical point in the portfolio's equity curve."""

    timestamp: Any
    equity: Decimal


@dataclass
class PositionRuntimeState:
    """Runtime state for frequency and exit controls."""

    entry_side: OrderSide | None = None
    entry_price: Decimal | None = None
    entry_qty: Decimal | None = None
    entry_time: Any = None
    entry_bar_index: int | None = None
    entry_commission: Decimal = Decimal("0")
    initial_stop: Decimal | None = None
    initial_risk: Decimal | None = None
    breakeven_moved: bool = False
    last_exit_bar_index: int | None = None
    entries_day: Any = None
    entries_today: int = 0


class PortfolioProxy:
    """A proxy object to provide a consistent interface for portfolio results."""

    def __init__(self, engine: BacktestEngine):
        self._engine = engine

    @property
    def equity(self) -> Decimal:
        return self._engine.current_equity

    @property
    def trades(self) -> list[Any]:
        return self._engine.trades

    @property
    def history(self) -> list[PortfolioPoint]:
        return self._engine.history


class BacktestEngine:
    """Event-driven backtesting engine."""

    _BPS_SCALE = Decimal("10000")

    def __init__(
        self,
        initial_capital: float | Decimal = 10000.0,
        *,
        commission_bps: float | Decimal = 0,
        slippage_bps: float | Decimal = 0,
        config: BacktestConfig | None = None,
    ):
        """Initialize engine."""
        self.capital = Decimal(str(initial_capital))
        self.initial_capital = self.capital
        self.positions: dict[str, Position] = {}
        self.current_time = None
        self.last_prices: dict[str, Decimal] = {}
        self.history: list[PortfolioPoint] = []
        self.closed_trades: list[RoundTripTrade] = []
        self._runtime: dict[str, PositionRuntimeState] = {}
        self._bar_index = -1
        self._timeframe = "1d"
        self._session_day: Any = None
        self._day_start_equity = self.initial_capital
        self._daily_realized_pnl = Decimal("0")

        self.config = config or BacktestConfig()
        self.commission_bps = Decimal(str(commission_bps))
        self.slippage_bps = Decimal(str(slippage_bps))
        if self.commission_bps < 0:
            raise ValueError("commission_bps must be >= 0")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be >= 0")
        if self.slippage_bps >= self._BPS_SCALE:
            raise ValueError("slippage_bps must be < 10000")

        self.oms = OrderManager()
        self.sizer = PositionSizer(self._get_capital)

    def _get_capital(self) -> Decimal:
        return self.capital

    def _runtime_state(self, symbol: str) -> PositionRuntimeState:
        state = self._runtime.get(symbol)
        if state is None:
            state = PositionRuntimeState()
            self._runtime[symbol] = state
        return state

    def run(self, strategy: Any, feed: DataFeed) -> dict[str, Any]:
        """Run the backtest loop."""
        self.positions.clear()
        self.capital = self.initial_capital
        self.current_time = None
        self.last_prices.clear()
        self.history.clear()
        self.closed_trades.clear()
        self._runtime.clear()
        self._bar_index = -1
        self._timeframe = feed.timeframe
        self._session_day = None
        self._day_start_equity = self.initial_capital
        self._daily_realized_pnl = Decimal("0")
        self.oms = OrderManager()
        self.sizer = PositionSizer(self._get_capital)

        strategy.prepare(feed.symbol, feed.timeframe)
        required_lookback = self._resolve_required_lookback(strategy)

        for bar in feed.stream():
            self._bar_index += 1
            self.current_time = bar.timestamp
            self.last_prices[feed.symbol] = Decimal(str(bar.close))
            self._roll_session_day(bar.timestamp)

            # 1) Match pending exit orders.
            filled_orders = self.oms.match_orders(bar, feed.symbol)
            for order, fill_price in filled_orders:
                trade = self._execute_trade(
                    order.symbol,
                    order.side,
                    order.qty,
                    fill_price,
                    order_id=order.id,
                    exit_reason=(
                        "STOP_LOSS"
                        if order.label == "SL"
                        else "TAKE_PROFIT"
                        if order.label == "TP"
                        else "ORDER_FILL"
                    ),
                )
                if trade and order.label in {"SL", "TP"}:
                    self.oms.cancel_symbol_orders(order.symbol)

            # 2) Time exits and stop adjustments (e.g., breakeven).
            self._apply_time_exit(feed.symbol, Decimal(str(bar.close)))
            self._apply_breakeven(feed.symbol, Decimal(str(bar.close)))

            # 3) Build context and evaluate strategy.
            market_data: Dataset = feed.get_history(feed.symbol, lookback=required_lookback)
            series = market_data.series(feed.symbol, feed.timeframe, source="ohlcv")
            if series and len(series) >= required_lookback:
                try:
                    signal_or_list = strategy.on_bar(market_data, feed.symbol, feed.timeframe)
                except Exception as e:  # pragma: no cover - strategy safety
                    print(f"Strategy Error at {self.current_time}: {e}")
                    signal_or_list = []

                if signal_or_list:
                    signals = (
                        signal_or_list if isinstance(signal_or_list, list) else [signal_or_list]
                    )
                    for signal in signals:
                        if signal is not None:
                            self._process_signal(signal, bar.close)

            # 4) Record equity snapshot.
            self.history.append(PortfolioPoint(timestamp=bar.timestamp, equity=self.current_equity))

        return self._generate_report()

    def _resolve_required_lookback(self, strategy: Any) -> int:
        required = getattr(strategy, "required_lookback", None)
        if callable(required):
            try:
                value = int(required())
                return max(2, value)
            except Exception:
                return 200
        return 200

    def _day_key(self, timestamp: Any) -> Any:
        if hasattr(timestamp, "date"):
            try:
                return timestamp.date()
            except Exception:  # pragma: no cover - defensive timestamp handling
                return timestamp
        return timestamp

    def _roll_session_day(self, timestamp: Any) -> None:
        day = self._day_key(timestamp)
        if day == self._session_day:
            return
        self._session_day = day
        self._daily_realized_pnl = Decimal("0")
        self._day_start_equity = self.current_equity

    def _is_daily_loss_limit_hit(self) -> bool:
        loss_fraction = self.config.daily_loss_limit_fraction
        if loss_fraction is None:
            return False
        if self._day_start_equity <= 0:
            return False
        max_loss = self._day_start_equity * loss_fraction
        return self._daily_realized_pnl <= -max_loss

    def _can_enter(self, symbol: str, side: OrderSide) -> bool:
        if side == OrderSide.BUY and not self.config.allows_long_entry():
            return False
        if side == OrderSide.SELL and not self.config.allows_short_entry():
            return False
        if self._is_daily_loss_limit_hit():
            return False

        runtime = self._runtime_state(symbol)
        max_entries_per_day = self.config.frequency.max_entries_per_day
        if max_entries_per_day is not None:
            if runtime.entries_day != self._session_day:
                runtime.entries_day = self._session_day
                runtime.entries_today = 0
            if runtime.entries_today >= max_entries_per_day:
                return False

        last_exit = runtime.last_exit_bar_index
        if last_exit is None:
            return True
        if not self.config.allow_entry_same_bar_as_exit and self._bar_index == last_exit:
            return False
        cooldown = self.config.frequency.cooldown_bars
        return not (cooldown > 0 and (self._bar_index - last_exit) <= cooldown)

    def _can_discretionary_exit(self, symbol: str) -> bool:
        min_hold = self.config.frequency.min_hold_bars
        if min_hold <= 0:
            return True
        runtime = self._runtime_state(symbol)
        if runtime.entry_bar_index is None:
            return True
        return (self._bar_index - runtime.entry_bar_index) >= min_hold

    def _process_signal(self, signal: Any, current_price: Any) -> None:
        """Process a Signal object."""
        if isinstance(signal, dict):
            return

        symbol = signal.symbol
        current_px = Decimal(str(current_price))
        if current_px <= 0:
            return

        pos = self.positions.get(symbol, Position(symbol=symbol))

        if signal.side == OrderSide.BUY:
            if pos.is_flat:
                if not self._can_enter(symbol, OrderSide.BUY):
                    return
                self._open_position(symbol, signal, current_px, OrderSide.BUY)
                return
            if pos.is_short and self._can_discretionary_exit(symbol):
                trade = self._execute_trade(
                    symbol,
                    OrderSide.BUY,
                    pos.abs_qty,
                    current_px,
                    exit_reason="EXIT_SIGNAL",
                )
                if trade:
                    self.oms.cancel_symbol_orders(symbol)
            return

        if signal.side == OrderSide.SELL:
            if pos.is_flat:
                if not self._can_enter(symbol, OrderSide.SELL):
                    return
                self._open_position(symbol, signal, current_px, OrderSide.SELL)
                return
            if pos.is_long and self._can_discretionary_exit(symbol):
                trade = self._execute_trade(
                    symbol,
                    OrderSide.SELL,
                    pos.abs_qty,
                    current_px,
                    exit_reason="EXIT_SIGNAL",
                )
                if trade:
                    self.oms.cancel_symbol_orders(symbol)

    def _open_position(
        self, symbol: str, signal: Any, current_px: Decimal, side: OrderSide
    ) -> None:
        qty = self.sizer.calculate_size(signal, current_px)
        if qty <= 0:
            return

        entry_trade = self._execute_trade(symbol, side, qty, current_px)
        if entry_trade is None:
            return

        self._setup_exit_orders(symbol, signal, entry_trade)

    def _setup_exit_orders(self, symbol: str, signal: Any, entry_trade: Any) -> None:
        entry_side = entry_trade.side
        exit_side = OrderSide.SELL if entry_side == OrderSide.BUY else OrderSide.BUY
        oco_group_id = f"OCO-{symbol}-{entry_trade.id}"

        sl_price: Decimal | None = None
        if signal.sl is not None:
            try:
                sl_price = self._parse_price(
                    signal.sl, entry_trade.price, is_sl=True, side=entry_side
                )
            except ValueError:
                sl_price = None
            if sl_price is not None:
                self.oms.create_order(
                    symbol=symbol,
                    side=exit_side,
                    qty=entry_trade.qty,
                    type=OrderType.STOP,
                    price=sl_price,
                    label="SL",
                    parent_order_id=entry_trade.order_id,
                    oco_group_id=oco_group_id,
                    timestamp=self.current_time,
                )

        if signal.tp is not None:
            try:
                tp_price = self._parse_price(
                    signal.tp, entry_trade.price, is_sl=False, side=entry_side
                )
            except ValueError:
                tp_price = None
            if tp_price is not None:
                self.oms.create_order(
                    symbol=symbol,
                    side=exit_side,
                    qty=entry_trade.qty,
                    type=OrderType.LIMIT,
                    price=tp_price,
                    label="TP",
                    parent_order_id=entry_trade.order_id,
                    oco_group_id=oco_group_id,
                    timestamp=self.current_time,
                )

        runtime = self._runtime_state(symbol)
        if sl_price is None:
            runtime.initial_stop = None
            runtime.initial_risk = None
            return

        if entry_side == OrderSide.BUY:
            risk = entry_trade.price - sl_price
        else:
            risk = sl_price - entry_trade.price
        if risk > 0:
            runtime.initial_stop = sl_price
            runtime.initial_risk = risk

    def _apply_time_exit(self, symbol: str, mark_price: Decimal) -> None:
        max_bars = self.config.frequency.max_bars_in_trade
        if max_bars is None:
            return

        pos = self.positions.get(symbol)
        if pos is None or pos.is_flat:
            return

        runtime = self._runtime_state(symbol)
        if runtime.entry_bar_index is None:
            return
        held_bars = self._bar_index - runtime.entry_bar_index
        if held_bars < max_bars:
            return

        exit_side = OrderSide.SELL if pos.is_long else OrderSide.BUY
        trade = self._execute_trade(
            symbol,
            exit_side,
            pos.abs_qty,
            mark_price,
            exit_reason="TIME_EXIT",
        )
        if trade:
            self.oms.cancel_symbol_orders(symbol)

    def _apply_breakeven(self, symbol: str, mark_price: Decimal) -> None:
        config = self.config.breakeven
        if not config.enabled:
            return

        pos = self.positions.get(symbol)
        if pos is None or pos.is_flat:
            return

        runtime = self._runtime_state(symbol)
        if runtime.breakeven_moved:
            return
        if (
            runtime.entry_side is None
            or runtime.entry_price is None
            or runtime.initial_risk is None
        ):
            return
        if runtime.initial_risk <= 0:
            return

        rr_trigger = runtime.initial_risk * Decimal(str(config.trigger_rr))
        entry_price = runtime.entry_price

        should_move = False
        target_stop: Decimal | None = None
        offset = config.offset_fraction
        sl_order = self.oms.find_active_order(symbol, "SL")
        if sl_order is None or sl_order.price is None:
            return

        if runtime.entry_side == OrderSide.BUY:
            if mark_price >= entry_price + rr_trigger:
                target_stop = entry_price * (1 + offset)
                should_move = target_stop > sl_order.price
        else:
            if mark_price <= entry_price - rr_trigger:
                target_stop = entry_price * (1 - offset)
                should_move = target_stop < sl_order.price

        if should_move and target_stop is not None:
            updated = self.oms.update_order_price(sl_order.id, target_stop)
            if updated:
                runtime.breakeven_moved = True

    def _parse_price(
        self, val: str | Decimal, entry_price: Decimal, is_sl: bool, side: OrderSide
    ) -> Decimal:
        """Parse price or percentage offset."""
        if isinstance(val, Decimal):
            if val <= 0:
                raise ValueError("Price levels must be positive")
            return val

        if isinstance(val, str) and val.endswith("%"):
            pct = Decimal(val[:-1]) / 100
            if pct <= 0:
                raise ValueError("Percentage offsets must be positive")
            if side == OrderSide.BUY:
                parsed = entry_price * (1 - pct) if is_sl else entry_price * (1 + pct)
            else:
                parsed = entry_price * (1 + pct) if is_sl else entry_price * (1 - pct)
            if parsed <= 0:
                raise ValueError("Computed price level must be positive")
            return parsed

        parsed = Decimal(str(val))
        if parsed <= 0:
            raise ValueError("Price levels must be positive")
        return parsed

    def _apply_slippage(self, side: OrderSide, price: Decimal) -> Decimal:
        if self.slippage_bps == 0:
            return price
        impact = self.slippage_bps / self._BPS_SCALE
        if side == OrderSide.BUY:
            return price * (1 + impact)
        return price * (1 - impact)

    def _execute_trade(
        self,
        symbol: str,
        side: OrderSide,
        qty: Any,
        price: Any,
        order_id: str | None = None,
        exit_reason: str | None = None,
    ) -> Any | None:
        """Execute trade and update state."""
        from ..models import Trade

        qty_dec = Decimal(str(qty))
        raw_price = Decimal(str(price))
        if qty_dec <= 0 or raw_price <= 0:
            return None

        position = self.positions.get(symbol)
        current_qty = position.qty if position else Decimal("0")

        # This engine uses one-way netting without pyramiding/reversal.
        if (current_qty > 0 and side == OrderSide.BUY) or (
            current_qty < 0 and side == OrderSide.SELL
        ):
            return None
        if current_qty > 0 and side == OrderSide.SELL and qty_dec != current_qty:
            return None
        if current_qty < 0 and side == OrderSide.BUY and qty_dec != abs(current_qty):
            return None

        fill_price = self._apply_slippage(side, raw_price)
        if fill_price <= 0:
            return None

        notional = qty_dec * fill_price
        commission = (notional * self.commission_bps) / self._BPS_SCALE

        # Cash guardrail for long entries.
        if side == OrderSide.BUY and current_qty >= 0:
            total_cost = notional + commission
            if total_cost > self.capital:
                return None

        if order_id is None:
            order_id = f"O-{len(self.oms.orders_history) + 1}"

        trade = Trade(
            id=f"T-{len(self.trades) + 1}",
            order_id=order_id,
            symbol=symbol,
            side=side,
            qty=qty_dec,
            price=fill_price,
            commission=commission,
            timestamp=self.current_time,
        )

        if position is None:
            position = Position(symbol=symbol)
            self.positions[symbol] = position
        previous_qty = position.qty
        position.update(trade)

        self.trades.append(trade)

        if side == OrderSide.BUY:
            self.capital -= notional + commission
        else:
            self.capital += notional - commission

        self._sync_runtime_after_trade(
            symbol,
            trade,
            previous_qty,
            position.qty,
            exit_reason=exit_reason,
        )
        return trade

    def _sync_runtime_after_trade(
        self,
        symbol: str,
        trade: Any,
        previous_qty: Decimal,
        new_qty: Decimal,
        *,
        exit_reason: str | None = None,
    ) -> None:
        runtime = self._runtime_state(symbol)

        if previous_qty == 0 and new_qty != 0:
            if runtime.entries_day != self._session_day:
                runtime.entries_day = self._session_day
                runtime.entries_today = 0
            runtime.entries_today += 1
            runtime.entry_side = trade.side
            runtime.entry_price = trade.price
            runtime.entry_qty = abs(new_qty)
            runtime.entry_time = trade.timestamp
            runtime.entry_bar_index = self._bar_index
            runtime.entry_commission = trade.commission
            runtime.initial_stop = None
            runtime.initial_risk = None
            runtime.breakeven_moved = False
            return

        if previous_qty != 0 and new_qty == 0:
            net_pnl = self._record_closed_trade(
                symbol,
                trade,
                runtime,
                exit_reason=exit_reason,
            )
            if net_pnl is not None:
                self._daily_realized_pnl += net_pnl
            runtime.last_exit_bar_index = self._bar_index
            runtime.entry_side = None
            runtime.entry_price = None
            runtime.entry_qty = None
            runtime.entry_time = None
            runtime.entry_bar_index = None
            runtime.entry_commission = Decimal("0")
            runtime.initial_stop = None
            runtime.initial_risk = None
            runtime.breakeven_moved = False

    def _record_closed_trade(
        self,
        symbol: str,
        exit_trade: Any,
        runtime: PositionRuntimeState,
        *,
        exit_reason: str | None = None,
    ) -> Decimal | None:
        if (
            runtime.entry_side is None
            or runtime.entry_price is None
            or runtime.entry_qty is None
            or runtime.entry_time is None
        ):
            return None

        qty_dec = runtime.entry_qty
        entry_price_dec = runtime.entry_price
        exit_price_dec = exit_trade.price
        entry_commission_dec = runtime.entry_commission
        exit_commission_dec = exit_trade.commission

        if runtime.entry_side == OrderSide.BUY:
            direction = "long"
            gross = (exit_price_dec - entry_price_dec) * qty_dec
        else:
            direction = "short"
            gross = (entry_price_dec - exit_price_dec) * qty_dec
        net_pnl = gross - entry_commission_dec - exit_commission_dec
        base_notional = entry_price_dec * qty_dec
        pnl_pct = (net_pnl / base_notional) if base_notional > 0 else Decimal("0")

        qty = float(qty_dec)
        entry_price = float(entry_price_dec)
        exit_price = float(exit_price_dec)

        holding_seconds = 0.0
        if hasattr(exit_trade.timestamp, "__sub__"):
            try:
                holding_seconds = max(
                    0.0, float((exit_trade.timestamp - runtime.entry_time).total_seconds())
                )
            except Exception:  # pragma: no cover - defensive timestamp handling
                holding_seconds = 0.0
        holding_bars = 0
        if runtime.entry_bar_index is not None:
            holding_bars = max(0, self._bar_index - runtime.entry_bar_index)

        self.closed_trades.append(
            RoundTripTrade(
                symbol=symbol,
                direction=direction,
                qty=qty,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=float(net_pnl),
                pnl_pct=float(pnl_pct),
                holding_seconds=holding_seconds,
                holding_bars=holding_bars,
                entry_time=runtime.entry_time,
                exit_time=exit_trade.timestamp,
                exit_reason=exit_reason,
            )
        )
        return net_pnl

    @property
    def trades(self):
        return self.oms.trades

    @property
    def current_equity(self) -> Decimal:
        """Total portfolio value (capital + unrealized PnL)."""
        market_value = Decimal("0")
        for symbol, pos in self.positions.items():
            if pos.is_flat:
                continue
            last_px = self.last_prices.get(symbol, pos.avg_entry_price)
            market_value += pos.qty * last_px
        return self.capital + market_value

    @property
    def portfolio(self) -> PortfolioProxy:
        """Backwards compatibility for reporting interfaces."""
        return PortfolioProxy(self)

    def _generate_report(self) -> dict[str, Any]:
        """Generate a performance report."""
        open_positions = {
            symbol: pos
            for symbol, pos in self.positions.items()
            if getattr(pos, "qty", Decimal("0")) != 0
        }
        open_market_value = Decimal("0")
        unrealized_pnl = Decimal("0")
        for symbol, pos in open_positions.items():
            last_price = self.last_prices.get(symbol)
            if last_price is None:
                continue
            open_market_value += pos.qty * last_price
            unrealized_pnl += (last_price - pos.avg_entry_price) * pos.qty

        final_equity = self.capital + open_market_value
        pnl = final_equity - self.initial_capital

        equity_values = [point.equity for point in self.history]
        if not equity_values:
            equity_values = [self.initial_capital, self.current_equity]
        periods_per_year = infer_periods_per_year(self._timeframe, fallback=252)
        performance = compute_performance_metrics(
            equity_values,
            periods_per_year=periods_per_year,
            risk_free_rate=self.config.risk_free_rate,
        )
        trade_metrics = compute_trade_metrics(self.closed_trades)

        return {
            "final_capital": self.capital,
            "final_equity": final_equity,
            "open_market_value": open_market_value,
            "unrealized_pnl": unrealized_pnl,
            "pnl": pnl,
            "total_trades": len(self.trades),
            "positions": self.positions,
            "open_positions": open_positions,
            "equity_curve": [{"timestamp": p.timestamp, "equity": p.equity} for p in self.history],
            "round_trips": [trade.__dict__ for trade in self.closed_trades],
            "performance": performance.to_dict(),
            "trade_metrics": trade_metrics.to_dict(),
            "risk_controls": {
                "daily_realized_pnl": self._daily_realized_pnl,
                "daily_loss_limit_hit": self._is_daily_loss_limit_hit(),
            },
        }
