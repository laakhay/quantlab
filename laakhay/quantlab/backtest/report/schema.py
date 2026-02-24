from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..domain.enums import RunMode


@dataclass(frozen=True)
class SingleBacktestReport:
    run_mode: RunMode = RunMode.SINGLE
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SymbolBacktestReport:
    symbol: str
    timeframe: str
    report: dict[str, Any]


@dataclass(frozen=True)
class BatchBacktestReport:
    run_mode: RunMode = RunMode.BATCH_INDEPENDENT
    symbol_reports: list[SymbolBacktestReport] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    errors_by_symbol: dict[str, str] = field(default_factory=dict)
