from __future__ import annotations

from laakhay.quantlab.backtest.report import (
    BatchBacktestReport,
    SymbolBacktestReport,
    batch_report_to_dict,
    batch_report_to_legacy_dict,
    build_batch_independent_report,
)


def test_build_batch_independent_report_sets_envelope_and_aggregate():
    data = build_batch_independent_report(
        symbol_reports=[
            SymbolBacktestReport(
                symbol="BTCUSDT",
                timeframe="1h",
                report={
                    "performance": {"total_return": 0.12, "max_drawdown": -0.04},
                    "total_trades": 5,
                },
            ),
            SymbolBacktestReport(
                symbol="ETHUSDT",
                timeframe="1h",
                report={
                    "performance": {"total_return": 0.02, "max_drawdown": -0.06},
                    "total_trades": 7,
                },
            ),
        ]
    )
    assert isinstance(data, BatchBacktestReport)
    assert data.run_mode.value == "batch_independent"
    assert data.aggregate["symbols"] == 2
    assert data.aggregate["total_trades"] == 12


def test_batch_report_adapters_produce_v2_and_legacy_shapes():
    report = build_batch_independent_report(
        symbol_reports=[
            SymbolBacktestReport(
                symbol="BTCUSDT",
                timeframe="1h",
                report={
                    "performance": {"total_return": 0.12, "max_drawdown": -0.04},
                    "total_trades": 5,
                },
            )
        ],
        errors_by_symbol={"SOLUSDT": "missing bars"},
    )
    v2 = batch_report_to_dict(report)
    assert v2["run_mode"] == "batch_independent"
    assert len(v2["symbol_reports"]) == 1
    assert v2["errors_by_symbol"]["SOLUSDT"] == "missing bars"

    legacy = batch_report_to_legacy_dict(report)
    assert "aggregate" in legacy
    assert "reports_by_symbol" in legacy
    assert legacy["reports_by_symbol"]["BTCUSDT"]["total_trades"] == 5
