# Decoupling Quantlab from TA/Data and Delegating Orchestration to Kendra

## Executive Answer
Yes, decoupling is feasible and recommended.

Current state has hard compile-time coupling to both `laakhay-data` and `laakhay-ta` inside Quantlab backtest strategy/feed layers. For long-term robustness, Quantlab should be a pure simulation/execution engine, while Kendra should orchestrate data acquisition + expression/indicator evaluation.

This plan describes how to do that aggressively (still pragmatic) without breaking backtest correctness.

---

## What is Coupled Today (Concrete)

## Direct TA coupling in Quantlab
- `quantlab/laakhay/quantlab/backtest/strategy/base.py`
  - imports:
    - `laakhay.ta.core.dataset.Dataset`
    - `laakhay.ta.expr.dsl.compiler.ExpressionCompiler`
    - `laakhay.ta.expr.dsl.parser.ExpressionParser`
    - `laakhay.ta.expr.planner.planner.plan_expression`
    - `laakhay.ta.expr.runtime.evaluator.RuntimeEvaluator`

## Direct Data coupling in Quantlab
- `quantlab/laakhay/quantlab/backtest/feed.py`
  - imports `laakhay.data.models.bar.Bar`
  - converts bars into TA `Dataset`/`OHLCV` directly
- `quantlab/laakhay/quantlab/backtest/models/__init__.py`
  - exports `Bar` from `laakhay.data.models.bar`
- tests also import data Bar directly

## Packaging coupling
- `quantlab/pyproject.toml`
  - hard runtime deps include both `laakhay-data` and `laakhay-ta`

Implication:
- Quantlab currently owns domain execution and expression runtime integration together.
- This makes library boundaries muddy and blocks independent evolution/deployment.

---

## Architectural Recommendation

Use **ports and adapters**:
- Quantlab core: simulation, order management, PnL, risk, metrics.
- Strategy signal production: externalized behind interfaces.
- Kendra: orchestration runtime (data + TA evaluation + strategy semantics).

### Boundary decision
- Quantlab should not import TA parser/compiler/runtime internals.
- Quantlab should not depend on a concrete `laakhay-data` `Bar` class.
- Quantlab should consume normalized market events and strategy decisions via contracts.

---

## Target Architecture

```text
quantlab/
  laakhay/quantlab/
    backtest/
      engine/               # pure execution
      models/               # quantlab-native contracts (no ta/data imports)
      ports/                # interfaces
        market_data.py
        signal_provider.py
        strategy_runtime.py
      adapters/
        local_ta/           # optional plugin adapter (dev/test only)
        local_data/         # optional plugin adapter (dev/test only)
        kendra/             # remote orchestration adapter (primary)
```

### Core ports

1. `MarketDataPort`
- `stream(...) -> Iterator[MarketBar]`
- `history(symbol, timeframe, lookback) -> MarketWindow`

2. `SignalProviderPort`
- `evaluate(context) -> list[Signal]`
- may be remote (Kendra) or local plugin

3. `StrategyRuntimePort`
- optional richer API for precompiled strategy handles (Kendra-managed)

### Core contracts (Quantlab-owned)

- `MarketBar` (timestamp/open/high/low/close/volume/is_closed + metadata)
- `MarketWindow` (minimal, serializable)
- `SignalDecision` (entry/exit/size/sl/tp + debug fields)

No imports from TA/Data types in these contracts.

---

## Delegation Model with Kendra

Kendra should own:
1. Strategy expression parsing/validation/compilation
2. Indicator execution and emissions
3. Source-specific data resolution/fetching
4. Capability/source constraints

Quantlab should own:
1. fill/OMS lifecycle
2. position accounting
3. constraints (cooldown, min hold, max bars, daily loss)
4. equity/metrics/reporting

### Interaction shape

Quantlab -> Kendra (request)
- symbol, timeframe, lookback window, latest bar, strategy ref/config

Kendra -> Quantlab (response)
- entry/exit booleans (or signals)
- optional debug payload (indicator values, eval flags)

This keeps execution deterministic in Quantlab while outsourcing strategy intelligence.

---

## Migration Plan (Step-by-Step)

## Phase 1: Introduce Quantlab-native contracts (no behavior change)

### 1.1 Add internal market model
Create:
- `quantlab/laakhay/quantlab/backtest/models/market.py`

Define:
- `MarketBar`
- `MarketWindow`

### 1.2 Add protocol ports
Create:
- `quantlab/laakhay/quantlab/backtest/ports/market_data.py`
- `quantlab/laakhay/quantlab/backtest/ports/signal_provider.py`

### 1.3 Update engine signatures to depend on ports/contracts
Edit:
- `backtest/engine/core.py`
- `backtest/feed.py` (or replace with adapter layer)

Keep compatibility adapters temporarily inside quantlab if needed, but no new direct ta/data imports in core engine.

Exit criteria:
- Backtest engine compiles/runs with port abstractions.

---

## Phase 2: Remove TA internals from Strategy base

### 2.1 Split strategy concerns
Current `Strategy` mixes:
- expression parsing/planning/evaluation (TA concern)
- signal shaping (Quantlab concern)

Refactor:
- `Strategy` becomes thin signal consumer/normalizer.
- move expression runtime to adapter.

Create:
- `backtest/adapters/local_ta/signal_provider.py` (optional local plugin)

Edit:
- `backtest/strategy/base.py` to depend on `SignalProviderPort`, not TA runtime.

Exit criteria:
- core strategy path has zero imports from `laakhay.ta.*`.

---

## Phase 3: Remove direct data model dependency

### 3.1 Replace `laakhay.data.models.bar.Bar` usage
Edit:
- `backtest/feed.py`
- `backtest/models/__init__.py`
- tests importing data Bar

Use `MarketBar` in quantlab core.

### 3.2 Local data adapter
Create optional adapter:
- `backtest/adapters/local_data/feed.py`

It converts `laakhay-data` bars -> `MarketBar`.

Exit criteria:
- core quantlab backtest no longer imports `laakhay.data.*`.

---

## Phase 4: Kendra adapter (primary orchestration path)

### 4.1 Add Kendra client adapter
Create:
- `backtest/adapters/kendra/client.py`
- `backtest/adapters/kendra/signal_provider.py`

Responsibilities:
- call Kendra strategy-eval endpoint
- map response -> `SignalDecision`
- retry/timeout/error policy

### 4.2 Add deterministic cache options
- optional local cache keyed by `(strategy_id, symbol, timeframe, bar_ts)` for replay consistency.

Exit criteria:
- quantlab can run backtests using Kendra signals only.

---

## Phase 5: Packaging split and dependency cleanup

### 5.1 Make TA/Data optional extras
Edit `quantlab/pyproject.toml`:
- remove hard runtime deps on `laakhay-data` and `laakhay-ta` from base dependencies
- add extras:
  - `local-ta` -> `laakhay-ta`
  - `local-data` -> `laakhay-data`
  - `kendra` -> http client deps

### 5.2 Update docs
- explain base engine vs optional adapters

Exit criteria:
- installing core quantlab does not pull ta/data by default.

---

## Phase 6: Test strategy

### 6.1 Contract tests (must)
- `ports` conformance tests
- adapter mapping tests (local_ta, local_data, kendra)

### 6.2 Engine invariants
- same signal sequence => identical trades/pnl regardless of provider backend

### 6.3 Golden scenario tests
- fixed bar feed + fixed kendra responses
- verify round-trip trades/equity are stable

---

## Risks and Mitigations

1. Remote orchestration adds latency/failure modes
- Mitigation: batched eval endpoints, retries, timeout fallback policy, offline replay cache.

2. Reproducibility drift between local TA and Kendra
- Mitigation: same strategy contract schema; parity test suite comparing local adapter vs kendra responses on fixed datasets.

3. Increased complexity from adapter layer
- Mitigation: strict ports, minimal contracts, no business logic in adapters.

4. Version skew across services
- Mitigation: explicit strategy/runtime contract versioning in requests.

---

## What Not to Do

- Do not move OMS/position lifecycle into Kendra. Keep execution accounting in Quantlab.
- Do not keep dual hidden TA paths inside engine. All strategy evaluation should go through a `SignalProviderPort`.
- Do not keep `laakhay-data` model types in core backtest interfaces.

---

## Suggested First PRs (order)

1. Introduce `MarketBar` + ports and wire engine to them (no behavior change).
2. Add local adapters for current TA/data behavior behind ports.
3. Remove direct TA/data imports from core backtest packages.
4. Add Kendra adapter + integration tests.
5. Flip default examples/docs to Kendra-orchestrated mode.

---

## Definition of Done

Done when:
1. Quantlab core backtest imports neither `laakhay-ta` nor `laakhay-data`.
2. Strategy evaluation is provided via pluggable `SignalProviderPort`.
3. Kendra adapter is production path; local TA/data adapters are optional.
4. Base quantlab package installs/runs without ta/data dependencies.
5. Backtest outputs are reproducible across providers given same signal stream.

