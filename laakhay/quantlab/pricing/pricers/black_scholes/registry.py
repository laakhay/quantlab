"""Registry system for Black-Scholes pricing formulas."""

from abc import ABC, abstractmethod
from laakhay.quantlab.backend import Backend
from ...market import MarketData
from ...greeks import Greeks


class PricingFormula(ABC):
    """Abstract base class for pricing formulas."""

    @abstractmethod
    def price(self, option, market: MarketData) -> object:
        """Calculate option price given market data."""
        pass

    @abstractmethod
    def supports(self, option) -> bool:
        """Check if this formula supports the given option type."""
        pass


class PricingFormulaRegistry:
    """Registry for managing pricing formulas."""

    def __init__(self):
        self._formulas: dict[type, PricingFormula] = {}
        self._priority_formulas: list[PricingFormula] = []

    def register(self, option_type: type, formula: PricingFormula) -> None:
        self._formulas[option_type] = formula

    def register_priority(self, formula: PricingFormula) -> None:
        self._priority_formulas.append(formula)

    def get_formula(self, option) -> PricingFormula:
        option_type = type(option)

        for formula in self._priority_formulas:
            if formula.supports(option):
                return formula

        if option_type in self._formulas:
            return self._formulas[option_type]

        for registered_type, formula in self._formulas.items():
            if isinstance(option, registered_type):
                return formula

        raise ValueError(f"No pricing formula registered for option type: {option_type}")

    def price(self, option, market: MarketData) -> object:
        if isinstance(option, (list, tuple, set)):
            backend = market.backend
            total_price = None
            for single_option in option:
                single_price = self.price(single_option, market)
                if total_price is None:
                    total_price = single_price
                else:
                    total_price = backend.add(total_price, single_price)
            return total_price if total_price is not None else 0.0

        formula = self.get_formula(option)
        return formula.price(option, market)

    def price_with_greeks(self, option, market: MarketData) -> tuple[object, Greeks]:
        if isinstance(option, (list, tuple, set)):
            backend = market.backend
            total_price = None
            total_greeks = None

            for single_option in option:
                single_price, single_greeks = self.price_with_greeks(single_option, market)

                if total_price is None:
                    total_price = single_price
                    total_greeks = single_greeks
                else:
                    total_price = backend.add(total_price, single_price)
                    total_greeks = Greeks(
                        delta=backend.add(total_greeks.delta, single_greeks.delta),
                        gamma=backend.add(total_greeks.gamma, single_greeks.gamma),
                        vega=backend.add(total_greeks.vega, single_greeks.vega),
                        theta=backend.add(total_greeks.theta, single_greeks.theta),
                        rho=backend.add(total_greeks.rho, single_greeks.rho),
                    )

            final_price = total_price if total_price is not None else 0.0
            final_greeks = total_greeks if total_greeks is not None else Greeks()
            return final_price, final_greeks

        formula = self.get_formula(option)

        if hasattr(formula, "price_with_greeks"):
            return formula.price_with_greeks(option, market)

        price = formula.price(option, market)
        backend = market.backend
        zero = backend.zeros_like(price) if hasattr(backend, "zeros_like") else 0.0
        fallback_greeks = Greeks(delta=zero, gamma=zero, vega=zero, theta=zero, rho=zero)

        return price, fallback_greeks

    def supports(self, option) -> bool:
        if isinstance(option, (list, tuple, set)):
            return all(self.supports(single_option) for single_option in option)

        try:
            self.get_formula(option)
            return True
        except ValueError:
            return False

    def clear(self) -> None:
        self._formulas.clear()
        self._priority_formulas.clear()


pricing_registry = PricingFormulaRegistry()
