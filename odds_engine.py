"""NFL betting model: market pricing utilities (not trained projections)."""
from dataclasses import dataclass


def implied_probability(american_odds: int) -> float:
    if american_odds == 0:
        raise ValueError('American odds cannot be zero')
    if american_odds > 0:
        return 100 / (american_odds + 100)
    return -american_odds / (-american_odds + 100)


def decimal_profit(american_odds: int) -> float:
    if american_odds == 0:
        raise ValueError('American odds cannot be zero')
    return american_odds / 100 if american_odds > 0 else 100 / -american_odds


def no_vig_two_way(odds_a: int, odds_b: int) -> tuple[float, float]:
    a, b = implied_probability(odds_a), implied_probability(odds_b)
    return a / (a + b), b / (a + b)


def expected_roi(model_probability: float, american_odds: int) -> float:
    if not 0 <= model_probability <= 1:
        raise ValueError('Probability must be between 0 and 1')
    return model_probability * decimal_profit(american_odds) - (1 - model_probability)


def kelly_fraction(model_probability: float, american_odds: int, fraction: float = 0.25) -> float:
    if not 0 <= model_probability <= 1 or not 0 <= fraction <= 1:
        raise ValueError('Probability and fraction must be between 0 and 1')
    b = decimal_profit(american_odds)
    return max(0.0, fraction * (b * model_probability - (1 - model_probability)) / b)


@dataclass(frozen=True)
class MarketQuote:
    game_id: str
    market: str
    selection: str
    line: float | None
    american_odds: int
    model_probability: float

    def evaluate(self) -> dict:
        return {
            'game_id': self.game_id,
            'market': self.market,
            'selection': self.selection,
            'line': self.line,
            'american_odds': self.american_odds,
            'model_probability': self.model_probability,
            'implied_probability': implied_probability(self.american_odds),
            'probability_edge': self.model_probability - implied_probability(self.american_odds),
            'expected_roi': expected_roi(self.model_probability, self.american_odds),
            'quarter_kelly_fraction': kelly_fraction(self.model_probability, self.american_odds),
        }
