"""Load and validate the user-editable portfolio YAML.

No instrument tickers or weights live in code. Invalid configs raise ConfigError
with a message that names the failing rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.paths import DEFAULT_CONFIG_PATH

ALLOWED_SLEEVES = frozenset({"rates", "ig_credit", "hy_em", "fx", "alternatives"})
WEIGHT_SUM_TOLERANCE = 0.001


class ConfigError(ValueError):
    """Raised when ``config/portfolio.yaml`` fails validation."""


@dataclass(frozen=True)
class Position:
    ticker: str
    sleeve: str
    weight: float
    rationale: str = ""


@dataclass(frozen=True)
class PortfolioConfig:
    portfolio_name: str
    base_currency: str
    total_capital: float
    benchmark: str
    positions: tuple[Position, ...]
    source_path: Path

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]

    @property
    def weights(self) -> dict[str, float]:
        return {p.ticker: p.weight for p in self.positions}

    @property
    def sleeve_of(self) -> dict[str, str]:
        return {p.ticker: p.sleeve for p in self.positions}

    @property
    def sleeve_weights(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for pos in self.positions:
            totals[pos.sleeve] = totals.get(pos.sleeve, 0.0) + pos.weight
        return totals

    @property
    def download_tickers(self) -> list[str]:
        """Position tickers plus benchmark (deduplicated, positions first)."""
        seen: list[str] = []
        for ticker in [*self.tickers, self.benchmark]:
            if ticker not in seen:
                seen.append(ticker)
        return seen

    def weight_sum(self) -> float:
        return float(sum(p.weight for p in self.positions))


def load_portfolio_config(path: Path | None = None) -> PortfolioConfig:
    """Parse YAML at ``path`` (default: ``config/portfolio.yaml``) and validate."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(
            f"Portfolio config not found: {config_path}. "
            "Create config/portfolio.yaml with positions, weights, and a benchmark."
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping, not {type(raw).__name__}.")
    return _validate_and_build(raw, config_path)


def _require(raw: dict[str, Any], key: str, path: Path) -> Any:
    if key not in raw or raw[key] is None or raw[key] == "":
        raise ConfigError(f"Missing required field '{key}' in {path}.")
    return raw[key]


def _validate_and_build(raw: dict[str, Any], path: Path) -> PortfolioConfig:
    name = str(_require(raw, "portfolio_name", path))
    base_currency = str(_require(raw, "base_currency", path))
    total_capital = _require(raw, "total_capital", path)
    try:
        total_capital_f = float(total_capital)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"'total_capital' must be numeric in {path}.") from exc

    benchmark = str(_require(raw, "benchmark", path)).strip()
    if not benchmark:
        raise ConfigError(f"'benchmark' ticker is missing in {path}.")

    positions_raw = _require(raw, "positions", path)
    if not isinstance(positions_raw, list) or len(positions_raw) == 0:
        raise ConfigError(f"'positions' must be a non-empty list in {path}.")

    positions: list[Position] = []
    seen_tickers: set[str] = set()
    for i, item in enumerate(positions_raw):
        loc = f"{path} positions[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{loc} must be a mapping with ticker, sleeve, and weight.")
        for field in ("ticker", "sleeve", "weight"):
            if field not in item or item[field] is None or item[field] == "":
                raise ConfigError(f"{loc} is missing required field '{field}'.")
        ticker = str(item["ticker"]).strip()
        sleeve = str(item["sleeve"]).strip()
        if ticker in seen_tickers:
            raise ConfigError(f"Duplicate ticker '{ticker}' in {path}.")
        seen_tickers.add(ticker)
        if sleeve not in ALLOWED_SLEEVES:
            allowed = ", ".join(sorted(ALLOWED_SLEEVES))
            raise ConfigError(
                f"{loc} has invalid sleeve '{sleeve}' for {ticker}. Allowed sleeves: {allowed}."
            )
        try:
            weight = float(item["weight"])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{loc} weight for {ticker} must be numeric.") from exc
        rationale = str(item.get("rationale") or "")
        positions.append(Position(ticker=ticker, sleeve=sleeve, weight=weight, rationale=rationale))

    weight_sum = sum(p.weight for p in positions)
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ConfigError(
            f"Position weights must sum to 1.0 (+/-{WEIGHT_SUM_TOLERANCE}); "
            f"got {weight_sum:.6f} in {path}."
        )

    return PortfolioConfig(
        portfolio_name=name,
        base_currency=base_currency,
        total_capital=total_capital_f,
        benchmark=benchmark,
        positions=tuple(positions),
        source_path=path,
    )
