"""Load and validate the user-editable portfolio YAML.

No instrument tickers or weights live in code. Invalid configs raise ConfigError
with a message that names the failing rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.paths import DEFAULT_CONFIG_PATH, DEFAULT_RISK_CONFIG_PATH, DEFAULT_THRESHOLDS_PATH

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


@dataclass(frozen=True)
class SleeveExposure:
    duration_years: float = 0.0
    spread_duration_years: float = 0.0


@dataclass(frozen=True)
class HistoricalScenarioSpec:
    name: str
    start: str
    end: str


@dataclass(frozen=True)
class VarSettings:
    confidence_levels: tuple[float, ...]
    lookback_days: int
    horizon_days: int
    method: str


@dataclass(frozen=True)
class PerformanceSettings:
    risk_free_rate_annual: float
    trading_days_per_year: int
    rolling_vol_window: int


@dataclass(frozen=True)
class HypotheticalShocks:
    rate_shock_bp: float
    credit_spread_widening_bp: float
    fx_depreciation_pct: float
    alternatives_drawdown_pct: float
    combined_severe: bool


@dataclass(frozen=True)
class RiskConfig:
    var: VarSettings
    performance: PerformanceSettings
    outsized_share_ratio: float
    exposure_assumptions: dict[str, SleeveExposure]
    historical_scenarios: tuple[HistoricalScenarioSpec, ...]
    hypothetical_shocks: HypotheticalShocks
    source_path: Path


def load_risk_config(
    path: Path | None = None,
    *,
    portfolio_sleeves: list[str] | None = None,
) -> RiskConfig:
    """Parse ``config/risk.yaml`` and validate assumptions used by the risk engine."""
    config_path = Path(path) if path is not None else DEFAULT_RISK_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(
            f"Risk config not found: {config_path}. "
            "Create config/risk.yaml with VaR, performance, exposure, and scenario blocks."
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping, not {type(raw).__name__}.")
    return _validate_risk_config(raw, config_path, portfolio_sleeves)


def _as_float(value: Any, loc: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{loc} must be numeric; got {value!r}.") from exc


def _as_int(value: Any, loc: str) -> int:
    try:
        as_int = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{loc} must be an integer; got {value!r}.") from exc
    if float(value) != as_int:
        raise ConfigError(f"{loc} must be an integer; got {value!r}.")
    return as_int


def _validate_risk_config(
    raw: dict[str, Any],
    path: Path,
    portfolio_sleeves: list[str] | None,
) -> RiskConfig:
    var_raw = _require(raw, "var", path)
    if not isinstance(var_raw, dict):
        raise ConfigError(f"'var' must be a mapping in {path}.")
    levels_raw = _require(var_raw, "confidence_levels", path)
    if not isinstance(levels_raw, list) or not levels_raw:
        raise ConfigError(f"'var.confidence_levels' must be a non-empty list in {path}.")
    levels: list[float] = []
    for i, item in enumerate(levels_raw):
        level = _as_float(item, f"{path} var.confidence_levels[{i}]")
        if not 0.0 < level < 1.0:
            raise ConfigError(
                f"Confidence level {level} is invalid in {path}. "
                "Each level must be strictly between 0 and 1 (exclusive)."
            )
        levels.append(level)
    lookback = _as_int(_require(var_raw, "lookback_days", path), f"{path} var.lookback_days")
    horizon = _as_int(_require(var_raw, "horizon_days", path), f"{path} var.horizon_days")
    if lookback < 2:
        raise ConfigError(f"'var.lookback_days' must be >= 2 in {path}.")
    if horizon < 1:
        raise ConfigError(f"'var.horizon_days' must be >= 1 in {path}.")
    method = str(_require(var_raw, "method", path)).strip().lower()
    if method not in {"historical", "parametric"}:
        raise ConfigError(
            f"'var.method' must be 'historical' or 'parametric' in {path}; got {method!r}."
        )

    perf_raw = _require(raw, "performance", path)
    if not isinstance(perf_raw, dict):
        raise ConfigError(f"'performance' must be a mapping in {path}.")
    rf = _as_float(
        _require(perf_raw, "risk_free_rate_annual", path),
        f"{path} performance.risk_free_rate_annual",
    )
    tdays = _as_int(
        _require(perf_raw, "trading_days_per_year", path),
        f"{path} performance.trading_days_per_year",
    )
    roll = _as_int(
        _require(perf_raw, "rolling_vol_window", path),
        f"{path} performance.rolling_vol_window",
    )
    if tdays < 1 or roll < 2:
        raise ConfigError(
            f"performance.trading_days_per_year must be >= 1 and "
            f"rolling_vol_window >= 2 in {path}."
        )

    contrib_raw = raw.get("risk_contribution") or {}
    if contrib_raw and not isinstance(contrib_raw, dict):
        raise ConfigError(f"'risk_contribution' must be a mapping in {path}.")
    outsized = _as_float(
        contrib_raw.get("outsized_share_ratio", 1.5) if isinstance(contrib_raw, dict) else 1.5,
        f"{path} risk_contribution.outsized_share_ratio",
    )
    if outsized <= 0:
        raise ConfigError(f"'risk_contribution.outsized_share_ratio' must be > 0 in {path}.")

    exp_raw = _require(raw, "exposure_assumptions", path)
    if not isinstance(exp_raw, dict) or not exp_raw:
        raise ConfigError(f"'exposure_assumptions' must be a non-empty mapping in {path}.")
    exposures: dict[str, SleeveExposure] = {}
    for sleeve, spec in exp_raw.items():
        loc = f"{path} exposure_assumptions.{sleeve}"
        if not isinstance(spec, dict):
            raise ConfigError(f"{loc} must be a mapping of duration fields.")
        duration = _as_float(spec.get("duration_years", 0.0), f"{loc}.duration_years")
        spread = _as_float(spec.get("spread_duration_years", 0.0), f"{loc}.spread_duration_years")
        exposures[str(sleeve)] = SleeveExposure(
            duration_years=duration, spread_duration_years=spread
        )

    required_sleeves = list(portfolio_sleeves) if portfolio_sleeves is not None else []
    missing = [s for s in required_sleeves if s not in exposures]
    if missing:
        raise ConfigError(
            f"Missing duration assumptions for sleeve(s) {missing} in {path}. "
            "Every sleeve present in portfolio.yaml must have an exposure_assumptions entry."
        )

    scenarios_raw = raw.get("historical_scenarios") or []
    if not isinstance(scenarios_raw, list):
        raise ConfigError(f"'historical_scenarios' must be a list in {path}.")
    scenarios: list[HistoricalScenarioSpec] = []
    for i, item in enumerate(scenarios_raw):
        loc = f"{path} historical_scenarios[{i}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{loc} must be a mapping with name, start, and end.")
        for field in ("name", "start", "end"):
            if field not in item or item[field] is None or item[field] == "":
                raise ConfigError(f"{loc} is missing required field '{field}'.")
        scenarios.append(
            HistoricalScenarioSpec(
                name=str(item["name"]),
                start=str(item["start"]),
                end=str(item["end"]),
            )
        )

    hyp_raw = _require(raw, "hypothetical_shocks", path)
    if not isinstance(hyp_raw, dict):
        raise ConfigError(f"'hypothetical_shocks' must be a mapping in {path}.")
    shocks = HypotheticalShocks(
        rate_shock_bp=_as_float(
            _require(hyp_raw, "rate_shock_bp", path), f"{path} hypothetical_shocks.rate_shock_bp"
        ),
        credit_spread_widening_bp=_as_float(
            _require(hyp_raw, "credit_spread_widening_bp", path),
            f"{path} hypothetical_shocks.credit_spread_widening_bp",
        ),
        fx_depreciation_pct=_as_float(
            _require(hyp_raw, "fx_depreciation_pct", path),
            f"{path} hypothetical_shocks.fx_depreciation_pct",
        ),
        alternatives_drawdown_pct=_as_float(
            _require(hyp_raw, "alternatives_drawdown_pct", path),
            f"{path} hypothetical_shocks.alternatives_drawdown_pct",
        ),
        combined_severe=bool(_require(hyp_raw, "combined_severe", path)),
    )

    return RiskConfig(
        var=VarSettings(
            confidence_levels=tuple(levels),
            lookback_days=lookback,
            horizon_days=horizon,
            method=method,
        ),
        performance=PerformanceSettings(
            risk_free_rate_annual=rf,
            trading_days_per_year=tdays,
            rolling_vol_window=roll,
        ),
        outsized_share_ratio=outsized,
        exposure_assumptions=exposures,
        historical_scenarios=tuple(scenarios),
        hypothetical_shocks=shocks,
        source_path=path,
    )


ALLOWED_STATUSES = frozenset({"green", "amber", "red"})
CANONICAL_STATUS_RULE = "red if any metric is red; else amber if any is amber; else green"
REQUIRED_THRESHOLD_METRICS = (
    "annualized_volatility_pct",
    "max_drawdown_pct",
    "var_99_1d_pct",
    "es_99_1d_pct",
    "single_position_weight_pct",
    "combined_scenario_loss_pct",
)


@dataclass(frozen=True)
class MetricLimit:
    metric: str
    amber: float
    red: float
    escalation_text: str


@dataclass(frozen=True)
class ThresholdsConfig:
    limits: dict[str, MetricLimit]
    overall_status_rule: str
    assumptions_and_limitations: tuple[str, ...]
    source_path: Path


def load_thresholds_config(path: Path | None = None) -> ThresholdsConfig:
    """Parse ``config/thresholds.yaml`` and validate alert limits."""
    config_path = Path(path) if path is not None else DEFAULT_THRESHOLDS_PATH
    if not config_path.is_file():
        raise ConfigError(
            f"Thresholds config not found: {config_path}. "
            "Create config/thresholds.yaml with limits and escalation_text per metric."
        )
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping, not {type(raw).__name__}.")
    return _validate_thresholds_config(raw, config_path)


def _normalize_rule(text: str) -> str:
    return " ".join(text.lower().split())


def _validate_thresholds_config(raw: dict[str, Any], path: Path) -> ThresholdsConfig:
    limits_raw = _require(raw, "limits", path)
    if not isinstance(limits_raw, dict) or not limits_raw:
        raise ConfigError(f"'limits' must be a non-empty mapping in {path}.")

    missing = [m for m in REQUIRED_THRESHOLD_METRICS if m not in limits_raw]
    if missing:
        raise ConfigError(
            f"Missing required metric limit(s) {missing} in {path}. "
            f"Required: {list(REQUIRED_THRESHOLD_METRICS)}."
        )

    limits: dict[str, MetricLimit] = {}
    for metric, spec in limits_raw.items():
        loc = f"{path} limits.{metric}"
        if not isinstance(spec, dict):
            raise ConfigError(f"{loc} must be a mapping with amber, red, and escalation_text.")
        for field_name in ("amber", "red", "escalation_text"):
            if field_name not in spec or spec[field_name] is None or spec[field_name] == "":
                raise ConfigError(f"{loc} is missing required field '{field_name}'.")
        amber = _as_float(spec["amber"], f"{loc}.amber")
        red = _as_float(spec["red"], f"{loc}.red")
        if red <= amber:
            raise ConfigError(
                f"{loc}: red ({red}) must be greater than amber ({amber}) for loss-type metrics."
            )
        text = str(spec["escalation_text"]).strip()
        if not text:
            raise ConfigError(f"{loc}.escalation_text must be non-empty.")
        limits[str(metric)] = MetricLimit(
            metric=str(metric), amber=amber, red=red, escalation_text=text
        )

    rule = str(_require(raw, "overall_status_rule", path)).strip()
    if _normalize_rule(rule) != _normalize_rule(CANONICAL_STATUS_RULE):
        raise ConfigError(
            f"'overall_status_rule' in {path} is not a supported rule. "
            f"Use exactly: {CANONICAL_STATUS_RULE!r}."
        )

    assumptions_raw = raw.get("assumptions_and_limitations") or []
    if not isinstance(assumptions_raw, list):
        raise ConfigError(f"'assumptions_and_limitations' must be a list in {path}.")
    assumptions = tuple(str(item).strip() for item in assumptions_raw if str(item).strip())

    return ThresholdsConfig(
        limits=limits,
        overall_status_rule=rule,
        assumptions_and_limitations=assumptions,
        source_path=path,
    )
