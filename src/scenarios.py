"""Historical window P&L and first-order hypothetical sleeve shocks."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config_loader import PortfolioConfig, RiskConfig, SleeveExposure
from src.data_cleaning import _unique_processed_path
from src.paths import PROCESSED_DIR

LOGGER = logging.getLogger(__name__)

APPROXIMATION_DISCLAIMER = (
    "First-order approximation - ignores convexity, correlation shifts, and second-round effects"
)


@dataclass
class ScenarioResults:
    table: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    output_path: Path | None = None


def duration_shock_pnl(duration_years: float, shock_bp: float, sleeve_value: float) -> float:
    """P&L ≈ -duration x (shock_bp / 10000) x sleeve_value."""
    return float(-duration_years * (shock_bp / 10000.0) * sleeve_value)


def percent_shock_pnl(shock_pct: float, sleeve_value: float) -> float:
    """P&L ≈ -(shock_pct / 100) x sleeve_value (shock_pct is in percent points, e.g. 5.0 = 5%)."""
    return float(-(shock_pct / 100.0) * sleeve_value)


def _worst_hit(sleeve_pnl: dict[str, float]) -> str:
    if not sleeve_pnl:
        return ""
    return min(sleeve_pnl, key=sleeve_pnl.get)


def _compound_pnl(returns: pd.Series) -> float:
    if returns.empty:
        return float("nan")
    return float((1.0 + returns.astype(float)).prod() - 1.0)


def historical_scenario_rows(
    portfolio_returns: pd.DataFrame,
    portfolio: PortfolioConfig,
    risk: RiskConfig,
) -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    warnings: list[str] = []
    idx = portfolio_returns.index
    data_start, data_end = idx.min(), idx.max()
    capital = portfolio.total_capital

    for spec in risk.historical_scenarios:
        start = pd.Timestamp(spec.start)
        end = pd.Timestamp(spec.end)
        if end < start:
            msg = f"Historical scenario '{spec.name}' has end before start; skipped."
            LOGGER.warning(msg)
            warnings.append(msg)
            continue
        if end < data_start or start > data_end:
            msg = (
                f"Historical scenario '{spec.name}' ({spec.start} to {spec.end}) "
                f"is outside available data ({data_start.date()} to {data_end.date()})."
            )
            LOGGER.warning(msg)
            warnings.append(msg)
            rows.append(
                {
                    "scenario": spec.name,
                    "type": "historical",
                    "pnl_pct": float("nan"),
                    "pnl_currency": float("nan"),
                    "worst_hit_sleeve": "",
                    "notes": msg,
                    "disclaimer": APPROXIMATION_DISCLAIMER,
                }
            )
            continue

        clipped_start = max(start, pd.Timestamp(data_start))
        clipped_end = min(end, pd.Timestamp(data_end))
        note = ""
        if clipped_start != start or clipped_end != end:
            note = (
                f"Window clipped to available data "
                f"({clipped_start.date()} to {clipped_end.date()}); configured "
                f"{spec.start} to {spec.end}."
            )
            LOGGER.warning("%s %s", spec.name, note)
            warnings.append(f"{spec.name}: {note}")

        window = portfolio_returns.loc[clipped_start:clipped_end]
        if window.empty:
            msg = f"Historical scenario '{spec.name}' has no overlapping return rows."
            LOGGER.warning(msg)
            warnings.append(msg)
            rows.append(
                {
                    "scenario": spec.name,
                    "type": "historical",
                    "pnl_pct": float("nan"),
                    "pnl_currency": float("nan"),
                    "worst_hit_sleeve": "",
                    "notes": msg,
                    "disclaimer": APPROXIMATION_DISCLAIMER,
                }
            )
            continue

        pnl_pct = _compound_pnl(window["portfolio_return"])
        pnl_ccy = pnl_pct * capital
        sleeve_pnl: dict[str, float] = {}
        for sleeve, sleeve_weight in portfolio.sleeve_weights.items():
            col = f"sleeve_{sleeve}_return"
            if col not in window.columns:
                continue
            sleeve_pct = _compound_pnl(window[col])
            sleeve_pnl[sleeve] = sleeve_pct * sleeve_weight * capital
        worst = _worst_hit(sleeve_pnl)
        rows.append(
            {
                "scenario": spec.name,
                "type": "historical",
                "pnl_pct": pnl_pct,
                "pnl_currency": pnl_ccy,
                "worst_hit_sleeve": worst,
                "notes": note or f"{len(window)} days; {clipped_start.date()} to {clipped_end.date()}",
                "disclaimer": APPROXIMATION_DISCLAIMER,
            }
        )
    return rows, warnings


def hypothetical_sleeve_pnl(
    sleeve: str,
    sleeve_value: float,
    exposure: SleeveExposure,
    risk: RiskConfig,
    *,
    apply_rate: bool,
    apply_credit: bool,
    apply_fx: bool,
    apply_alt: bool,
) -> float:
    shocks = risk.hypothetical_shocks
    pnl = 0.0
    if apply_rate:
        pnl += duration_shock_pnl(exposure.duration_years, shocks.rate_shock_bp, sleeve_value)
    if apply_credit:
        pnl += duration_shock_pnl(
            exposure.spread_duration_years, shocks.credit_spread_widening_bp, sleeve_value
        )
    if apply_fx and sleeve == "fx":
        pnl += percent_shock_pnl(shocks.fx_depreciation_pct, sleeve_value)
    if apply_alt and sleeve == "alternatives":
        pnl += percent_shock_pnl(shocks.alternatives_drawdown_pct, sleeve_value)
    return pnl


def hypothetical_scenario_rows(
    portfolio: PortfolioConfig,
    risk: RiskConfig,
) -> list[dict[str, object]]:
    capital = portfolio.total_capital
    shocks = risk.hypothetical_shocks
    specs: list[tuple[str, bool, bool, bool, bool]] = [
        (f"Parallel rates +{shocks.rate_shock_bp:.0f}bp", True, False, False, False),
        (
            f"Credit spread +{shocks.credit_spread_widening_bp:.0f}bp",
            False,
            True,
            False,
            False,
        ),
        (f"FX depreciation {shocks.fx_depreciation_pct:.1f}%", False, False, True, False),
        (
            f"Alternatives drawdown {shocks.alternatives_drawdown_pct:.1f}%",
            False,
            False,
            False,
            True,
        ),
    ]
    if shocks.combined_severe:
        specs.append(("Combined severe", True, True, True, True))

    rows: list[dict[str, object]] = []
    for name, use_rate, use_credit, use_fx, use_alt in specs:
        sleeve_pnl: dict[str, float] = {}
        total = 0.0
        for sleeve, weight in portfolio.sleeve_weights.items():
            exposure = risk.exposure_assumptions.get(sleeve, SleeveExposure())
            value = capital * weight
            pnl = hypothetical_sleeve_pnl(
                sleeve,
                value,
                exposure,
                risk,
                apply_rate=use_rate,
                apply_credit=use_credit,
                apply_fx=use_fx,
                apply_alt=use_alt,
            )
            sleeve_pnl[sleeve] = pnl
            total += pnl
        pnl_pct = total / capital if capital else float("nan")
        kind = "combined" if name == "Combined severe" else "hypothetical"
        rows.append(
            {
                "scenario": name,
                "type": kind,
                "pnl_pct": pnl_pct,
                "pnl_currency": total,
                "worst_hit_sleeve": _worst_hit(sleeve_pnl),
                "notes": "first-order sleeve shocks from config/risk.yaml",
                "disclaimer": APPROXIMATION_DISCLAIMER,
            }
        )
    return rows


def run_scenarios(
    portfolio_returns: pd.DataFrame,
    portfolio: PortfolioConfig,
    risk: RiskConfig,
    *,
    processed_dir: Path | None = None,
    as_of: datetime | None = None,
    save: bool = True,
) -> ScenarioResults:
    """Build the historical + hypothetical scenario summary table."""
    hist_rows, warnings = historical_scenario_rows(portfolio_returns, portfolio, risk)
    hyp_rows = hypothetical_scenario_rows(portfolio, risk)
    table = pd.DataFrame(hist_rows + hyp_rows)
    if "disclaimer" in table.columns:
        if not table["disclaimer"].eq(APPROXIMATION_DISCLAIMER).all():
            raise RuntimeError("Every scenario row must carry the first-order approximation disclaimer.")

    output_path = None
    if save:
        dest = processed_dir if processed_dir is not None else PROCESSED_DIR
        dest.mkdir(parents=True, exist_ok=True)
        stamp = (as_of or datetime.now(timezone.utc)).strftime("%Y%m%d")
        output_path = _unique_processed_path(dest, "scenario_results", stamp)
        table.to_csv(output_path, index=False)
        LOGGER.info("Wrote %s.", output_path)

    return ScenarioResults(table=table, warnings=warnings, output_path=output_path)
