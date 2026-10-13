"""Duration-shock algebra, disclaimer, and out-of-sample historical windows."""

from __future__ import annotations

from src.config_loader import SleeveExposure
from src.data_cleaning import clean_prices
from src.portfolio import compute_portfolio_returns
from src.scenarios import (
    APPROXIMATION_DISCLAIMER,
    duration_shock_pnl,
    hypothetical_sleeve_pnl,
    run_scenarios,
)


def test_duration_shock_toy_sleeve() -> None:
    """duration 7, +100bp, $25m sleeve -> -$1.75m."""
    pnl = duration_shock_pnl(duration_years=7.0, shock_bp=100.0, sleeve_value=25_000_000.0)
    assert abs(pnl - (-1_750_000.0)) < 1e-8


def test_credit_shock_uses_spread_duration(valid_risk_config) -> None:
    exposure = SleeveExposure(duration_years=8.0, spread_duration_years=7.0)
    pnl = hypothetical_sleeve_pnl(
        "ig_credit",
        25_000_000.0,
        exposure,
        valid_risk_config,
        apply_rate=False,
        apply_credit=True,
        apply_fx=False,
        apply_alt=False,
    )
    # -7.0 * 100/10000 * 25e6 = -1.75e6
    assert abs(pnl - (-1_750_000.0)) < 1e-8


def test_every_scenario_row_has_disclaimer(
    valid_config, valid_risk_config, synthetic_prices, tmp_path
) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    port = compute_portfolio_returns(cleaned.returns, valid_config, save=False)
    result = run_scenarios(port.returns, valid_config, valid_risk_config, save=False)
    assert not result.table.empty
    assert result.table["disclaimer"].eq(APPROXIMATION_DISCLAIMER).all()
    assert any("outside available data" in w for w in result.warnings)


def test_fx_and_alt_percent_shocks(valid_config, valid_risk_config) -> None:
    capital = valid_config.total_capital
    fx_weight = valid_config.sleeve_weights.get("fx", 0.0)
    # fixture has no fx sleeve; alternatives is 0.25 of 1000 = 250
    alt_value = capital * valid_config.sleeve_weights["alternatives"]
    pnl = hypothetical_sleeve_pnl(
        "alternatives",
        alt_value,
        valid_risk_config.exposure_assumptions["alternatives"],
        valid_risk_config,
        apply_rate=False,
        apply_credit=False,
        apply_fx=False,
        apply_alt=True,
    )
    expected = -(valid_risk_config.hypothetical_shocks.alternatives_drawdown_pct / 100.0) * alt_value
    assert abs(pnl - expected) < 1e-10
    assert fx_weight == 0.0
