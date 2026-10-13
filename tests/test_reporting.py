"""Report files contain engine numbers, not invented commentary."""

from __future__ import annotations

from src.config_loader import load_portfolio_config
from src.data_cleaning import clean_prices
from src.pipeline import run_week2_pipeline
from src.portfolio import compute_portfolio_returns
from src.reporting import USER_COMMENTARY_EXEC, USER_COMMENTARY_RISKS, write_report
from src.scenarios import APPROXIMATION_DISCLAIMER
from tests.helpers import write_portfolio_yaml, write_risk_yaml


def test_report_contains_computed_values_and_placeholders(
    tmp_path,
    valid_portfolio_payload,
    valid_risk_payload,
    valid_thresholds_config,
    synthetic_prices,
) -> None:
    port_path = write_portfolio_yaml(tmp_path / "portfolio.yaml", valid_portfolio_payload)
    risk_path = write_risk_yaml(tmp_path / "risk.yaml", valid_risk_payload)
    cfg = load_portfolio_config(port_path)
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    port = compute_portfolio_returns(cleaned.returns, cfg, save=False)
    week2 = run_week2_pipeline(
        portfolio_path=port_path,
        risk_path=risk_path,
        asset_returns=cleaned.returns,
        portfolio_returns=port.returns,
        processed_dir=tmp_path,
        save=False,
    )
    vol = float(
        week2.metrics.summary.loc[
            week2.metrics.summary["metric"] == "annualized_volatility", "value"
        ].iloc[0]
    )
    paths = write_report(
        week2,
        thresholds=valid_thresholds_config,
        reports_dir=tmp_path / "reports",
    )
    assert paths.markdown.is_file()
    assert paths.html.is_file()
    text = paths.markdown.read_text(encoding="utf-8")
    assert f"{vol:.4%}" in text
    assert USER_COMMENTARY_EXEC in text
    assert USER_COMMENTARY_RISKS in text
    assert APPROXIMATION_DISCLAIMER in text
    assert "[USER COMMENTARY:" in text
