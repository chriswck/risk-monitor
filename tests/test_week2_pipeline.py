"""Week 2 engine writes the four processed CSVs from in-memory Week 1 outputs."""

from __future__ import annotations

from src.config_loader import load_portfolio_config
from src.data_cleaning import clean_prices
from src.pipeline import run_week2_pipeline
from src.portfolio import compute_portfolio_returns
from tests.helpers import write_portfolio_yaml, write_risk_yaml


def test_week2_writes_output_csvs(
    tmp_path,
    valid_portfolio_payload,
    valid_risk_payload,
    synthetic_prices,
) -> None:
    port_path = write_portfolio_yaml(tmp_path / "portfolio.yaml", valid_portfolio_payload)
    risk_path = write_risk_yaml(tmp_path / "risk.yaml", valid_risk_payload)
    dest = tmp_path / "processed"
    dest.mkdir()
    cleaned = clean_prices(synthetic_prices, processed_dir=dest, save=False)
    cfg = load_portfolio_config(port_path)
    port = compute_portfolio_returns(cleaned.returns, cfg, save=False)
    result = run_week2_pipeline(
        portfolio_path=port_path,
        risk_path=risk_path,
        asset_returns=cleaned.returns,
        portfolio_returns=port.returns,
        processed_dir=dest,
        save=True,
    )
    assert result.metrics.summary_path is not None and result.metrics.summary_path.is_file()
    assert result.metrics.correlation_path is not None and result.metrics.correlation_path.is_file()
    assert result.contribution.output_path is not None and result.contribution.output_path.is_file()
    assert result.scenarios.output_path is not None and result.scenarios.output_path.is_file()
    assert result.metrics.summary_path.name.startswith("risk_metrics_summary_")
    assert result.metrics.correlation_path.name.startswith("correlation_matrix_")
    assert result.contribution.output_path.name.startswith("risk_contribution_")
    assert result.scenarios.output_path.name.startswith("scenario_results_")
