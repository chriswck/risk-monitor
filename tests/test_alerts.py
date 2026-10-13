"""Traffic-light status, overall rule, and escalation text."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.alerts import classify_status, evaluate_alerts, overall_status
from src.config_loader import ConfigError, load_thresholds_config
from src.data_cleaning import clean_prices
from src.pipeline import run_week2_pipeline
from src.portfolio import compute_portfolio_returns
from tests.helpers import write_portfolio_yaml, write_risk_yaml, write_thresholds_yaml


def test_classify_status_boundaries() -> None:
    amber, red = 8.0, 12.0
    assert classify_status(7.999, amber, red) == "green"
    assert classify_status(8.0, amber, red) == "amber"
    assert classify_status(11.999, amber, red) == "amber"
    assert classify_status(12.0, amber, red) == "red"
    assert classify_status(20.0, amber, red) == "red"


def test_overall_status_rule() -> None:
    assert overall_status(["green", "green"]) == "green"
    assert overall_status(["green", "amber", "green"]) == "amber"
    assert overall_status(["amber", "red", "green"]) == "red"
    assert overall_status(["red"]) == "red"


def test_invalid_thresholds_red_not_greater_than_amber(
    tmp_path: Path, valid_thresholds_payload: dict
) -> None:
    valid_thresholds_payload["limits"]["max_drawdown_pct"]["red"] = 5.0
    path = write_thresholds_yaml(tmp_path / "bad.yaml", valid_thresholds_payload)
    with pytest.raises(ConfigError, match="must be greater than amber"):
        load_thresholds_config(path)


def test_invalid_thresholds_missing_amber(
    tmp_path: Path, valid_thresholds_payload: dict
) -> None:
    del valid_thresholds_payload["limits"]["var_99_1d_pct"]["amber"]
    path = write_thresholds_yaml(tmp_path / "bad.yaml", valid_thresholds_payload)
    with pytest.raises(ConfigError, match="amber"):
        load_thresholds_config(path)


def test_breaches_have_escalation_text(
    tmp_path,
    valid_portfolio_payload,
    valid_risk_payload,
    valid_thresholds_payload,
    synthetic_prices,
    valid_thresholds_config,
) -> None:
    port_path = write_portfolio_yaml(tmp_path / "portfolio.yaml", valid_portfolio_payload)
    risk_path = write_risk_yaml(tmp_path / "risk.yaml", valid_risk_payload)
    from src.config_loader import load_portfolio_config

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
    report = evaluate_alerts(week2, valid_thresholds_config)
    for row in report.breaches():
        assert row.escalation_text.strip()
    for row in report.rows:
        if row.status == "green":
            assert row.escalation_text == ""
