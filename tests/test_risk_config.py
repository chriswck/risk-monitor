"""Risk YAML validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config_loader import ConfigError, load_risk_config
from tests.helpers import write_risk_yaml


def test_checked_in_risk_yaml_loads_against_portfolio() -> None:
    from src.config_loader import load_portfolio_config

    portfolio = load_portfolio_config()
    cfg = load_risk_config(portfolio_sleeves=list(portfolio.sleeve_weights.keys()))
    assert cfg.var.confidence_levels
    assert all(0.0 < c < 1.0 for c in cfg.var.confidence_levels)


def test_rejects_invalid_confidence_level(
    tmp_path: Path, valid_risk_payload: dict, valid_config
) -> None:
    valid_risk_payload["var"]["confidence_levels"] = [0.95, 1.05]
    path = write_risk_yaml(tmp_path / "risk.yaml", valid_risk_payload)
    with pytest.raises(ConfigError, match="strictly between 0 and 1"):
        load_risk_config(path, portfolio_sleeves=list(valid_config.sleeve_weights.keys()))


def test_rejects_confidence_at_boundary(
    tmp_path: Path, valid_risk_payload: dict, valid_config
) -> None:
    valid_risk_payload["var"]["confidence_levels"] = [0.0]
    path = write_risk_yaml(tmp_path / "risk.yaml", valid_risk_payload)
    with pytest.raises(ConfigError, match="strictly between 0 and 1"):
        load_risk_config(path, portfolio_sleeves=list(valid_config.sleeve_weights.keys()))


def test_rejects_missing_sleeve_duration(
    tmp_path: Path, valid_risk_payload: dict, valid_config
) -> None:
    del valid_risk_payload["exposure_assumptions"]["ig_credit"]
    path = write_risk_yaml(tmp_path / "risk.yaml", valid_risk_payload)
    with pytest.raises(ConfigError, match="Missing duration assumptions"):
        load_risk_config(path, portfolio_sleeves=list(valid_config.sleeve_weights.keys()))


def test_rejects_non_numeric_shock(tmp_path: Path, valid_risk_payload: dict, valid_config) -> None:
    valid_risk_payload["hypothetical_shocks"]["rate_shock_bp"] = "one hundred"
    path = write_risk_yaml(tmp_path / "risk.yaml", valid_risk_payload)
    with pytest.raises(ConfigError, match="must be numeric"):
        load_risk_config(path, portfolio_sleeves=list(valid_config.sleeve_weights.keys()))
