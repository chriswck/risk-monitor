"""Config loader rejects invalid YAML and accepts a well-formed portfolio."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config_loader import ConfigError, load_portfolio_config
from tests.helpers import write_portfolio_yaml


def test_checked_in_portfolio_yaml_is_valid() -> None:
    cfg = load_portfolio_config()
    assert abs(cfg.weight_sum() - 1.0) <= 0.001
    assert cfg.positions
    assert cfg.benchmark


def test_valid_config_loads(valid_config) -> None:
    assert abs(valid_config.weight_sum() - 1.0) <= 0.001
    assert valid_config.benchmark == "BENCH"
    assert valid_config.tickers == ["AAA", "BBB", "CCC"]
    assert "BENCH" in valid_config.download_tickers


def test_weights_must_sum_to_one(tmp_path: Path, valid_portfolio_payload: dict) -> None:
    valid_portfolio_payload["positions"][0]["weight"] = 0.99
    path = write_portfolio_yaml(tmp_path / "bad.yaml", valid_portfolio_payload)
    with pytest.raises(ConfigError, match="sum to 1.0"):
        load_portfolio_config(path)


def test_missing_position_field(tmp_path: Path, valid_portfolio_payload: dict) -> None:
    del valid_portfolio_payload["positions"][1]["sleeve"]
    path = write_portfolio_yaml(tmp_path / "bad.yaml", valid_portfolio_payload)
    with pytest.raises(ConfigError, match="sleeve"):
        load_portfolio_config(path)


def test_invalid_sleeve(tmp_path: Path, valid_portfolio_payload: dict) -> None:
    valid_portfolio_payload["positions"][0]["sleeve"] = "equity"
    path = write_portfolio_yaml(tmp_path / "bad.yaml", valid_portfolio_payload)
    with pytest.raises(ConfigError, match="invalid sleeve"):
        load_portfolio_config(path)


def test_missing_benchmark(tmp_path: Path, valid_portfolio_payload: dict) -> None:
    del valid_portfolio_payload["benchmark"]
    path = write_portfolio_yaml(tmp_path / "bad.yaml", valid_portfolio_payload)
    with pytest.raises(ConfigError, match="benchmark"):
        load_portfolio_config(path)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_portfolio_config(tmp_path / "nope.yaml")
