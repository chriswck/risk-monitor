"""Shared fixtures: synthetic prices and a temporary valid portfolio YAML."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from src.config_loader import load_portfolio_config
from tests.helpers import write_portfolio_yaml


@pytest.fixture
def valid_portfolio_payload() -> dict:
    return {
        "portfolio_name": "Test Portfolio",
        "base_currency": "USD",
        "total_capital": 1000,
        "benchmark": "BENCH",
        "positions": [
            {"ticker": "AAA", "sleeve": "rates", "weight": 0.40, "rationale": "test rates"},
            {"ticker": "BBB", "sleeve": "ig_credit", "weight": 0.35, "rationale": "test ig"},
            {"ticker": "CCC", "sleeve": "alternatives", "weight": 0.25, "rationale": "test alt"},
        ],
    }


@pytest.fixture
def valid_config(tmp_path: Path, valid_portfolio_payload: dict):
    path = write_portfolio_yaml(tmp_path / "portfolio.yaml", valid_portfolio_payload)
    return load_portfolio_config(path)


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """~3 years of business-day prices with a short gap, a long gap, and one extreme day."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2021-01-04", "2023-12-29")
    n = len(idx)
    rets = rng.normal(0.0002, 0.004, size=(n, 4))
    # One extreme daily return on AAA, kept on purpose
    rets[100, 0] = 0.15
    levels = 100 * np.exp(np.cumsum(np.log1p(rets), axis=0))
    prices = pd.DataFrame(levels, index=idx, columns=["AAA", "BBB", "CCC", "BENCH"])
    # Short gap (<=3) in BBB
    prices.loc[idx[20:22], "BBB"] = np.nan
    # Long gap (>3) in CCC
    prices.loc[idx[50:56], "CCC"] = np.nan
    return prices
