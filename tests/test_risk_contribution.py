"""Euler vol contributions sum to portfolio volatility."""

from __future__ import annotations

import numpy as np

from src.data_cleaning import clean_prices
from src.risk_contribution import CONTRIB_ATOL, compute_risk_contribution


def test_component_vols_sum_to_portfolio_vol(
    valid_config, valid_risk_config, synthetic_prices, tmp_path
) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    result = compute_risk_contribution(cleaned.returns, valid_config, valid_risk_config, save=False)
    pos = result.table[result.table["level"] == "position"]
    sleeve = result.table[result.table["level"] == "sleeve"]
    assert abs(float(pos["vol_contribution"].sum()) - result.portfolio_vol) < CONTRIB_ATOL
    assert abs(float(sleeve["vol_contribution"].sum()) - result.portfolio_vol) < CONTRIB_ATOL


def test_outsized_flag_uses_config_ratio(valid_config, valid_risk_config, synthetic_prices, tmp_path) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    result = compute_risk_contribution(cleaned.returns, valid_config, valid_risk_config, save=False)
    pos = result.table[result.table["level"] == "position"]
    for _, row in pos.iterrows():
        expected = bool(row["share_ratio"] > valid_risk_config.outsized_share_ratio)
        assert bool(row["outsized_risk_flag"]) is expected
    assert np.isfinite(result.portfolio_vol)
