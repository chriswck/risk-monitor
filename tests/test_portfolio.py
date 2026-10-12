"""Static-weight portfolio returns match the weighted-sum identity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_cleaning import clean_prices
from src.portfolio import compute_portfolio_returns


def test_portfolio_return_equals_weighted_sum(valid_config, synthetic_prices, tmp_path) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    result = compute_portfolio_returns(cleaned.returns, valid_config, processed_dir=tmp_path, save=True)
    weights = pd.Series(valid_config.weights)
    expected = cleaned.returns[valid_config.tickers].mul(weights, axis=1).sum(axis=1)
    err = (result.returns["portfolio_return"] - expected).abs().max()
    assert err < 1e-10
    assert result.output_path is not None
    assert result.output_path.is_file()


def test_sleeve_contributions_sum_to_portfolio(valid_config, synthetic_prices, tmp_path) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    result = compute_portfolio_returns(cleaned.returns, valid_config, save=False)
    contrib_cols = [c for c in result.returns.columns if c.endswith("_contribution")]
    sleeve_sum = result.returns[contrib_cols].sum(axis=1)
    err = (sleeve_sum - result.returns["portfolio_return"]).abs().max()
    assert err < 1e-10


def test_cumulative_series_compound(valid_config, synthetic_prices, tmp_path) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    result = compute_portfolio_returns(cleaned.returns, valid_config, save=False)
    reconstructed = (1.0 + result.returns["portfolio_return"]).cumprod()
    np.testing.assert_allclose(
        result.returns["cumulative_portfolio"].to_numpy(),
        reconstructed.to_numpy(),
        rtol=0,
        atol=1e-12,
    )
