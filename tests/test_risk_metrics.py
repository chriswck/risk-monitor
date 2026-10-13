"""VaR/ES, correlation, and lookback-warning behaviour."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_cleaning import clean_prices
from src.portfolio import compute_portfolio_returns
from src.risk_metrics import compute_risk_metrics, historical_var_es


def test_es_at_least_var_loss_positive(valid_config, valid_risk_config, synthetic_prices, tmp_path) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    port = compute_portfolio_returns(cleaned.returns, valid_config, save=False)
    result = compute_risk_metrics(
        cleaned.returns, port.returns, valid_config, valid_risk_config, save=False
    )
    for est in result.var_es:
        assert est.historical_es >= est.historical_var - 1e-12
        assert est.parametric_es >= est.parametric_var - 1e-12
        assert est.historical_var > 0 or est.historical_es >= 0


def test_es_hand_computable_toy_series() -> None:
    """n=4, conf=0.75, linear quantile.

    returns = [-0.04, -0.02, 0.01, 0.03]
    losses  = [ 0.04,  0.02,-0.01,-0.03]
    sorted losses = [-0.03, -0.01, 0.02, 0.04]
    index = (4-1)*0.75 = 2.25 -> VaR = 0.02 + 0.25*(0.04-0.02) = 0.025
    ES = mean{losses >= 0.025} = 0.04
    """
    returns = np.array([-0.04, -0.02, 0.01, 0.03])
    losses = -returns
    var, es = historical_var_es(losses, 0.75)
    assert abs(var - 0.025) < 1e-12
    assert abs(es - 0.04) < 1e-12
    assert es >= var


def test_correlation_symmetric_unit_diagonal(
    valid_config, valid_risk_config, synthetic_prices, tmp_path
) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    port = compute_portfolio_returns(cleaned.returns, valid_config, save=False)
    result = compute_risk_metrics(
        cleaned.returns, port.returns, valid_config, valid_risk_config, save=False
    )
    corr = result.correlation
    np.testing.assert_allclose(corr.to_numpy(), corr.T.to_numpy(), atol=1e-12)
    np.testing.assert_allclose(np.diag(corr.to_numpy()), np.ones(len(corr)), atol=1e-12)


def test_short_lookback_warns_not_crash(
    valid_config, valid_risk_config, synthetic_prices, tmp_path
) -> None:
    from dataclasses import replace

    short = replace(valid_risk_config.var, lookback_days=10_000)
    risk = replace(valid_risk_config, var=short)
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    port = compute_portfolio_returns(cleaned.returns, valid_config, save=False)
    result = compute_risk_metrics(cleaned.returns, port.returns, valid_config, risk, save=False)
    assert result.warnings
    assert any("lookback_days" in w for w in result.warnings)
