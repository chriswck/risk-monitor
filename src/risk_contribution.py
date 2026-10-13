"""Euler (Euler-homogeneous) contributions to portfolio volatility.

For weights w and covariance Sigma of asset returns,
    sigma = sqrt(w' Sigma w)
    component_i = w_i * (Sigma w)_i / sigma
and sum_i component_i = sigma.
Figures are annualized with the configured trading-day count.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config_loader import PortfolioConfig, RiskConfig
from src.data_cleaning import _unique_processed_path
from src.paths import PROCESSED_DIR

LOGGER = logging.getLogger(__name__)

CONTRIB_ATOL = 1e-6


@dataclass
class RiskContributionResult:
    table: pd.DataFrame
    portfolio_vol: float
    output_path: Path | None = None


def _annual_cov(returns: pd.DataFrame, trading_days: int) -> pd.DataFrame:
    return returns.cov(ddof=1) * trading_days


def compute_risk_contribution(
    asset_returns: pd.DataFrame,
    portfolio: PortfolioConfig,
    risk: RiskConfig,
    *,
    processed_dir: Path | None = None,
    as_of: datetime | None = None,
    save: bool = True,
) -> RiskContributionResult:
    """Per-position and per-sleeve Euler contributions to annualized portfolio vol."""
    tdays = risk.performance.trading_days_per_year
    tickers = [t for t in portfolio.tickers if t in asset_returns.columns]
    missing = [t for t in portfolio.tickers if t not in asset_returns.columns]
    if missing:
        raise ValueError(f"Missing asset returns for risk contribution: {missing}")

    rets = asset_returns[tickers].astype(float).dropna(how="any")
    weights = pd.Series({t: portfolio.weights[t] for t in tickers}, dtype=float)
    cov = _annual_cov(rets, tdays)
    sigma_w = cov.dot(weights)
    port_var = float(weights.dot(sigma_w))
    if port_var < 0 and abs(port_var) < 1e-12:
        port_var = 0.0
    port_vol = float(np.sqrt(port_var))
    if port_vol == 0.0:
        components = pd.Series(0.0, index=tickers)
    else:
        components = weights * sigma_w / port_vol

    residual = abs(float(components.sum()) - port_vol)
    if residual > CONTRIB_ATOL:
        LOGGER.warning(
            "Euler contributions do not sum to portfolio vol: |sum-c| = %.3e (tol %s).",
            residual,
            CONTRIB_ATOL,
        )

    standalone = rets.std(ddof=1) * np.sqrt(tdays)
    rows: list[dict[str, object]] = []
    ratio_limit = risk.outsized_share_ratio
    for ticker in tickers:
        weight = float(weights[ticker])
        contrib = float(components[ticker])
        c_share = contrib / port_vol if port_vol else 0.0
        w_share = weight
        share_ratio = (c_share / w_share) if w_share else float("nan")
        flag = bool(share_ratio > ratio_limit) if np.isfinite(share_ratio) else False
        rows.append(
            {
                "level": "position",
                "name": ticker,
                "sleeve": portfolio.sleeve_of[ticker],
                "weight": weight,
                "standalone_vol": float(standalone[ticker]),
                "vol_contribution": contrib,
                "contribution_share": c_share,
                "weight_share": w_share,
                "share_ratio": share_ratio,
                "outsized_risk_flag": flag,
                "outsized_rule": f"contribution_share / weight_share > {ratio_limit}",
            }
        )

    for sleeve, sleeve_weight in portfolio.sleeve_weights.items():
        members = [t for t in tickers if portfolio.sleeve_of[t] == sleeve]
        contrib = float(components[members].sum()) if members else 0.0
        c_share = contrib / port_vol if port_vol else 0.0
        sleeve_rets = rets[members].mul(weights[members], axis=1).sum(axis=1)
        if sleeve_weight > 0:
            sleeve_r = sleeve_rets / sleeve_weight
            stand = float(sleeve_r.std(ddof=1) * np.sqrt(tdays))
        else:
            stand = 0.0
        w_share = float(sleeve_weight)
        share_ratio = (c_share / w_share) if w_share else float("nan")
        flag = bool(share_ratio > ratio_limit) if np.isfinite(share_ratio) else False
        rows.append(
            {
                "level": "sleeve",
                "name": sleeve,
                "sleeve": sleeve,
                "weight": w_share,
                "standalone_vol": stand,
                "vol_contribution": contrib,
                "contribution_share": c_share,
                "weight_share": w_share,
                "share_ratio": share_ratio,
                "outsized_risk_flag": flag,
                "outsized_rule": f"contribution_share / weight_share > {ratio_limit}",
            }
        )

    table = pd.DataFrame(rows)
    output_path = None
    if save:
        dest = processed_dir if processed_dir is not None else PROCESSED_DIR
        dest.mkdir(parents=True, exist_ok=True)
        stamp = (as_of or datetime.now(timezone.utc)).strftime("%Y%m%d")
        output_path = _unique_processed_path(dest, "risk_contribution", stamp)
        table.to_csv(output_path, index=False)
        LOGGER.info("Wrote %s.", output_path)

    return RiskContributionResult(table=table, portfolio_vol=port_vol, output_path=output_path)
