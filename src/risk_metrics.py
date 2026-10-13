"""Portfolio risk metrics, historical/parametric VaR, and Expected Shortfall.

Loss-positive convention (stated in every VaR/ES output):
    loss_t = -return_t
    VaR_alpha = empirical (or Gaussian) alpha-quantile of losses
    ES_alpha  = mean of losses at or beyond VaR_alpha
Positive VaR/ES numbers are losses. A 95% 1-day VaR of 0.012 means a 1.2% loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from src.config_loader import PortfolioConfig, RiskConfig
from src.data_cleaning import _unique_processed_path
from src.paths import PROCESSED_DIR

LOGGER = logging.getLogger(__name__)

LOSS_CONVENTION = "loss-positive (VaR and ES are positive numbers representing losses)"


@dataclass
class DrawdownStats:
    max_drawdown: float
    peak_date: pd.Timestamp | None
    trough_date: pd.Timestamp | None


@dataclass
class VarEsEstimate:
    confidence: float
    lookback_used: int
    horizon_days: int
    historical_var: float
    historical_es: float
    parametric_var: float
    parametric_es: float


@dataclass
class RiskMetricsResult:
    summary: pd.DataFrame
    correlation: pd.DataFrame
    rolling_vol: pd.Series
    rolling_sleeve_corr: pd.DataFrame
    drawdown_series: pd.Series
    cumulative: pd.Series
    var_es: list[VarEsEstimate]
    warnings: list[str] = field(default_factory=list)
    summary_path: Path | None = None
    correlation_path: Path | None = None


def annualized_return(returns: pd.Series, trading_days: int) -> float:
    return float(returns.mean() * trading_days)


def annualized_vol(returns: pd.Series, trading_days: int) -> float:
    return float(returns.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(returns: pd.Series, risk_free_annual: float, trading_days: int) -> float:
    vol = annualized_vol(returns, trading_days)
    if vol == 0.0 or np.isnan(vol):
        return float("nan")
    excess = annualized_return(returns, trading_days) - risk_free_annual
    return float(excess / vol)


def max_drawdown(returns: pd.Series) -> tuple[DrawdownStats, pd.Series]:
    """Peak-to-trough drawdown of a compounded wealth index (negative number)."""
    wealth = (1.0 + returns.astype(float)).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    if dd.empty:
        return DrawdownStats(max_drawdown=float("nan"), peak_date=None, trough_date=None), dd
    trough = dd.idxmin()
    peak_date = wealth.loc[:trough].idxmax()
    stats = DrawdownStats(
        max_drawdown=float(dd.loc[trough]),
        peak_date=pd.Timestamp(peak_date),
        trough_date=pd.Timestamp(trough),
    )
    return stats, dd


def rolling_volatility(returns: pd.Series, window: int, trading_days: int) -> pd.Series:
    vol = returns.rolling(window).std(ddof=1) * np.sqrt(trading_days)
    vol.name = "rolling_vol"
    return vol


def horizon_returns(returns: pd.Series, horizon_days: int) -> pd.Series:
    """Compounded simple returns over ``horizon_days`` (identity when horizon is 1)."""
    series = returns.dropna().astype(float)
    if horizon_days <= 1:
        return series
    compounded = (1.0 + series).rolling(horizon_days).apply(lambda x: float(np.prod(x) - 1.0), raw=True)
    return compounded.dropna()


def historical_var_es(losses: np.ndarray, confidence: float) -> tuple[float, float]:
    """VaR/ES on a loss-positive array. ES is the mean of losses >= VaR."""
    if losses.size == 0:
        return float("nan"), float("nan")
    var = float(np.quantile(losses, confidence, method="linear"))
    tail = losses[losses >= var]
    if tail.size == 0:
        es = var
    else:
        es = float(tail.mean())
    return var, es


def parametric_var_es(
    returns: np.ndarray,
    confidence: float,
    horizon_days: int,
) -> tuple[float, float]:
    """Gaussian VaR/ES of losses = -returns, scaled to the horizon.

    Loss ~ N(-mu, sigma^2). VaR = -mu + z_c * sigma.
    ES  = -mu + sigma * phi(z_c) / (1-c).
    """
    if returns.size < 2:
        return float("nan"), float("nan")
    mu = float(np.mean(returns)) * horizon_days
    sigma = float(np.std(returns, ddof=1)) * np.sqrt(horizon_days)
    if sigma == 0.0:
        loss_mean = -mu
        return loss_mean, loss_mean
    z = NormalDist().inv_cdf(confidence)
    var = -mu + sigma * z
    pdf = NormalDist().pdf(z)
    es = -mu + sigma * pdf / (1.0 - confidence)
    return float(var), float(es)


def compute_risk_metrics(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    portfolio: PortfolioConfig,
    risk: RiskConfig,
    *,
    processed_dir: Path | None = None,
    as_of: datetime | None = None,
    save: bool = True,
) -> RiskMetricsResult:
    """Compute Week 2 metrics from aligned asset and portfolio return panels."""
    warnings: list[str] = []
    tdays = risk.performance.trading_days_per_year
    port = portfolio_returns["portfolio_return"].dropna().astype(float)
    if port.empty:
        raise ValueError("portfolio_return series is empty; cannot compute risk metrics.")

    lookback = risk.var.lookback_days
    if len(port) < lookback:
        msg = (
            f"VaR lookback_days={lookback} exceeds available history ({len(port)} observations). "
            "Using the full available window; VaR/ES are noisier than configured."
        )
        LOGGER.warning(msg)
        warnings.append(msg)
        window = port
    else:
        window = port.iloc[-lookback:]

    horizon = risk.var.horizon_days
    h_returns = horizon_returns(window, horizon)
    losses = (-h_returns.to_numpy(dtype=float))

    var_es: list[VarEsEstimate] = []
    for conf in risk.var.confidence_levels:
        h_var, h_es = historical_var_es(losses, conf)
        p_var, p_es = parametric_var_es(window.to_numpy(dtype=float), conf, horizon)
        var_es.append(
            VarEsEstimate(
                confidence=conf,
                lookback_used=int(len(window)),
                horizon_days=horizon,
                historical_var=h_var,
                historical_es=h_es,
                parametric_var=p_var,
                parametric_es=p_es,
            )
        )
        LOGGER.info(
            "VaR/ES %s | historical VaR=%.6f ES=%.6f | parametric VaR=%.6f ES=%.6f | %s",
            f"{conf:.0%}",
            h_var,
            h_es,
            p_var,
            p_es,
            LOSS_CONVENTION,
        )

    dd_stats, dd_series = max_drawdown(port)
    roll = rolling_volatility(port, risk.performance.rolling_vol_window, tdays)
    wealth = (1.0 + port).cumprod()
    wealth.name = "cumulative_portfolio"

    tickers = [t for t in portfolio.tickers if t in asset_returns.columns]
    corr = asset_returns[tickers].corr()

    sleeve_corr = pd.DataFrame(index=port.index)
    window_n = risk.performance.rolling_vol_window
    for sleeve in portfolio.sleeve_weights:
        col = f"sleeve_{sleeve}_return"
        if col not in portfolio_returns.columns:
            continue
        sleeve_corr[sleeve] = (
            portfolio_returns[col].rolling(window_n).corr(port)
        )

    ann_ret = annualized_return(port, tdays)
    ann_vol = annualized_vol(port, tdays)
    sharpe = sharpe_ratio(port, risk.performance.risk_free_rate_annual, tdays)

    rows: list[dict[str, object]] = [
        {"metric": "annualized_return", "value": ann_ret, "notes": f"mean daily x {tdays}"},
        {"metric": "annualized_volatility", "value": ann_vol, "notes": f"sample std x sqrt({tdays})"},
        {
            "metric": "sharpe_ratio",
            "value": sharpe,
            "notes": f"(ann. return - {risk.performance.risk_free_rate_annual}) / ann. vol",
        },
        {"metric": "max_drawdown", "value": dd_stats.max_drawdown, "notes": "peak-to-trough, negative"},
        {
            "metric": "max_drawdown_peak_date",
            "value": str(dd_stats.peak_date.date()) if dd_stats.peak_date is not None else "",
            "notes": "",
        },
        {
            "metric": "max_drawdown_trough_date",
            "value": str(dd_stats.trough_date.date()) if dd_stats.trough_date is not None else "",
            "notes": "",
        },
        {"metric": "observations", "value": int(len(port)), "notes": "full sample"},
        {"metric": "var_lookback_used", "value": int(len(window)), "notes": f"configured {lookback}"},
        {"metric": "var_horizon_days", "value": horizon, "notes": ""},
        {"metric": "var_primary_method", "value": risk.var.method, "notes": "parametric reported as comparison"},
        {"metric": "loss_convention", "value": LOSS_CONVENTION, "notes": ""},
    ]
    for est in var_es:
        tag = f"{est.confidence:.2f}".replace("0.", "")
        rows.extend(
            [
                {
                    "metric": f"historical_var_{tag}",
                    "value": est.historical_var,
                    "notes": LOSS_CONVENTION,
                },
                {
                    "metric": f"historical_es_{tag}",
                    "value": est.historical_es,
                    "notes": "mean of losses at or beyond historical VaR",
                },
                {
                    "metric": f"parametric_var_{tag}",
                    "value": est.parametric_var,
                    "notes": "Gaussian comparison, " + LOSS_CONVENTION,
                },
                {
                    "metric": f"parametric_es_{tag}",
                    "value": est.parametric_es,
                    "notes": "Gaussian comparison, mean tail loss",
                },
            ]
        )
    summary = pd.DataFrame(rows)

    summary_path = None
    corr_path = None
    if save:
        dest = processed_dir if processed_dir is not None else PROCESSED_DIR
        dest.mkdir(parents=True, exist_ok=True)
        stamp = (as_of or datetime.now(timezone.utc)).strftime("%Y%m%d")
        summary_path = _unique_processed_path(dest, "risk_metrics_summary", stamp)
        corr_path = _unique_processed_path(dest, "correlation_matrix", stamp)
        summary.to_csv(summary_path, index=False)
        corr.to_csv(corr_path)
        LOGGER.info("Wrote %s and %s.", summary_path, corr_path)

    return RiskMetricsResult(
        summary=summary,
        correlation=corr,
        rolling_vol=roll,
        rolling_sleeve_corr=sleeve_corr,
        drawdown_series=dd_series,
        cumulative=wealth,
        var_es=var_es,
        warnings=warnings,
        summary_path=summary_path,
        correlation_path=corr_path,
    )
