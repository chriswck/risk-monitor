"""Static-weight portfolio, benchmark, and sleeve contribution returns."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config_loader import PortfolioConfig, load_portfolio_config
from src.data_cleaning import _unique_processed_path
from src.paths import PROCESSED_DIR

LOGGER = logging.getLogger(__name__)

IDENTITY_ATOL = 1e-10


class PortfolioError(ValueError):
    """Raised when portfolio returns cannot be formed from the available data."""


@dataclass
class PortfolioResult:
    returns: pd.DataFrame
    output_path: Path | None
    missing_tickers: list[str]


def compute_portfolio_returns(
    asset_returns: pd.DataFrame,
    config: PortfolioConfig | None = None,
    *,
    processed_dir: Path | None = None,
    as_of: datetime | None = None,
    save: bool = True,
) -> PortfolioResult:
    """Weighted daily portfolio returns using static config weights (no rebalancing).

    Columns written:
    - portfolio_return: sum_i w_i * r_i
    - benchmark_return: daily return of the configured benchmark ticker
    - cumulative_portfolio / cumulative_benchmark: compounded (1+r) wealth index
    - sleeve_<name>: sum of w_i * r_i for names in that sleeve (contribution)
    """
    cfg = config if config is not None else load_portfolio_config()
    available = set(asset_returns.columns)
    missing = [t for t in cfg.tickers if t not in available]
    if missing:
        raise PortfolioError(
            "Cannot compute portfolio returns; missing return series for: "
            + ", ".join(missing)
            + ". Re-run ingestion or remove those tickers from config/portfolio.yaml."
        )

    weights = cfg.weights
    weighted = asset_returns[cfg.tickers].mul(pd.Series(weights), axis=1)
    portfolio_r = weighted.sum(axis=1)
    reconstructed = weighted.sum(axis=1)
    residual = (portfolio_r - reconstructed).abs().max()
    if residual > IDENTITY_ATOL:
        raise PortfolioError(
            f"Internal identity check failed: max |r_p - sum(w_i r_i)| = {residual}."
        )

    frame = pd.DataFrame({"portfolio_return": portfolio_r}, index=asset_returns.index)

    if cfg.benchmark in asset_returns.columns:
        frame["benchmark_return"] = asset_returns[cfg.benchmark]
    else:
        LOGGER.warning(
            "Benchmark ticker %s is not in the return panel; benchmark columns will be empty.",
            cfg.benchmark,
        )
        frame["benchmark_return"] = pd.NA

    frame["cumulative_portfolio"] = (1.0 + frame["portfolio_return"]).cumprod()
    bench = frame["benchmark_return"].astype(float)
    frame["cumulative_benchmark"] = (1.0 + bench.fillna(0.0)).cumprod()
    if frame["benchmark_return"].isna().all():
        frame["cumulative_benchmark"] = pd.NA

    for sleeve, sleeve_weight in cfg.sleeve_weights.items():
        members = [p.ticker for p in cfg.positions if p.sleeve == sleeve]
        contrib = asset_returns[members].mul(pd.Series({m: weights[m] for m in members}), axis=1).sum(axis=1)
        frame[f"sleeve_{sleeve}_contribution"] = contrib
        if sleeve_weight > 0:
            frame[f"sleeve_{sleeve}_return"] = contrib / sleeve_weight
        else:
            frame[f"sleeve_{sleeve}_return"] = 0.0

    frame.index.name = "Date"

    output_path: Path | None = None
    if save:
        dest = processed_dir if processed_dir is not None else PROCESSED_DIR
        dest.mkdir(parents=True, exist_ok=True)
        stamp = (as_of or datetime.now(timezone.utc)).strftime("%Y%m%d")
        output_path = _unique_processed_path(dest, "portfolio_returns", stamp)
        frame.to_csv(output_path, index=True)
        LOGGER.info("Wrote portfolio returns %s.", output_path)

    return PortfolioResult(returns=frame, output_path=output_path, missing_tickers=missing)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.data_cleaning import clean_prices
    from src.data_ingestion import ingest_prices

    cfg = load_portfolio_config()
    ingested = ingest_prices(cfg)
    cleaned = clean_prices(ingested.prices)
    result = compute_portfolio_returns(cleaned.returns, cfg)
    print(f"Portfolio returns: {result.output_path}")
    print(result.returns.tail())
