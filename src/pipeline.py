"""Week 1 and Week 2 end-to-end orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from datetime import datetime, timezone

import pandas as pd

from src.config_loader import PortfolioConfig, RiskConfig, load_portfolio_config, load_risk_config
from src.data_cleaning import CleaningResult, clean_prices
from src.data_ingestion import IngestionResult, ingest_prices
from src.paths import PROCESSED_DIR, latest_processed_csv
from src.portfolio import PortfolioResult, compute_portfolio_returns
from src.risk_contribution import RiskContributionResult, compute_risk_contribution
from src.risk_metrics import RiskMetricsResult, compute_risk_metrics
from src.scenarios import ScenarioResults, run_scenarios
from src.validation import ValidationReport, run_validation

LOGGER = logging.getLogger(__name__)


@dataclass
class Week1Result:
    config: PortfolioConfig
    ingestion: IngestionResult
    cleaning: CleaningResult
    portfolio: PortfolioResult
    validation: ValidationReport


def run_week1_pipeline(
    config_path: Path | None = None,
    *,
    save: bool = True,
) -> Week1Result:
    """Load config, ingest, clean, compute portfolio returns, and validate."""
    config = load_portfolio_config(config_path)
    LOGGER.info("Loaded %s (%s positions).", config.portfolio_name, len(config.positions))

    ingestion = ingest_prices(config)
    cleaning = clean_prices(
        ingestion.prices,
        as_of=ingestion.pulled_at_utc,
        save=save,
    )
    portfolio = compute_portfolio_returns(
        cleaning.returns,
        config,
        as_of=ingestion.pulled_at_utc,
        save=save,
    )
    validation = run_validation(
        config,
        cleaning.prices,
        cleaning.returns,
        portfolio.returns,
        cleaning,
    )
    return Week1Result(
        config=config,
        ingestion=ingestion,
        cleaning=cleaning,
        portfolio=portfolio,
        validation=validation,
    )


@dataclass
class Week2Result:
    portfolio_config: PortfolioConfig
    risk_config: RiskConfig
    asset_returns: pd.DataFrame
    portfolio_returns: pd.DataFrame
    metrics: RiskMetricsResult
    contribution: RiskContributionResult
    scenarios: ScenarioResults


def _stamp_from_csv(path: Path) -> datetime:
    for part in reversed(path.stem.split("_")):
        if part.isdigit() and len(part) == 8:
            return datetime.strptime(part, "%Y%m%d").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def run_week2_pipeline(
    *,
    portfolio_path: Path | None = None,
    risk_path: Path | None = None,
    asset_returns: pd.DataFrame | None = None,
    portfolio_returns: pd.DataFrame | None = None,
    processed_dir: Path | None = None,
    save: bool = True,
) -> Week2Result:
    """Run the risk engine on Week 1 processed returns (or caller-supplied frames)."""
    portfolio_cfg = load_portfolio_config(portfolio_path)
    risk_cfg = load_risk_config(
        risk_path,
        portfolio_sleeves=list(portfolio_cfg.sleeve_weights.keys()),
    )
    dest = processed_dir if processed_dir is not None else PROCESSED_DIR

    if asset_returns is None:
        returns_path = latest_processed_csv("returns", dest)
        asset_returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        LOGGER.info("Loaded asset returns %s %s.", returns_path, asset_returns.shape)
        as_of = _stamp_from_csv(returns_path)
    else:
        as_of = datetime.now(timezone.utc)

    if portfolio_returns is None:
        port_path = latest_processed_csv("portfolio_returns", dest)
        portfolio_returns = pd.read_csv(port_path, index_col=0, parse_dates=True)
        LOGGER.info("Loaded portfolio returns %s %s.", port_path, portfolio_returns.shape)
        as_of = _stamp_from_csv(port_path)

    metrics = compute_risk_metrics(
        asset_returns,
        portfolio_returns,
        portfolio_cfg,
        risk_cfg,
        processed_dir=dest,
        as_of=as_of,
        save=save,
    )
    contribution = compute_risk_contribution(
        asset_returns,
        portfolio_cfg,
        risk_cfg,
        processed_dir=dest,
        as_of=as_of,
        save=save,
    )
    scenarios = run_scenarios(
        portfolio_returns,
        portfolio_cfg,
        risk_cfg,
        processed_dir=dest,
        as_of=as_of,
        save=save,
    )
    return Week2Result(
        portfolio_config=portfolio_cfg,
        risk_config=risk_cfg,
        asset_returns=asset_returns,
        portfolio_returns=portfolio_returns,
        metrics=metrics,
        contribution=contribution,
        scenarios=scenarios,
    )
