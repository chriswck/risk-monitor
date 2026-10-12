"""Week 1 end-to-end orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config_loader import PortfolioConfig, load_portfolio_config
from src.data_cleaning import CleaningResult, clean_prices
from src.data_ingestion import IngestionResult, ingest_prices
from src.portfolio import PortfolioResult, compute_portfolio_returns
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
