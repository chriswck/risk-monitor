"""Week 1 validation checks, also exercised as pytest tests."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from src.data_cleaning import clean_prices
from src.portfolio import compute_portfolio_returns
from src.validation import (
    check_extreme_returns_reported,
    check_minimum_coverage,
    check_no_duplicate_dates,
    check_portfolio_return_identity,
    check_weights_sum_to_one,
    run_validation,
)


def test_weights_sum_check_passes(valid_config) -> None:
    result = check_weights_sum_to_one(valid_config)
    assert result.passed


def test_no_duplicate_dates_detects_dupes(synthetic_prices) -> None:
    ok = check_no_duplicate_dates(synthetic_prices)
    assert ok.passed
    duped = pd.concat([synthetic_prices, synthetic_prices.iloc[[0]]])
    bad = check_no_duplicate_dates(duped)
    assert not bad.passed


def test_minimum_coverage_warns_on_short_series(synthetic_prices) -> None:
    short = synthetic_prices.iloc[:40].copy()
    result = check_minimum_coverage(short)
    assert result.is_warning
    long_enough = synthetic_prices.copy()
    long_enough.index = pd.bdate_range("2019-01-02", periods=len(long_enough))
    # Force span >= 2 years by stretching index
    start = long_enough.index.min()
    long_enough.index = pd.date_range(start, periods=len(long_enough), freq="B")
    if long_enough.index.max() - long_enough.index.min() < timedelta(days=365 * 2):
        long_enough.index = pd.date_range("2018-01-02", periods=len(long_enough), freq="B")
    ok = check_minimum_coverage(long_enough)
    assert ok.passed
    assert not ok.is_warning


def test_portfolio_identity_on_random_dates(valid_config, synthetic_prices, tmp_path) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    portfolio = compute_portfolio_returns(cleaned.returns, valid_config, save=False)
    result = check_portfolio_return_identity(
        cleaned.returns,
        portfolio.returns,
        valid_config,
        rng=np.random.default_rng(7),
    )
    assert result.passed, result.message


def test_extreme_flags_are_reported(synthetic_prices, tmp_path) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    result = check_extreme_returns_reported(cleaned)
    assert result.passed
    assert result.is_warning
    assert cleaned.extreme_flags


def test_run_validation_suite(valid_config, synthetic_prices, tmp_path) -> None:
    cleaned = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    portfolio = compute_portfolio_returns(cleaned.returns, valid_config, save=False)
    report = run_validation(
        valid_config,
        cleaned.prices,
        cleaned.returns,
        portfolio.returns,
        cleaned,
        rng=np.random.default_rng(1),
    )
    assert report.passed
    names = {c.name for c in report.checks}
    assert names == {
        "config_weights_sum_to_one",
        "no_duplicate_dates",
        "minimum_data_coverage",
        "portfolio_return_identity",
        "extreme_return_flags_reported",
    }
