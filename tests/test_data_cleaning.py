"""Cleaning logs gap handling and does not silently drop extreme returns."""

from __future__ import annotations

import pandas as pd

from src.data_cleaning import EXTREME_RETURN_THRESHOLD, clean_prices


def test_short_gaps_are_forward_filled(synthetic_prices: pd.DataFrame, tmp_path) -> None:
    result = clean_prices(synthetic_prices, processed_dir=tmp_path, save=True)
    assert any("forward-filled" in d and "BBB" in d for d in result.decisions)
    # Short-gap dates should survive in the aligned panel for BBB
    filled_days = synthetic_prices.index[20:22]
    for day in filled_days:
        if day in result.prices.index:
            assert pd.notna(result.prices.loc[day, "BBB"])


def test_long_gaps_are_dropped_and_reported(synthetic_prices: pd.DataFrame, tmp_path) -> None:
    result = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    long_gap_days = list(synthetic_prices.index[50:56])
    for day in long_gap_days:
        assert day not in result.prices.index
    assert any("not filling gap" in d and "CCC" in d for d in result.decisions)


def test_inner_join_common_calendar(synthetic_prices: pd.DataFrame, tmp_path) -> None:
    result = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    assert result.prices.isna().sum().sum() == 0
    assert any("Inner-joined" in d for d in result.decisions)


def test_extreme_returns_flagged_not_removed(synthetic_prices: pd.DataFrame, tmp_path) -> None:
    result = clean_prices(synthetic_prices, processed_dir=tmp_path, save=False)
    assert result.extreme_flags, "expected the planted 15% AAA return to be flagged"
    assert any(f.ticker == "AAA" and abs(f.daily_return) > EXTREME_RETURN_THRESHOLD for f in result.extreme_flags)
    flagged_dates = {f.date for f in result.extreme_flags}
    for date in flagged_dates:
        assert date in result.returns.index


def test_duplicate_dates_reported(synthetic_prices: pd.DataFrame, tmp_path) -> None:
    duped = pd.concat([synthetic_prices.iloc[:3], synthetic_prices.iloc[2:3], synthetic_prices.iloc[3:]])
    result = clean_prices(duped, processed_dir=tmp_path, save=False)
    assert not result.prices.index.duplicated().any()
    assert any("duplicate date" in d for d in result.decisions)
