"""Pipeline checks used both at runtime and by pytest."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd

from src.config_loader import PortfolioConfig, WEIGHT_SUM_TOLERANCE
from src.data_cleaning import CleaningResult, EXTREME_RETURN_THRESHOLD
from src.portfolio import IDENTITY_ATOL

MIN_COVERAGE_DAYS = 365 * 2
IDENTITY_SAMPLE_SIZE = 3


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    is_warning: bool = False


@dataclass
class ValidationReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def hard_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and not c.is_warning]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if c.is_warning]

    @property
    def passed(self) -> bool:
        return not self.hard_failures

    def add(self, check: CheckResult) -> None:
        self.checks.append(check)

    def summary_lines(self) -> list[str]:
        lines: list[str] = []
        for check in self.checks:
            if check.passed and not check.is_warning:
                tag = "PASS"
            elif check.is_warning:
                tag = "WARN"
            else:
                tag = "FAIL"
            lines.append(f"[{tag}] {check.name}: {check.message}")
        return lines


def check_weights_sum_to_one(config: PortfolioConfig) -> CheckResult:
    total = config.weight_sum()
    ok = abs(total - 1.0) <= WEIGHT_SUM_TOLERANCE
    return CheckResult(
        name="config_weights_sum_to_one",
        passed=ok,
        message=(
            f"weights sum to {total:.6f} (tolerance +/-{WEIGHT_SUM_TOLERANCE})."
            if ok
            else f"weights sum to {total:.6f}, expected 1.0 +/- {WEIGHT_SUM_TOLERANCE}."
        ),
    )


def check_no_duplicate_dates(prices: pd.DataFrame) -> CheckResult:
    n_dup = int(prices.index.duplicated().sum())
    ok = n_dup == 0
    return CheckResult(
        name="no_duplicate_dates",
        passed=ok,
        message="price index has unique dates." if ok else f"found {n_dup} duplicate date row(s).",
    )


def check_minimum_coverage(
    prices: pd.DataFrame,
    *,
    min_days: int = MIN_COVERAGE_DAYS,
) -> CheckResult:
    """Warn (do not fail) if any ticker spans fewer than two calendar years."""
    short: list[str] = []
    details: list[str] = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            short.append(ticker)
            details.append(f"{ticker}: no observations")
            continue
        span = series.index.max() - series.index.min()
        if span < timedelta(days=min_days):
            short.append(ticker)
            details.append(f"{ticker}: {span.days} calendar days")
    if short:
        return CheckResult(
            name="minimum_data_coverage",
            passed=True,
            is_warning=True,
            message="tickers below 2-year coverage: " + "; ".join(details),
        )
    span = prices.index.max() - prices.index.min() if len(prices) else timedelta(0)
    return CheckResult(
        name="minimum_data_coverage",
        passed=True,
        message=f"all {prices.shape[1]} tickers cover >= {min_days} calendar days "
        f"(aligned panel span {span.days} days).",
    )


def check_portfolio_return_identity(
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    config: PortfolioConfig,
    *,
    n_dates: int = IDENTITY_SAMPLE_SIZE,
    rng: np.random.Generator | None = None,
) -> CheckResult:
    """Verify r_p == sum(w_i * r_i) on at least ``n_dates`` random dates."""
    common = asset_returns.index.intersection(portfolio_returns.index)
    if len(common) < n_dates:
        return CheckResult(
            name="portfolio_return_identity",
            passed=False,
            message=f"need at least {n_dates} overlapping dates; found {len(common)}.",
        )
    generator = rng if rng is not None else np.random.default_rng(42)
    chosen = generator.choice(common.to_numpy(), size=n_dates, replace=False)
    weights = pd.Series(config.weights)
    max_err = 0.0
    used: list[str] = []
    for ts in chosen:
        date = pd.Timestamp(ts)
        asset_row = asset_returns.loc[date, config.tickers].astype(float)
        expected = float((weights * asset_row).sum())
        actual = float(portfolio_returns.loc[date, "portfolio_return"])
        err = abs(actual - expected)
        max_err = max(max_err, err)
        used.append(str(date.date()))
        if err > IDENTITY_ATOL:
            return CheckResult(
                name="portfolio_return_identity",
                passed=False,
                message=(
                    f"identity failed on {date.date()}: portfolio_return={actual:.12e} "
                    f"vs sum(w_i r_i)={expected:.12e} (abs err {err:.12e} > {IDENTITY_ATOL:.0e})."
                ),
            )
    return CheckResult(
        name="portfolio_return_identity",
        passed=True,
        message=(
            f"portfolio_return == sum(weight_i * return_i) on {n_dates} random dates "
            f"({', '.join(used)}); max abs error {max_err:.2e} <= {IDENTITY_ATOL:.0e}."
        ),
    )


def check_extreme_returns_reported(cleaning: CleaningResult) -> CheckResult:
    abs_max = cleaning.returns.abs().max().max() if not cleaning.returns.empty else 0.0
    computed_flags = int((cleaning.returns.abs() > EXTREME_RETURN_THRESHOLD).sum().sum())
    reported = len(cleaning.extreme_flags)
    if computed_flags != reported:
        return CheckResult(
            name="extreme_return_flags_reported",
            passed=False,
            message=(
                f"panel contains {computed_flags} |r| > {EXTREME_RETURN_THRESHOLD:.0%} "
                f"observation(s) but only {reported} were reported."
            ),
        )
    if reported == 0:
        return CheckResult(
            name="extreme_return_flags_reported",
            passed=True,
            message=f"no |daily return| > {EXTREME_RETURN_THRESHOLD:.0%} (max |r|={abs_max:.4%}).",
        )
    return CheckResult(
        name="extreme_return_flags_reported",
        passed=True,
        is_warning=True,
        message=(
            f"{reported} |daily return| > {EXTREME_RETURN_THRESHOLD:.0%} observation(s) reported "
            f"(max |r|={abs_max:.4%}); series were not dropped."
        ),
    )


def run_validation(
    config: PortfolioConfig,
    prices: pd.DataFrame,
    asset_returns: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    cleaning: CleaningResult,
    *,
    rng: np.random.Generator | None = None,
) -> ValidationReport:
    """Run the Week 1 check suite and return a structured report."""
    report = ValidationReport()
    report.add(check_weights_sum_to_one(config))
    report.add(check_no_duplicate_dates(prices))
    report.add(check_minimum_coverage(prices))
    report.add(
        check_portfolio_return_identity(
            asset_returns, portfolio_returns, config, rng=rng
        )
    )
    report.add(check_extreme_returns_reported(cleaning))
    return report
