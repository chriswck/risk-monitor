"""Align prices, handle gaps explicitly, compute returns, and flag extremes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.paths import PROCESSED_DIR

LOGGER = logging.getLogger(__name__)

MAX_FFILL_BUSINESS_DAYS = 3
EXTREME_RETURN_THRESHOLD = 0.10


@dataclass
class ExtremeReturnFlag:
    date: pd.Timestamp
    ticker: str
    daily_return: float


@dataclass
class CleaningResult:
    prices: pd.DataFrame
    returns: pd.DataFrame
    decisions: list[str]
    extreme_flags: list[ExtremeReturnFlag]
    dropped_dates: list[pd.Timestamp]
    returns_path: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def extreme_flags_frame(self) -> pd.DataFrame:
        if not self.extreme_flags:
            return pd.DataFrame(columns=["date", "ticker", "daily_return"])
        return pd.DataFrame(
            [
                {"date": f.date, "ticker": f.ticker, "daily_return": f.daily_return}
                for f in self.extreme_flags
            ]
        )


def clean_prices(
    raw_prices: pd.DataFrame,
    *,
    processed_dir: Path | None = None,
    as_of: datetime | None = None,
    save: bool = True,
) -> CleaningResult:
    """Clean a wide price panel (DatetimeIndex, one column per ticker).

    Decisions (always logged, never silent):
    - sort and require a DatetimeIndex
    - duplicate dates: keep the first row, report the rest
    - outer-align tickers, forward-fill gaps of <= 3 consecutive trading days
    - longer gaps: leave as NA, then drop those dates (inner calendar)
    - daily simple returns; |r| > 10% is flagged and kept
    """
    decisions: list[str] = []
    warnings: list[str] = []

    if raw_prices.empty:
        raise ValueError("Cannot clean an empty price panel.")

    prices = raw_prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.apply(pd.to_numeric, errors="coerce")

    dup_mask = prices.index.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    if n_dup:
        dup_dates = prices.index[dup_mask].unique().tolist()
        msg = (
            f"Found {n_dup} duplicate date row(s) ({_preview_dates(dup_dates)}). "
            "Keeping the first occurrence of each date; later duplicates dropped."
        )
        decisions.append(msg)
        warnings.append(msg)
        prices = prices.loc[~dup_mask]
    else:
        decisions.append("No duplicate dates in the raw price panel.")

    prices = prices.dropna(axis=1, how="all")
    if prices.empty:
        raise ValueError("All ticker columns were empty after numeric conversion.")

    filled, gap_decisions, long_gap_dates = _ffill_short_gaps(prices)
    decisions.extend(gap_decisions)

    before_drop = len(filled)
    still_na = filled.isna().any(axis=1)
    dropped = filled.index[still_na].tolist()
    aligned = filled.loc[~still_na].copy()
    n_dropped = before_drop - len(aligned)
    decisions.append(
        f"Inner-joined onto a common trading calendar after gap handling: "
        f"{len(aligned)} dates remaining "
        f"({aligned.index.min().date() if len(aligned) else 'n/a'} to "
        f"{aligned.index.max().date() if len(aligned) else 'n/a'}); "
        f"dropped {n_dropped} date(s) that still had missing values "
        f"(leading NA, long gaps, or tickers that never listed that day)."
    )
    if long_gap_dates:
        warnings.append(
            f"Longer-than-{MAX_FFILL_BUSINESS_DAYS}-day gaps reported; "
            f"affected dates dropped from the aligned panel: {_preview_dates(long_gap_dates)}."
        )

    if len(aligned) < 2:
        raise ValueError(
            "Aligned price panel has fewer than 2 rows after cleaning; "
            "cannot compute daily returns. Check ticker history overlap."
        )

    aligned.index.name = "Date"
    returns = aligned.pct_change(fill_method=None)
    first_idx = returns.index[0]
    returns = returns.iloc[1:]
    decisions.append(
        f"Computed daily simple returns (p_t / p_{{t-1}} - 1); dropped the first date "
        f"{first_idx.date()} which has no prior close."
    )

    extreme_flags = _flag_extreme_returns(returns)
    if extreme_flags:
        msg = (
            f"Flagged {len(extreme_flags)} observation(s) with |daily return| > "
            f"{EXTREME_RETURN_THRESHOLD:.0%}. Rows were NOT removed."
        )
        decisions.append(msg)
        warnings.append(msg)
        for flag in extreme_flags:
            detail = (
                f"  extreme: {flag.date.date()} {flag.ticker} "
                f"return={flag.daily_return:.4%}"
            )
            decisions.append(detail)
            LOGGER.warning(detail)
    else:
        decisions.append(
            f"No |daily return| > {EXTREME_RETURN_THRESHOLD:.0%} observations found."
        )

    for line in decisions:
        LOGGER.info(line)

    returns_path: Path | None = None
    if save:
        dest = processed_dir if processed_dir is not None else PROCESSED_DIR
        dest.mkdir(parents=True, exist_ok=True)
        stamp = (as_of or datetime.now(timezone.utc)).strftime("%Y%m%d")
        returns_path = _unique_processed_path(dest, "returns", stamp)
        returns.to_csv(returns_path, index=True)
        decisions.append(f"Wrote processed returns to {returns_path}.")
        LOGGER.info("Wrote processed returns %s.", returns_path)

    return CleaningResult(
        prices=aligned,
        returns=returns,
        decisions=decisions,
        extreme_flags=extreme_flags,
        dropped_dates=dropped,
        returns_path=returns_path,
        warnings=warnings,
    )


def _ffill_short_gaps(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[pd.Timestamp]]:
    """Forward-fill consecutive trading-day gaps of length <= MAX_FFILL_BUSINESS_DAYS.

    Longer consecutive NA runs are left as NA so they can be dropped later.
    """
    out = prices.copy()
    decisions: list[str] = []
    long_gap_dates: list[pd.Timestamp] = []

    for ticker in out.columns:
        series = out[ticker]
        is_na = series.isna()
        if not bool(is_na.any()):
            decisions.append(f"{ticker}: no missing values on the union calendar.")
            continue
        run_id = (is_na != is_na.shift(fill_value=False)).cumsum()
        filled_runs = 0
        long_runs = 0
        for _, idx in series.groupby(run_id).groups.items():
            run = series.loc[idx]
            if not bool(run.isna().all()):
                continue
            gap_len = len(idx)
            start = idx.min()
            end = idx.max()
            loc = series.index.get_indexer([start])[0]
            has_prior = loc > 0 and pd.notna(series.iloc[loc - 1])
            if gap_len <= MAX_FFILL_BUSINESS_DAYS and has_prior:
                prior_value = series.iloc[loc - 1]
                out.loc[idx, ticker] = prior_value
                filled_runs += 1
                decisions.append(
                    f"{ticker}: forward-filled {gap_len} trading-day gap "
                    f"{pd.Timestamp(start).date()} to {pd.Timestamp(end).date()} "
                    f"(<= {MAX_FFILL_BUSINESS_DAYS} business days)."
                )
            else:
                long_runs += 1
                long_gap_dates.extend(list(idx))
                reason = (
                    "no prior close to fill from"
                    if not has_prior
                    else f"gap of {gap_len} exceeds {MAX_FFILL_BUSINESS_DAYS} business days"
                )
                decisions.append(
                    f"{ticker}: not filling gap {pd.Timestamp(start).date()} to "
                    f"{pd.Timestamp(end).date()} ({gap_len} trading days; {reason}). "
                    "Those dates will be dropped from the aligned panel if still missing."
                )
        if filled_runs == 0 and long_runs == 0:
            decisions.append(f"{ticker}: missing values inspected; no consecutive NA runs classified.")

    # unique long-gap dates, preserve order
    seen: set[pd.Timestamp] = set()
    unique_long: list[pd.Timestamp] = []
    for d in long_gap_dates:
        ts = pd.Timestamp(d)
        if ts not in seen:
            seen.add(ts)
            unique_long.append(ts)
    return out, decisions, unique_long


def _flag_extreme_returns(returns: pd.DataFrame) -> list[ExtremeReturnFlag]:
    flags: list[ExtremeReturnFlag] = []
    abs_ret = returns.abs()
    mask = abs_ret > EXTREME_RETURN_THRESHOLD
    if not bool(mask.any().any()):
        return flags
    for date, row in returns.loc[mask.any(axis=1)].iterrows():
        for ticker, value in row.items():
            if pd.notna(value) and abs(float(value)) > EXTREME_RETURN_THRESHOLD:
                flags.append(
                    ExtremeReturnFlag(
                        date=pd.Timestamp(date),
                        ticker=str(ticker),
                        daily_return=float(value),
                    )
                )
    return flags


def _preview_dates(dates: list[pd.Timestamp], limit: int = 8) -> str:
    if not dates:
        return "(none)"
    texts = [str(pd.Timestamp(d).date()) for d in dates[:limit]]
    extra = len(dates) - limit
    if extra > 0:
        texts.append(f"... +{extra} more")
    return ", ".join(texts)


def _unique_processed_path(directory: Path, stem: str, day: str) -> Path:
    candidate = directory / f"{stem}_{day}.csv"
    if not candidate.exists():
        return candidate
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return directory / f"{stem}_{stamp}.csv"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.data_ingestion import ingest_prices

    ingested = ingest_prices()
    cleaned = clean_prices(ingested.prices)
    print(f"Aligned prices: {cleaned.prices.shape}")
    print(f"Returns: {cleaned.returns_path}")
    print("Decisions:")
    for decision in cleaned.decisions:
        print(f"  - {decision}")
