"""Pull adjusted close prices from Yahoo Finance and snapshot them to disk."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config_loader import PortfolioConfig, load_portfolio_config
from src.paths import DATA_SOURCES_LOG, RAW_DIR

LOGGER = logging.getLogger(__name__)

DEFAULT_LOOKBACK_YEARS = 5
MIN_COVERAGE_DAYS = 365 * 2
SOURCE_NAME = "yfinance (Yahoo Finance adjusted close)"
DOWNLOAD_RETRIES = 4
RETRY_BASE_SECONDS = 1.5


@dataclass
class IngestionResult:
    prices: pd.DataFrame
    snapshot_path: Path
    warnings: list[str] = field(default_factory=list)
    failed_tickers: list[str] = field(default_factory=list)
    skipped_short_history: list[str] = field(default_factory=list)
    pulled_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def ingest_prices(
    config: PortfolioConfig | None = None,
    *,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    raw_dir: Path | None = None,
    log_path: Path | None = None,
    end: datetime | None = None,
) -> IngestionResult:
    """Download ~5y of daily adjusted closes for config tickers.

    Failed tickers and series with fewer than 2 years of history are warned
    about and omitted; remaining tickers still proceed.
    """
    cfg = config if config is not None else load_portfolio_config()
    dest_dir = raw_dir if raw_dir is not None else RAW_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    sources_log = log_path if log_path is not None else DATA_SOURCES_LOG

    as_of = end if end is not None else datetime.now(timezone.utc)
    start = as_of - timedelta(days=int(365.25 * lookback_years))
    start_str = start.strftime("%Y-%m-%d")
    end_str = (as_of + timedelta(days=1)).strftime("%Y-%m-%d")

    warnings: list[str] = []
    failed: list[str] = []
    short_history: list[str] = []
    tickers = cfg.download_tickers
    series, fetch_warnings = _fetch_all_adjusted_closes(tickers, start_str, end_str)
    warnings.extend(fetch_warnings)

    usable: dict[str, pd.Series] = {}
    for ticker in tickers:
        close = series.get(ticker)
        if close is None:
            failed.append(ticker)
            continue
        coverage_days = (close.index.max() - close.index.min()).days
        n_obs = int(close.notna().sum())
        if coverage_days < MIN_COVERAGE_DAYS or n_obs < 2:
            msg = (
                f"{ticker}: insufficient history "
                f"({coverage_days} calendar days, {n_obs} observations; need >= {MIN_COVERAGE_DAYS} days). "
                "Ticker skipped; remaining names continue."
            )
            LOGGER.warning(msg)
            warnings.append(msg)
            short_history.append(ticker)
            continue
        usable[ticker] = close

    if not usable:
        raise RuntimeError(
            "No tickers produced usable price history. "
            f"Failed: {failed or 'none'}; short history: {short_history or 'none'}. "
            "Check ticker symbols in config/portfolio.yaml and network access to Yahoo Finance. "
            "Yahoo may rate-limit; wait a minute and re-run python run_week1.py."
        )

    prices = pd.concat(usable, axis=1).sort_index()
    prices.index.name = "Date"
    snapshot_path = _unique_snapshot_path(dest_dir, as_of)
    prices.to_csv(snapshot_path, index=True)
    LOGGER.info("Wrote raw snapshot %s (%s rows x %s tickers).", snapshot_path, len(prices), prices.shape[1])

    notes_parts = []
    if failed:
        notes_parts.append("failed=" + ",".join(failed))
    if short_history:
        notes_parts.append("short_history=" + ",".join(short_history))
    notes = "; ".join(notes_parts) if notes_parts else "ok"
    _append_data_sources_log(
        sources_log,
        pulled_at=as_of,
        tickers=list(prices.columns),
        n_rows=len(prices),
        n_cols=prices.shape[1],
        output_file=snapshot_path,
        notes=notes,
    )

    for w in warnings:
        LOGGER.warning(w)

    return IngestionResult(
        prices=prices,
        snapshot_path=snapshot_path,
        warnings=warnings,
        failed_tickers=failed,
        skipped_short_history=short_history,
        pulled_at_utc=as_of,
    )


def _fetch_all_adjusted_closes(
    tickers: list[str],
    start: str,
    end: str,
) -> tuple[dict[str, pd.Series], list[str]]:
    """Batch-download first, then retry missing names individually with backoff."""
    warnings: list[str] = []
    series: dict[str, pd.Series] = {}

    batch, batch_warnings = _try_batch_download(tickers, start, end)
    warnings.extend(batch_warnings)
    series.update(batch)

    missing = [t for t in tickers if t not in series]
    for ticker in missing:
        close, ticker_warnings = _download_adjusted_close(ticker, start, end)
        warnings.extend(ticker_warnings)
        if close is not None:
            series[ticker] = close
    return series, warnings


def _try_batch_download(
    tickers: list[str],
    start: str,
    end: str,
) -> tuple[dict[str, pd.Series], list[str]]:
    if len(tickers) <= 1:
        return {}, []
    warnings: list[str] = []
    try:
        frame = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
    except Exception as exc:
        msg = f"Batch download failed ({exc}); falling back to per-ticker pulls."
        LOGGER.warning(msg)
        warnings.append(msg)
        return {}, warnings
    if frame is None or frame.empty:
        msg = "Batch download returned no rows; falling back to per-ticker pulls."
        LOGGER.warning(msg)
        warnings.append(msg)
        return {}, warnings

    found: dict[str, pd.Series] = {}
    for ticker in tickers:
        try:
            close = _extract_close(frame, ticker).dropna()
        except Exception:
            continue
        if close.empty:
            continue
        close.name = ticker
        found[ticker] = close
    if not found:
        warnings.append("Batch download could not be parsed per ticker; falling back to per-ticker pulls.")
    return found, warnings


def _download_adjusted_close(
    ticker: str,
    start: str,
    end: str,
) -> tuple[pd.Series | None, list[str]]:
    warnings: list[str] = []
    last_error = "unknown error"
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            frame = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if frame is not None and not frame.empty:
                close = _extract_close(frame, ticker).dropna()
                if not close.empty:
                    close.name = ticker
                    return close, warnings
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                close = hist["Close"].astype(float).dropna()
                if not close.empty:
                    close.name = ticker
                    return close, warnings
            last_error = "empty response"
        except Exception as exc:
            last_error = str(exc)
            LOGGER.warning("%s: attempt %s/%s failed (%s).", ticker, attempt, DOWNLOAD_RETRIES, exc)
            if not _is_retryable(exc):
                msg = f"{ticker}: download failed ({exc}). Continuing with remaining tickers."
                LOGGER.warning(msg)
                warnings.append(msg)
                return None, warnings
        if attempt < DOWNLOAD_RETRIES:
            delay = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            LOGGER.info("%s: retrying in %.1fs (Yahoo rate limits are common).", ticker, delay)
            time.sleep(delay)

    msg = (
        f"{ticker}: download failed after {DOWNLOAD_RETRIES} attempts ({last_error}). "
        "Continuing with remaining tickers."
    )
    LOGGER.warning(msg)
    warnings.append(msg)
    return None, warnings


def _is_retryable(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in ("rate limit", "too many requests", "timeout", "timed out", "temporarily")
    )


def _extract_close(frame: pd.DataFrame, ticker: str) -> pd.Series:
    """Return a 1-d Close series from yfinance's single- or multi-index columns."""
    if isinstance(frame.columns, pd.MultiIndex):
        level0 = list(frame.columns.get_level_values(0))
        if ticker in level0:
            sub = frame[ticker]
            if isinstance(sub, pd.DataFrame):
                return _extract_close(sub, ticker)
            return sub.astype(float)
        if "Close" in level0:
            close = frame["Close"]
        elif "Adj Close" in level0:
            close = frame["Adj Close"]
        else:
            close = frame.xs("Close", axis=1, level=-1, drop_level=True)
        if isinstance(close, pd.DataFrame):
            if ticker in close.columns:
                close = close[ticker]
            else:
                close = close.iloc[:, 0]
        return close.astype(float)
    if ticker in frame.columns and ticker not in {"Close", "Adj Close"}:
        return frame[ticker].astype(float)
    if "Close" in frame.columns:
        return frame["Close"].astype(float)
    if "Adj Close" in frame.columns:
        return frame["Adj Close"].astype(float)
    raise RuntimeError(f"{ticker}: downloaded frame has no Close / Adj Close column.")


def _unique_snapshot_path(raw_dir: Path, as_of: datetime) -> Path:
    """Dated snapshot that is never overwritten if the same day is pulled twice."""
    day = as_of.strftime("%Y%m%d")
    candidate = raw_dir / f"prices_{day}.csv"
    if not candidate.exists():
        return candidate
    stamp = as_of.strftime("%Y%m%d_%H%M%S")
    return raw_dir / f"prices_{stamp}.csv"


def _append_data_sources_log(
    log_path: Path,
    *,
    pulled_at: datetime,
    tickers: list[str],
    n_rows: int,
    n_cols: int,
    output_file: Path,
    notes: str,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text(
            "# Data sources log\n\n"
            "| Pulled at (UTC) | Source | Tickers | Rows | Columns | Output file | Notes |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n",
            encoding="utf-8",
        )
    ts = pulled_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    ticker_cell = ", ".join(tickers)
    line = (
        f"| {ts} | {SOURCE_NAME} | {ticker_cell} | {n_rows} | {n_cols} | "
        f"{output_file.as_posix()} | {notes} |\n"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = ingest_prices()
    print(f"Snapshot: {result.snapshot_path}")
    print(f"Shape: {result.prices.shape}")
    if result.warnings:
        print("Warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
