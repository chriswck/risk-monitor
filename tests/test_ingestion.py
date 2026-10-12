"""Ingestion writes a dated snapshot, logs the pull, and continues on ticker failure."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data_ingestion import ingest_prices


class _EmptyTicker:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, **kwargs):
        return pd.DataFrame()


def _fast_retries(monkeypatch) -> None:
    monkeypatch.setattr("src.data_ingestion.DOWNLOAD_RETRIES", 1)
    monkeypatch.setattr("src.data_ingestion.RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr("src.data_ingestion.yf.Ticker", _EmptyTicker)


def test_ingest_skips_failed_and_short_tickers(monkeypatch, valid_config, tmp_path: Path) -> None:
    idx = pd.bdate_range("2019-01-02", "2024-01-02")
    long_close = pd.DataFrame({"Close": 100 + pd.Series(range(len(idx)), index=idx).astype(float)})
    short_idx = pd.bdate_range("2023-06-01", "2023-12-01")
    short_close = pd.DataFrame({"Close": 50 + pd.Series(range(len(short_idx)), index=short_idx).astype(float)})

    def fake_download(ticker, **kwargs):
        if ticker == "AAA":
            return long_close.copy()
        if ticker == "BBB":
            return long_close.copy()
        if ticker == "CCC":
            raise RuntimeError("simulated network error")
        if ticker == "BENCH":
            return short_close.copy()
        return pd.DataFrame()

    _fast_retries(monkeypatch)
    monkeypatch.setattr("src.data_ingestion.yf.download", fake_download)

    raw_dir = tmp_path / "raw"
    log_path = tmp_path / "data_sources.md"
    result = ingest_prices(
        valid_config,
        raw_dir=raw_dir,
        log_path=log_path,
        end=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )
    assert "CCC" in result.failed_tickers
    assert "BENCH" in result.skipped_short_history
    assert "AAA" in result.prices.columns
    assert result.snapshot_path.is_file()
    log_text = log_path.read_text(encoding="utf-8")
    assert "yfinance" in log_text
    assert "AAA" in log_text

    # Second pull on the same UTC day must not overwrite the first snapshot
    second = ingest_prices(
        valid_config,
        raw_dir=raw_dir,
        log_path=log_path,
        end=datetime(2024, 1, 15, 12, tzinfo=timezone.utc),
    )
    assert second.snapshot_path != result.snapshot_path
    assert result.snapshot_path.is_file()
    assert second.snapshot_path.is_file()


def test_ingest_raises_when_nothing_usable(monkeypatch, valid_config, tmp_path: Path) -> None:
    def fake_download(ticker, **kwargs):
        return pd.DataFrame()

    _fast_retries(monkeypatch)
    monkeypatch.setattr("src.data_ingestion.yf.download", fake_download)
    with pytest.raises(RuntimeError, match="No tickers produced usable price history"):
        ingest_prices(valid_config, raw_dir=tmp_path / "raw", log_path=tmp_path / "log.md")
