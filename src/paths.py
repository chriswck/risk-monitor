"""Resolved project paths (independent of the current working directory)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "portfolio.yaml"
DEFAULT_RISK_CONFIG_PATH = CONFIG_DIR / "risk.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_SOURCES_LOG = PROJECT_ROOT / "data_sources.md"


def latest_processed_csv(stem: str, directory: Path | None = None) -> Path:
    """Return the most recently named ``{stem}_YYYYMMDD.csv`` (or stamped variant)."""
    dest = directory if directory is not None else PROCESSED_DIR
    matches = sorted(dest.glob(f"{stem}_*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"No processed files matching {stem}_*.csv in {dest}. "
            "Run `python run_week1.py` first to build returns."
        )
    return matches[-1]
