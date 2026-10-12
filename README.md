# APAC Risk Monitor (Week 1)

APAC Risk Monitor is a config-driven foundation for watching a hypothetical APAC-oriented fixed-income / multi-asset book: it loads positions from YAML, pulls daily adjusted closes, cleans and validates the panel in the open, and builds static-weight portfolio, sleeve, and benchmark return series. Later phases (VaR, stress tests, dashboards) are intentionally out of scope here.

## Setup

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config/portfolio.yaml`: set `benchmark`, replace the example `positions` with your own tickers/weights/rationales, and keep weights summing to `1.0` (±0.001). Allowed sleeves are `rates`, `ig_credit`, `hy_em`, `fx`, and `alternatives`. A working hypothetical example is checked in so the pipeline can run immediately; it is not a recommendation.

## How to run

From the repository root:

```bash
python run_week1.py
```

That single command loads the config, downloads ~5 years of data, writes dated snapshots, computes returns, and prints validation. Equivalent stepwise run:

```bash
python -m src.data_ingestion
python -m src.data_cleaning
python -m src.portfolio
```

Then open `notebooks/01_week1_exploration.ipynb` for composition, cumulative performance vs the benchmark, and the validation report.

Tests (no live market-data calls):

```bash
pytest
```

Outputs:

- `data/raw/prices_YYYYMMDD.csv` — dated raw snapshots (never overwritten)
- `data/processed/returns_YYYYMMDD.csv` — aligned daily simple returns
- `data/processed/portfolio_returns_YYYYMMDD.csv` — portfolio, benchmark, cumulative, and sleeve series
- `data_sources.md` — append-only pull log (source, tickers, date, rows)

## Data sources and assumptions

- **Source:** Yahoo Finance via `yfinance`, daily **adjusted** close (splits/dividends). This is a public proxy feed, not a gold-copy vendor.
- **Universe:** whatever tickers you put in `config/portfolio.yaml`. The shipped file uses liquid USD-listed ETFs as stand-ins for rates, IG credit, HY/EM, JPY FX, gold, and an Asia ex-Japan equity benchmark. They are not the cash bonds, local curves, or OTC FX a real APAC book would hold.
- **History:** about five years; a name with under two years of data is warned and skipped; other names continue.
- **Calendar:** series are aligned on a **common trading calendar** (inner join after gap handling). A gap of **≤3 trading days** is forward-filled; longer gaps are reported and those dates are dropped. Extreme days with **|daily return| > 10%** are flagged and **kept**.
- **Portfolio math:** **static weights**, no intra-period rebalancing, no transaction costs, no FX translation beyond the ETF’s own USD NAV. Sleeve columns are weighted **contributions** (`w_i * r_i`) plus a sleeve return scaled by the sleeve’s total weight.
- **Currency:** `base_currency` is recorded as USD; the pipeline does not convert local-currency prices.

Every cleaning decision is printed/logged. Invalid config (weights, sleeves, missing fields) raises a clear error rather than guessing.
