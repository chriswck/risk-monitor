# APAC Risk Monitor (Week 1–2)

APAC Risk Monitor is a config-driven tool for a hypothetical APAC-oriented fixed-income / multi-asset book. Week 1 loads positions from YAML, pulls daily adjusted closes, cleans and validates the panel, and builds static-weight portfolio, sleeve, and benchmark return series. Week 2 is the **risk engine**: performance metrics, historical (and Gaussian) VaR/ES, Euler risk contribution, and historical plus first-order hypothetical stress scenarios. Dashboards and report generators are out of scope until Week 3.

## Setup

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config/portfolio.yaml` (tickers, sleeves, weights) and `config/risk.yaml` (VaR window, risk-free rate, duration assumptions, scenario dates and shocks). Weights must sum to `1.0` (±0.001). Allowed sleeves: `rates`, `ig_credit`, `hy_em`, `fx`, `alternatives`.

## How to run

**Week 1** (data + portfolio returns), from the repository root:

```bash
python run_week1.py
```

**Week 2** (risk engine on the latest processed returns):

```bash
python run_week2.py
```

Then open `notebooks/01_week1_exploration.ipynb` and `notebooks/02_week2_risk_engine.ipynb`.

```bash
pytest
```

Outputs (`data/processed/`, dated, never overwritten):

- `returns_YYYYMMDD.csv`, `portfolio_returns_YYYYMMDD.csv` — Week 1
- `risk_metrics_summary_YYYYMMDD.csv`
- `correlation_matrix_YYYYMMDD.csv`
- `risk_contribution_YYYYMMDD.csv`
- `scenario_results_YYYYMMDD.csv`

## Risk methodology (Week 2)

- **Return / vol / Sharpe:** daily mean and sample std, annualized with `trading_days_per_year`; Sharpe uses `risk_free_rate_annual` from `config/risk.yaml`.
- **Drawdown:** peak-to-trough of the compounded wealth index, with peak and trough dates.
- **VaR / ES (loss-positive):** `loss = -return`. Historical VaR is the configured quantile of losses over `lookback_days` (warns and shortens if history is thinner). ES is the mean of losses at or beyond that VaR. A Gaussian (parametric) VaR/ES is reported alongside for comparison. Horizon scaling uses `horizon_days`.
- **Risk contribution:** Euler allocation `w_i (Σ w)_i / σ_p` on the annualized covariance; position and sleeve contributions sum to portfolio volatility. A position is flagged when contribution share / weight share exceeds `outsized_share_ratio`.
- **Historical scenarios:** compound P&L over configured date windows; windows outside the sample are warned, not silently filled.
- **Hypothetical shocks:** first-order sleeve P&L: rates `-duration × Δy × value`, credit `-spread_duration × Δs × value`, FX and alternatives as percentage NAV hits. Combined severe applies every configured shock at once.

## Known limitations

- Historical VaR/ES depend on the lookback window and the ETF proxy sample; they are not a forward-looking loss distribution.
- Duration and spread-duration shocks are **first-order**: they ignore convexity, basis, correlation shifts, and second-round liquidity effects. Every scenario row repeats that disclaimer.
- Portfolio math uses **static weights** (no rebalancing, no transaction costs).
- Market data are **public ETF proxies** via Yahoo Finance, not the cash bonds, local curves, or OTC FX a live APAC book would hold.
- Parametric VaR assumes Gaussian i.i.d. returns; it will typically understate tail loss relative to historical ES.

## Data sources and assumptions (Week 1)

- **Source:** Yahoo Finance via `yfinance`, daily **adjusted** close (splits/dividends).
- **History:** about five years; a name with under two years of data is warned and skipped.
- **Calendar:** common trading calendar after short-gap forward-fill (≤3 trading days). Longer gaps are dropped and reported. Extreme days with **|daily return| > 10%** are flagged and **kept**.
- **Currency:** `base_currency` is recorded as USD; the pipeline does not convert local-currency prices.

Every cleaning decision and every risk warning is printed. Invalid config raises a named error rather than guessing.
