# APAC Risk Monitor (Week 1–3)

APAC Risk Monitor is a config-driven tool for a hypothetical APAC-oriented fixed-income / multi-asset book. Week 1 loads positions from YAML, pulls daily adjusted closes, and builds static-weight returns. Week 2 is the **risk engine** (metrics, VaR/ES, Euler contribution, scenarios). Week 3 is the **presentation layer**: a Streamlit dashboard with threshold alerts and a dated markdown/HTML risk report. The dashboard calls existing `src/` modules; it does not reimplement analytics.

## Setup

Python 3.11+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Edit:

- `config/portfolio.yaml` — tickers, sleeves, weights
- `config/risk.yaml` — VaR window, durations, shocks
- `config/thresholds.yaml` — amber/red limits and escalation text

Weights must sum to `1.0` (±0.001). Allowed sleeves: `rates`, `ig_credit`, `hy_em`, `fx`, `alternatives`.

## How to run

**Week 1** (data + portfolio returns):

```bash
python run_week1.py
```

**Week 2** (risk engine):

```bash
python run_week2.py
```

**Week 3 — dashboard** (requires processed returns from Week 1; the app runs the Week 2 engine itself):

```bash
streamlit run app.py
```

**Week 3 — written report:**

```bash
python run_week3.py
# equivalent: python -m src.reporting
```

Writes `reports/risk_report_YYYYMMDD.md` and `.html`. Executive summary and top-risk narrative are `[USER COMMENTARY: …]` placeholders — fill those in yourself.

Notebooks: `notebooks/01_week1_exploration.ipynb`, `notebooks/02_week2_risk_engine.ipynb`.

```bash
pytest
```

## Screenshot placeholder

![Dashboard screenshot](docs/dashboard_screenshot.png)

_Add a screenshot of the Streamlit **Executive Summary** tab here after your first local run._

## Project structure

```
risk_portfolio/
├── app.py                      # Streamlit dashboard
├── run_week1.py / run_week2.py / run_week3.py
├── config/
│   ├── portfolio.yaml
│   ├── risk.yaml
│   └── thresholds.yaml
├── src/                        # engine + alerts + reporting
├── data/raw/  data/processed/
├── notebooks/
├── reports/                    # dated markdown + HTML
├── tests/
└── README.md
```

## Risk methodology (Week 2)

- **Return / vol / Sharpe:** daily mean and sample std, annualized with `trading_days_per_year`; Sharpe uses `risk_free_rate_annual`.
- **Drawdown:** peak-to-trough of the compounded wealth index.
- **VaR / ES (loss-positive):** `loss = -return`. Historical VaR is the configured quantile of losses; ES is the mean of losses at or beyond VaR. Gaussian VaR/ES is a comparison only.
- **Risk contribution:** Euler allocation; contributions sum to portfolio volatility.
- **Hypothetical shocks:** first-order sleeve P&L. Combined severe applies every configured shock at once.

## Known limitations

- Historical VaR/ES depend on the lookback window and the ETF proxy sample; they are not a forward-looking loss distribution.
- Duration and spread-duration shocks are **first-order**: they ignore convexity, basis, correlation shifts, and second-round liquidity effects. Every scenario row repeats that disclaimer.
- Portfolio math uses **static weights** (no rebalancing, no transaction costs).
- Market data are **public ETF proxies** via Yahoo Finance, not the cash bonds, local curves, or OTC FX a live APAC book would hold.
- Parametric VaR assumes Gaussian i.i.d. returns; it will typically understate tail loss relative to historical ES.
- Alert thresholds are user-chosen committee limits, not regulatory capital rules.

## Data sources and assumptions (Week 1)

- **Source:** Yahoo Finance via `yfinance`, daily **adjusted** close.
- **History:** about five years; names with under two years are warned and skipped.
- **Calendar:** common trading calendar after short-gap forward-fill (≤3 trading days). Extreme days with **|daily return| > 10%** are flagged and **kept**.
- **Currency:** `base_currency` is recorded as USD; the pipeline does not convert local-currency prices.

Every cleaning decision and risk warning is printed. Invalid config raises a named error rather than guessing.
