"""Streamlit presentation layer. All numbers come from src/ Week 2 modules."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.alerts import evaluate_alerts, top_risks
from src.config_loader import load_thresholds_config
from src.pipeline import Week2Result, run_week2_pipeline
from src.reporting import compound_return, kpi_snapshot
from src.risk_metrics import historical_var_es, horizon_returns
from src.scenarios import APPROXIMATION_DISCLAIMER

st.set_page_config(
    page_title="APAC Risk Monitor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BANNER = {
    "green": ("#1f5133", "#e7eee9"),
    "amber": ("#6b5414", "#f3ead4"),
    "red": ("#7a2424", "#f4e6e6"),
}


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; max-width: 1200px; }
        h1, h2, h3 { font-weight: 600; letter-spacing: -0.01em; }
        [data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Running risk engine…")
def load_engine() -> Week2Result:
    return run_week2_pipeline(save=False)


@st.cache_data(show_spinner=False)
def rolling_var_series(
    returns: pd.Series, lookback: int, confidence: float, horizon: int
) -> pd.Series:
    """Recompute historical VaR on each trailing window (existing engine function)."""
    series = returns.dropna().astype(float)
    values: list[float] = []
    index: list[pd.Timestamp] = []
    start = max(lookback, 2)
    for end in range(start, len(series) + 1):
        window = series.iloc[end - lookback : end] if end >= lookback else series.iloc[:end]
        h = horizon_returns(window, horizon)
        var, _ = historical_var_es((-h.to_numpy(dtype=float)), confidence)
        values.append(var)
        index.append(pd.Timestamp(series.index[end - 1]))
    return pd.Series(values, index=pd.DatetimeIndex(index), name=f"historical_var_{confidence:.0%}")


def _status_banner(status: str, reason: str) -> None:
    fg, bg = BANNER.get(status, BANNER["amber"])
    st.markdown(
        f"<div style='background:{bg};color:{fg};border:1px solid {fg}33;"
        f"padding:12px 16px;margin-bottom:12px;font-size:1.05rem;'>"
        f"<strong>PORTFOLIO STATUS: {status.upper()}</strong>"
        f"<span style='margin-left:12px'>{reason}</span></div>",
        unsafe_allow_html=True,
    )


def _heatmap(corr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), list(corr.columns), rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), list(corr.index))
    ax.set_title("Asset return correlation")
    fig.colorbar(im, ax=ax, fraction=0.046)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)


def tab_executive(week2: Week2Result, alerts, kpis, thresholds) -> None:
    _status_banner(alerts.overall_status, alerts.overall_reason)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric(f"NAV ({kpis.base_currency})", f"{kpis.nav:,.0f}")
    c2.metric("YTD return", f"{kpis.ytd_return:.2%}")
    c3.metric("1Y return", f"{kpis.one_year_return:.2%}")
    c4.metric("Ann. volatility", f"{kpis.annualized_vol:.2%}")
    c5.metric("Max drawdown", f"{kpis.max_drawdown:.2%}")
    c6.metric(
        "99% 1d VaR / ES",
        f"{kpis.var_99:.2%} / {kpis.es_99:.2%}",
        help=(
            f"VaR {kpis.var_99_ccy:,.0f} {kpis.base_currency}; "
            f"ES {kpis.es_99_ccy:,.0f} {kpis.base_currency}. Loss-positive convention."
        ),
    )
    st.caption(
        f"VaR {kpis.var_99_ccy:,.0f} {kpis.base_currency} | "
        f"ES {kpis.es_99_ccy:,.0f} {kpis.base_currency} | as of {kpis.as_of} | "
        "loss-positive (positive VaR/ES = loss)."
    )

    st.subheader("Top 3 current risks")
    st.dataframe(top_risks(alerts, week2, n=3), hide_index=True, use_container_width=True)

    st.subheader("Cumulative performance vs benchmark")
    perf = week2.portfolio_returns[["cumulative_portfolio", "cumulative_benchmark"]].copy()
    perf.columns = [week2.portfolio_config.portfolio_name, f"Benchmark ({week2.portfolio_config.benchmark})"]
    st.line_chart(perf)


def tab_exposures(week2: Week2Result) -> None:
    left, right = st.columns(2)
    weights = pd.Series(week2.portfolio_config.weights, name="weight")
    sleeve_w = pd.Series(week2.portfolio_config.sleeve_weights, name="weight")
    with left:
        st.subheader("Allocation by position")
        st.bar_chart(weights)
    with right:
        st.subheader("Allocation by sleeve")
        st.bar_chart(sleeve_w)

    st.subheader("Sleeve performance contribution")
    rows = []
    port = week2.portfolio_returns
    for sleeve, weight in week2.portfolio_config.sleeve_weights.items():
        rcol = f"sleeve_{sleeve}_return"
        ccol = f"sleeve_{sleeve}_contribution"
        sleeve_ret = compound_return(port[rcol]) if rcol in port.columns else float("nan")
        book_contrib = compound_return(port[ccol]) if ccol in port.columns else float("nan")
        rows.append(
            {
                "sleeve": sleeve,
                "weight": weight,
                "sleeve_compound_return": sleeve_ret,
                "contribution_to_book": book_contrib,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.subheader("Rolling volatility")
    st.line_chart(week2.metrics.rolling_vol.dropna())
    st.subheader("Drawdown")
    st.line_chart(week2.metrics.drawdown_series.dropna())


def tab_decomposition(week2: Week2Result) -> None:
    st.subheader("Correlation heatmap")
    _heatmap(week2.metrics.correlation)

    st.subheader("Risk contribution")
    contrib = week2.contribution.table.copy()
    st.caption(f"Portfolio vol (ann.): {week2.contribution.portfolio_vol:.4%}. Euler contributions sum to this vol.")
    st.dataframe(contrib, hide_index=True, use_container_width=True)

    st.subheader("VaR / ES")
    st.caption("Loss-positive convention: positive figures are losses. Parametric is a Gaussian comparison only.")
    var_rows = [
        {
            "confidence": est.confidence,
            "lookback_used": est.lookback_used,
            "horizon_days": est.horizon_days,
            "historical_var": est.historical_var,
            "historical_es": est.historical_es,
            "parametric_var": est.parametric_var,
            "parametric_es": est.parametric_es,
        }
        for est in week2.metrics.var_es
    ]
    st.dataframe(pd.DataFrame(var_rows), hide_index=True, use_container_width=True)

    st.subheader("VaR trend (rolling historical)")
    lookback = week2.risk_config.var.lookback_days
    horizon = week2.risk_config.var.horizon_days
    conf = 0.99 if 0.99 in week2.risk_config.var.confidence_levels else max(week2.risk_config.var.confidence_levels)
    trend = rolling_var_series(
        week2.portfolio_returns["portfolio_return"],
        lookback,
        conf,
        horizon,
    )
    st.caption(
        f"Each point is historical VaR at {conf:.0%} using the configured lookback of {lookback} days "
        f"(engine function `historical_var_es`)."
    )
    st.line_chart(trend)


def tab_scenarios(week2: Week2Result, alerts, thresholds) -> None:
    st.subheader("Scenario summary")
    st.info(APPROXIMATION_DISCLAIMER)
    sc = week2.scenarios.table
    show_cols = [c for c in ["scenario", "type", "pnl_pct", "pnl_currency", "worst_hit_sleeve", "disclaimer"] if c in sc.columns]
    st.dataframe(sc[show_cols] if show_cols else sc, hide_index=True, use_container_width=True)
    if week2.scenarios.warnings:
        st.caption("Warnings")
        for w in week2.scenarios.warnings:
            st.write(f"- {w}")

    st.subheader("Alerts")
    frame = alerts.to_frame()
    st.dataframe(frame, hide_index=True, use_container_width=True)
    breaches = alerts.breaches()
    if breaches:
        st.markdown("**Escalation text (from config/thresholds.yaml)**")
        for row in breaches:
            st.markdown(f"- **{row.metric}** ({row.status}): {row.escalation_text}")

    with st.expander("Assumptions & limitations"):
        items = thresholds.assumptions_and_limitations
        if items:
            for item in items:
                st.write(f"- {item}")
        else:
            st.write("See README.md Known limitations.")


def main() -> None:
    _inject_css()
    st.title("APAC Risk Monitor")
    st.caption("Committee dashboard — Week 1–2 engine outputs. No new analytics in this layer.")

    try:
        week2 = load_engine()
        thresholds = load_thresholds_config()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()
        return

    alerts = evaluate_alerts(week2, thresholds)
    kpis = kpi_snapshot(week2)

    t1, t2, t3, t4 = st.tabs(
        [
            "Executive Summary",
            "Exposures & Performance",
            "Risk Decomposition",
            "Scenarios & Alerts",
        ]
    )
    with t1:
        tab_executive(week2, alerts, kpis, thresholds)
    with t2:
        tab_exposures(week2)
    with t3:
        tab_decomposition(week2)
    with t4:
        tab_scenarios(week2, alerts, thresholds)


if __name__ == "__main__":
    main()
