"""Repeatable markdown + HTML risk report from the Week 2 engine (no invented opinions)."""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.alerts import AlertReport, evaluate_alerts, top_risks
from src.config_loader import ThresholdsConfig, load_thresholds_config
from src.paths import PROJECT_ROOT, REPORTS_DIR
from src.pipeline import Week2Result, run_week2_pipeline
from src.scenarios import APPROXIMATION_DISCLAIMER

LOGGER = logging.getLogger(__name__)

USER_COMMENTARY_EXEC = (
    "[USER COMMENTARY: Executive summary — replace this block with your own "
    "judgment of what the committee should do. Numbers below are engine outputs, not advice.]"
)
USER_COMMENTARY_RISKS = (
    "[USER COMMENTARY: Top 3 risks — replace this block with your interpretation "
    "of drivers, hedges, and whether the flags are action-worthy.]"
)


@dataclass
class KpiSnapshot:
    portfolio_name: str
    base_currency: str
    nav: float
    ytd_return: float
    one_year_return: float
    annualized_vol: float
    max_drawdown: float
    var_99: float
    es_99: float
    var_99_ccy: float
    es_99_ccy: float
    as_of: str


def compound_return(returns: pd.Series) -> float:
    series = returns.dropna().astype(float)
    if series.empty:
        return float("nan")
    return float((1.0 + series).prod() - 1.0)


def kpi_snapshot(week2: Week2Result) -> KpiSnapshot:
    port = week2.portfolio_returns["portfolio_return"].dropna().astype(float)
    last = pd.Timestamp(port.index.max())
    ytd = port[port.index >= pd.Timestamp(year=last.year, month=1, day=1)]
    tdays = week2.risk_config.performance.trading_days_per_year
    one_year = port.iloc[-tdays:] if len(port) >= tdays else port
    wealth = week2.metrics.cumulative.dropna().astype(float)
    nav = float(week2.portfolio_config.total_capital * float(wealth.iloc[-1]))
    vol = float(
        week2.metrics.summary.loc[
            week2.metrics.summary["metric"] == "annualized_volatility", "value"
        ].iloc[0]
    )
    dd = float(
        week2.metrics.summary.loc[
            week2.metrics.summary["metric"] == "max_drawdown", "value"
        ].iloc[0]
    )
    var99, es99 = float("nan"), float("nan")
    for est in week2.metrics.var_es:
        if abs(est.confidence - 0.99) < 1e-12:
            var99, es99 = est.historical_var, est.historical_es
            break
    capital = week2.portfolio_config.total_capital
    return KpiSnapshot(
        portfolio_name=week2.portfolio_config.portfolio_name,
        base_currency=week2.portfolio_config.base_currency,
        nav=nav,
        ytd_return=compound_return(ytd),
        one_year_return=compound_return(one_year),
        annualized_vol=vol,
        max_drawdown=dd,
        var_99=var99,
        es_99=es99,
        var_99_ccy=var99 * capital,
        es_99_ccy=es99 * capital,
        as_of=str(last.date()),
    )


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_(none)_\n"
    cols = list(frame.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [header, sep]
    for _, row in frame.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:.6g}")
            else:
                cells.append(str(val).replace("|", "/").replace("\n", " "))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _limitations_block(thresholds: ThresholdsConfig) -> str:
    items = thresholds.assumptions_and_limitations
    if not items:
        readme = PROJECT_ROOT / "README.md"
        if readme.is_file():
            text = readme.read_text(encoding="utf-8")
            if "## Known limitations" in text:
                chunk = text.split("## Known limitations", 1)[1]
                chunk = chunk.split("\n## ", 1)[0]
                return chunk.strip()
        return "_(none configured)_"
    return "\n".join(f"- {item}" for item in items)


def render_markdown(
    week2: Week2Result,
    alerts: AlertReport,
    thresholds: ThresholdsConfig,
    kpis: KpiSnapshot,
) -> str:
    risks = top_risks(alerts, week2, n=3)
    metrics = week2.metrics.summary[["metric", "value", "notes"]].copy()
    alerts_tbl = alerts.to_frame()
    scenarios = week2.scenarios.table.copy()
    contrib = week2.contribution.table.copy()

    lines = [
        f"# Risk report — {kpis.portfolio_name}",
        "",
        f"As of **{kpis.as_of}**. Generated from the Week 2 risk engine. "
        f"Base currency: {kpis.base_currency}.",
        "",
        "## Executive summary",
        "",
        f"**Overall status: {alerts.overall_status.upper()}** — {alerts.overall_reason}",
        "",
        f"> {USER_COMMENTARY_EXEC}",
        "",
        "## Key metrics",
        "",
        "| KPI | Value |",
        "| --- | --- |",
        f"| NAV | {kpis.nav:,.0f} {kpis.base_currency} |",
        f"| YTD return | {kpis.ytd_return:.4%} |",
        f"| 1Y return | {kpis.one_year_return:.4%} |",
        f"| Annualized volatility | {kpis.annualized_vol:.4%} |",
        f"| Max drawdown | {kpis.max_drawdown:.4%} |",
        f"| 99% 1-day historical VaR | {kpis.var_99:.4%} ({kpis.var_99_ccy:,.0f} {kpis.base_currency}) |",
        f"| 99% 1-day historical ES | {kpis.es_99:.4%} ({kpis.es_99_ccy:,.0f} {kpis.base_currency}) |",
        "",
        "Engine summary:",
        "",
        _md_table(metrics),
        "",
        "## Top 3 risks with evidence",
        "",
        f"> {USER_COMMENTARY_RISKS}",
        "",
        _md_table(risks),
        "",
        "Risk contribution (positions):",
        "",
        _md_table(contrib[contrib["level"] == "position"]),
        "",
        "## Scenario results",
        "",
        f"_{APPROXIMATION_DISCLAIMER}_",
        "",
        _md_table(
            scenarios[["scenario", "type", "pnl_pct", "pnl_currency", "worst_hit_sleeve", "disclaimer"]]
            if not scenarios.empty
            else scenarios
        ),
        "",
        "## Alerts and recommended actions",
        "",
        "Escalation text is copied from `config/thresholds.yaml` for amber/red breaches only.",
        "",
        _md_table(alerts_tbl),
        "",
        "## Data notes & limitations",
        "",
        _limitations_block(thresholds),
        "",
        f"Portfolio config: `{week2.portfolio_config.source_path}`. "
        f"Risk config: `{week2.risk_config.source_path}`. "
        f"Thresholds: `{thresholds.source_path}`.",
        "",
    ]
    return "\n".join(lines)


def render_html(markdown_text: str, title: str) -> str:
    """Minimal HTML wrapper so the report can be opened without extra libraries."""
    escaped = html.escape(markdown_text)
    return (
        "<!DOCTYPE html>\n<html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{font-family:Segoe UI,Helvetica,Arial,sans-serif;max-width:960px;"
        "margin:24px auto;padding:0 16px;color:#1f2933;background:#f7f7f5;}"
        "pre{white-space:pre-wrap;background:#fff;border:1px solid #d9d9d6;"
        "padding:16px;line-height:1.45;}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        "<p>Plain rendering of the markdown report (source of truth is the .md file).</p>"
        f"<pre>{escaped}</pre></body></html>\n"
    )


@dataclass
class ReportPaths:
    markdown: Path
    html: Path


def write_report(
    week2: Week2Result | None = None,
    *,
    thresholds: ThresholdsConfig | None = None,
    reports_dir: Path | None = None,
    as_of: datetime | None = None,
) -> ReportPaths:
    """Write ``reports/risk_report_YYYYMMDD.md`` and a companion HTML file."""
    snapshot = week2 if week2 is not None else run_week2_pipeline(save=False)
    limits = thresholds if thresholds is not None else load_thresholds_config()
    alerts = evaluate_alerts(snapshot, limits)
    kpis = kpi_snapshot(snapshot)
    markdown = render_markdown(snapshot, alerts, limits, kpis)

    dest = reports_dir if reports_dir is not None else REPORTS_DIR
    dest.mkdir(parents=True, exist_ok=True)
    stamp = (as_of or datetime.now(timezone.utc)).strftime("%Y%m%d")
    md_path = dest / f"risk_report_{stamp}.md"
    if md_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        md_path = dest / f"risk_report_{stamp}.md"
    html_path = dest / (md_path.stem + ".html")
    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(
        render_html(markdown, f"Risk report — {kpis.portfolio_name}"),
        encoding="utf-8",
    )
    LOGGER.info("Wrote %s and %s.", md_path, html_path)
    return ReportPaths(markdown=md_path, html=html_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = write_report()
    print(paths.markdown)
    print(paths.html)
