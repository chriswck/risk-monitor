"""Evaluate Week 2 engine outputs against ``config/thresholds.yaml``."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config_loader import (
    ALLOWED_STATUSES,
    REQUIRED_THRESHOLD_METRICS,
    ThresholdsConfig,
)
from src.pipeline import Week2Result

STATUS_RANK = {"green": 0, "amber": 1, "red": 2}


@dataclass
class AlertRow:
    metric: str
    current_value: float
    amber_limit: float
    red_limit: float
    status: str
    likely_driver: str
    escalation_text: str

    @property
    def is_breach(self) -> bool:
        return self.status in {"amber", "red"}


@dataclass
class AlertReport:
    rows: list[AlertRow]
    overall_status: str
    overall_reason: str
    warnings: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "metric": r.metric,
                    "current_value": r.current_value,
                    "amber_limit": r.amber_limit,
                    "red_limit": r.red_limit,
                    "status": r.status,
                    "likely_driver": r.likely_driver,
                    "escalation_text": r.escalation_text,
                }
                for r in self.rows
            ]
        )

    def breaches(self) -> list[AlertRow]:
        return [r for r in self.rows if r.is_breach]


def classify_status(value: float, amber: float, red: float) -> str:
    """Loss-type traffic light. Boundaries: value >= amber is amber; value >= red is red."""
    if value != value:  # NaN
        return "green"
    if value >= red:
        return "red"
    if value >= amber:
        return "amber"
    return "green"


def overall_status(statuses: list[str]) -> str:
    """red if any metric is red; else amber if any is amber; else green."""
    cleaned = [s for s in statuses if s in ALLOWED_STATUSES]
    if any(s == "red" for s in cleaned):
        return "red"
    if any(s == "amber" for s in cleaned):
        return "amber"
    return "green"


def _summary_value(summary: pd.DataFrame, metric: str) -> float:
    match = summary.loc[summary["metric"] == metric, "value"]
    if match.empty:
        return float("nan")
    try:
        return float(match.iloc[0])
    except (TypeError, ValueError):
        return float("nan")


def _var_es_at(week2: Week2Result, confidence: float) -> tuple[float, float]:
    for est in week2.metrics.var_es:
        if abs(est.confidence - confidence) < 1e-12:
            return est.historical_var, est.historical_es
    if week2.metrics.var_es:
        last = max(week2.metrics.var_es, key=lambda e: e.confidence)
        return last.historical_var, last.historical_es
    return float("nan"), float("nan")


def _top_sleeve(week2: Week2Result) -> str:
    table = week2.contribution.table
    sleeves = table[table["level"] == "sleeve"]
    if sleeves.empty:
        return ""
    idx = sleeves["vol_contribution"].astype(float).idxmax()
    return str(sleeves.loc[idx, "name"])


def _largest_position(week2: Week2Result) -> tuple[str, float]:
    weights = week2.portfolio_config.weights
    if not weights:
        return "", float("nan")
    ticker = max(weights, key=weights.get)
    return ticker, float(weights[ticker]) * 100.0


def _combined_loss_pct(week2: Week2Result) -> tuple[float, str]:
    table = week2.scenarios.table
    if table.empty:
        return float("nan"), ""
    combined = table[table["type"] == "combined"]
    if combined.empty:
        combined = table[table["scenario"].astype(str).str.contains("Combined", case=False)]
    if combined.empty:
        return float("nan"), ""
    row = combined.iloc[0]
    pnl = row["pnl_pct"]
    try:
        pnl_f = float(pnl)
    except (TypeError, ValueError):
        return float("nan"), str(row.get("worst_hit_sleeve") or "")
    loss = max(0.0, -pnl_f) * 100.0
    return loss, str(row.get("worst_hit_sleeve") or "")


def current_values(week2: Week2Result) -> dict[str, tuple[float, str]]:
    """Map threshold metric id -> (current value in pct points, likely driver)."""
    vol = _summary_value(week2.metrics.summary, "annualized_volatility") * 100.0
    dd = abs(_summary_value(week2.metrics.summary, "max_drawdown")) * 100.0
    var99, es99 = _var_es_at(week2, 0.99)
    top_sleeve = _top_sleeve(week2)
    largest, largest_pct = _largest_position(week2)
    combined_loss, worst_hit = _combined_loss_pct(week2)
    return {
        "annualized_volatility_pct": (vol, f"dominant vol sleeve: {top_sleeve}" if top_sleeve else ""),
        "max_drawdown_pct": (dd, f"dominant vol sleeve: {top_sleeve}" if top_sleeve else ""),
        "var_99_1d_pct": (var99 * 100.0, f"dominant vol sleeve: {top_sleeve}" if top_sleeve else ""),
        "es_99_1d_pct": (es99 * 100.0, f"dominant vol sleeve: {top_sleeve}" if top_sleeve else ""),
        "single_position_weight_pct": (
            largest_pct,
            f"largest position: {largest}" if largest else "",
        ),
        "combined_scenario_loss_pct": (
            combined_loss,
            f"worst-hit sleeve: {worst_hit}" if worst_hit else "",
        ),
    }


def evaluate_alerts(week2: Week2Result, thresholds: ThresholdsConfig) -> AlertReport:
    """Build the status table and overall portfolio traffic light."""
    values = current_values(week2)
    rows: list[AlertRow] = []
    warnings: list[str] = []
    for metric in REQUIRED_THRESHOLD_METRICS:
        limit = thresholds.limits[metric]
        current, driver = values[metric]
        status = classify_status(current, limit.amber, limit.red)
        escalation = limit.escalation_text if status in {"amber", "red"} else ""
        if status in {"amber", "red"} and not escalation:
            warnings.append(f"{metric}: breach has empty escalation_text in {thresholds.source_path}.")
        rows.append(
            AlertRow(
                metric=metric,
                current_value=current,
                amber_limit=limit.amber,
                red_limit=limit.red,
                status=status,
                likely_driver=driver,
                escalation_text=escalation,
            )
        )

    overall = overall_status([r.status for r in rows])
    if overall == "green":
        reason = "All monitored metrics are inside the configured amber limits."
    else:
        offenders = [r for r in rows if r.status == overall]
        names = ", ".join(r.metric for r in offenders)
        reason = f"Overall {overall}: {names}."
    return AlertReport(rows=rows, overall_status=overall, overall_reason=reason, warnings=warnings)


def top_risks(report: AlertReport, week2: Week2Result, n: int = 3) -> pd.DataFrame:
    """Breaches first (red then amber, by distance past amber), padded with vol contributors."""
    ranked = sorted(
        report.breaches(),
        key=lambda r: (STATUS_RANK[r.status], r.current_value - r.amber_limit),
        reverse=True,
    )
    records: list[dict[str, object]] = []
    for row in ranked[:n]:
        records.append(
            {
                "rank": len(records) + 1,
                "source": "alert",
                "item": row.metric,
                "status": row.status,
                "value": row.current_value,
                "evidence": row.likely_driver,
            }
        )
    if len(records) < n:
        contrib = week2.contribution.table
        pos = contrib[contrib["level"] == "position"].sort_values(
            "vol_contribution", ascending=False
        )
        for _, crow in pos.iterrows():
            if len(records) >= n:
                break
            records.append(
                {
                    "rank": len(records) + 1,
                    "source": "risk_contribution",
                    "item": str(crow["name"]),
                    "status": "flag" if bool(crow.get("outsized_risk_flag")) else "watch",
                    "value": float(crow["vol_contribution"]),
                    "evidence": (
                        f"sleeve {crow['sleeve']}; share_ratio={float(crow['share_ratio']):.2f}"
                    ),
                }
            )
    return pd.DataFrame(records)
