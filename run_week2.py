"""Single entry point for the Week 2 risk engine."""

from __future__ import annotations

import logging

from src.pipeline import run_week2_pipeline
from src.scenarios import APPROXIMATION_DISCLAIMER


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_week2_pipeline()

    print()
    print(f"Portfolio: {result.portfolio_config.portfolio_name}")
    print(f"Risk config: {result.risk_config.source_path}")
    if result.metrics.summary_path:
        print(f"Metrics:     {result.metrics.summary_path}")
    if result.metrics.correlation_path:
        print(f"Correlation: {result.metrics.correlation_path}")
    if result.contribution.output_path:
        print(f"Contribution:{result.contribution.output_path}")
    if result.scenarios.output_path:
        print(f"Scenarios:   {result.scenarios.output_path}")

    print()
    print("Key metrics")
    print(result.metrics.summary.to_string(index=False))

    print()
    print("VaR / ES (loss-positive convention)")
    for est in result.metrics.var_es:
        print(
            f"  {est.confidence:.0%}  historical VaR={est.historical_var:.6f}  "
            f"ES={est.historical_es:.6f}  |  parametric VaR={est.parametric_var:.6f}  "
            f"ES={est.parametric_es:.6f}"
        )

    print()
    print("Risk contribution (positions)")
    pos = result.contribution.table[result.contribution.table["level"] == "position"]
    print(
        pos[
            [
                "name",
                "weight",
                "standalone_vol",
                "vol_contribution",
                "share_ratio",
                "outsized_risk_flag",
            ]
        ].to_string(index=False)
    )

    print()
    print("Scenarios")
    print(
        result.scenarios.table[
            ["scenario", "type", "pnl_pct", "pnl_currency", "worst_hit_sleeve"]
        ].to_string(index=False)
    )
    print(APPROXIMATION_DISCLAIMER)

    if result.metrics.warnings or result.scenarios.warnings:
        print()
        print("Warnings")
        for warning in [*result.metrics.warnings, *result.scenarios.warnings]:
            print(f"  - {warning}")

    print()
    print("Risk engine finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
