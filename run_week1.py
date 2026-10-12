"""Single entry point for the Week 1 data pipeline."""

from __future__ import annotations

import logging
import sys

from src.pipeline import run_week1_pipeline


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_week1_pipeline()

    print()
    print(f"Portfolio: {result.config.portfolio_name}")
    print(f"Config:    {result.config.source_path}")
    print(f"Raw:       {result.ingestion.snapshot_path}  {result.ingestion.prices.shape}")
    if result.cleaning.returns_path:
        print(f"Returns:   {result.cleaning.returns_path}  {result.cleaning.returns.shape}")
    if result.portfolio.output_path:
        print(f"Portfolio: {result.portfolio.output_path}  {result.portfolio.returns.shape}")

    print()
    print("Cleaning decisions")
    for decision in result.cleaning.decisions:
        print(f"  - {decision}")

    print()
    print("Validation")
    for line in result.validation.summary_lines():
        print(f"  {line}")

    if result.ingestion.warnings:
        print()
        print("Ingestion warnings")
        for warning in result.ingestion.warnings:
            print(f"  - {warning}")

    if result.validation.hard_failures:
        print()
        print("Pipeline finished with validation failures.", file=sys.stderr)
        return 1
    print()
    print("Pipeline finished successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
