"""Generate the dated Week 3 risk report (markdown + HTML)."""

from __future__ import annotations

import logging

from src.reporting import write_report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = write_report()
    print(paths.markdown)
    print(paths.html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
