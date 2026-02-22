"""
CLI entry point.

python -m stockanalyzeV2 [options]

Examples:
    python -m stockanalyzeV2 --market JP --export-csv --top 5
    python -m stockanalyzeV2 --market US --top 10 --output-dir /tmp/results
    python -m stockanalyzeV2 --market JP --edinet-key $EDINET_API_KEY --estat-id $ESTAT_APP_ID
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure package root is on the Python path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stockanalyzeV2.analyzer import SectorAnalyzer


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    # Suppress noisy third-party loggers
    for noisy in ("urllib3", "requests", "yfinance", "peewee"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args(argv: list) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog        = "python -m stockanalyzeV2",
        description = "Sector Rotation & Fund Flow Analyzer (Top-Down Approach)",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--market",
        choices = ["JP", "US"],
        default = "JP",
        help    = "Target market",
    )
    p.add_argument(
        "--top",
        type    = int,
        default = 10,
        metavar = "N",
        help    = "Number of LONG/SHORT candidates to display",
    )
    p.add_argument(
        "--lookback",
        type    = int,
        default = 252,
        metavar = "DAYS",
        help    = "Price history window for indicator computation (trading days)",
    )
    p.add_argument(
        "--output-dir",
        default = "output",
        metavar = "DIR",
        help    = "Directory for CSV exports",
    )
    p.add_argument(
        "--export-csv",
        action  = "store_true",
        help    = "Write sector metrics and trade candidates to CSV files",
    )
    p.add_argument(
        "--edinet-key",
        default = os.getenv("EDINET_API_KEY", ""),
        metavar = "KEY",
        help    = "EDINET API key (JP only; overrides EDINET_API_KEY env var)",
    )
    p.add_argument(
        "--estat-id",
        default = os.getenv("ESTAT_APP_ID", ""),
        metavar = "ID",
        help    = "e-STAT application ID (JP only; overrides ESTAT_APP_ID env var)",
    )
    p.add_argument(
        "--verbose", "-v",
        action = "store_true",
        help   = "Enable debug logging",
    )
    return p.parse_args(argv)


def main(argv: list = None) -> None:
    args = _parse_args(argv or sys.argv[1:])
    _setup_logging(args.verbose)

    analyzer = SectorAnalyzer(
        market         = args.market,
        lookback_days  = args.lookback,
        output_dir     = args.output_dir,
        edinet_api_key = args.edinet_key,
        estat_app_id   = args.estat_id,
        top_n          = args.top,
    )

    result = analyzer.run(
        export_csv   = args.export_csv,
        print_report = True,
    )

    # Exit code: 0 if any candidates found, 1 if none
    has_signals = bool(result.long_candidates or result.short_candidates)
    sys.exit(0 if has_signals else 1)


if __name__ == "__main__":
    main()
