"""
Sector Rotation & Fund Flow Analyzer — Main Orchestrator.

Pipeline:
  1. Fetch stock data (yfinance) for all sectors
  2. Compute Sector Composite Scores (SCS)
  3. Detect sector rotation (vs. previous snapshot if provided)
  4. Generate LONG / SHORT trade signals
  5. Fetch EDINET filings (optional, JP only)
  6. Fetch e-STAT macro indicators (optional, JP only)
  7. Print report and export CSVs

Usage (as a library):
    from stockanalyzeV2.analyzer import SectorAnalyzer

    analyzer = SectorAnalyzer(market="JP")
    result   = analyzer.run()

Usage (CLI):
    python -m stockanalyzeV2 --market JP --export-csv --top 5
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Optional, Dict

from stockanalyzeV2.config import (
    JP_SECTOR_TICKERS,
    US_SECTOR_TICKERS,
    SCSWeights,
    SignalConfig,
    RotationConfig,
)
from stockanalyzeV2.data_fetchers.yfinance_fetcher  import YFinanceFetcher
from stockanalyzeV2.data_fetchers.edinet_fetcher    import EDINETFetcher
from stockanalyzeV2.data_fetchers.estat_fetcher     import ESTATFetcher
from stockanalyzeV2.analysis.scs_calculator         import SCSCalculator
from stockanalyzeV2.analysis.rotation_detector      import RotationDetector
from stockanalyzeV2.analysis.signal_generator       import SignalGenerator
from stockanalyzeV2.analysis.reporter               import Reporter
from stockanalyzeV2.models.sector_models            import (
    AnalysisResult,
    SectorMetrics,
    EDINETEvent,
    ESTATIndicator,
)

logger = logging.getLogger(__name__)


class SectorAnalyzer:
    """
    Top-level orchestrator for the Sector Rotation & Fund Flow analysis pipeline.

    Parameters
    ----------
    market : str
        "JP" (Japan) or "US" (United States).
    previous_metrics : List[SectorMetrics], optional
        Prior-period sector metrics for rotation detection.
        If None, rotation detection is skipped.
    lookback_days : int
        History window for yfinance indicator calculation.
    output_dir : str
        Directory to write CSV exports.
    edinet_api_key : str
        EDINET API key (JP only; skipped if empty).
    estat_app_id : str
        e-STAT application ID (JP only; skipped if empty).
    top_n : int
        Number of top LONG/SHORT candidates to return.
    weights : SCSWeights, optional
        Custom SCS weighting configuration.
    """

    def __init__(
        self,
        market:            str                         = "JP",
        previous_metrics:  Optional[List[SectorMetrics]] = None,
        lookback_days:     int                         = 252,
        output_dir:        str                         = "output",
        edinet_api_key:    str                         = "",
        estat_app_id:      str                         = "",
        top_n:             int                         = 10,
        weights:           Optional[SCSWeights]        = None,
    ):
        if market not in ("JP", "US"):
            raise ValueError(f"market must be 'JP' or 'US', got '{market}'")

        self.market           = market
        self.previous_metrics = previous_metrics or []
        self.output_dir       = output_dir
        self.sector_map       = JP_SECTOR_TICKERS if market == "JP" else US_SECTOR_TICKERS

        # Initialise sub-components
        self.fetcher    = YFinanceFetcher(lookback_days=lookback_days)
        self.calculator = SCSCalculator(weights=weights)
        self.detector   = RotationDetector()
        self.generator  = SignalGenerator(signal_cfg=SignalConfig(top_n=top_n))
        self.reporter   = Reporter(output_dir=output_dir)

        self.edinet = EDINETFetcher(api_key=edinet_api_key) if market == "JP" else None
        self.estat  = ESTATFetcher(app_id=estat_app_id)     if market == "JP" else None

    # ------------------------------------------------------------------
    def run(
        self,
        export_csv:    bool = False,
        print_report:  bool = True,
    ) -> AnalysisResult:
        """
        Execute the full analysis pipeline.

        Parameters
        ----------
        export_csv : bool
            Write sector_metrics and trade_candidates CSV files.
        print_report : bool
            Print a formatted console report.

        Returns
        -------
        AnalysisResult
        """
        logger.info("Starting analysis for market: %s", self.market)

        # ── Step 1: Fetch stock data ───────────────────────────────────
        logger.info("Step 1/6 – Fetching stock data via yfinance …")
        sector_stocks = self.fetcher.fetch_all_sectors(self.sector_map, market=self.market)

        # ── Step 2: Compute SCS ───────────────────────────────────────
        logger.info("Step 2/6 – Computing Sector Composite Scores …")
        sector_metrics = self.calculator.compute_all(sector_stocks, market=self.market)

        # ── Step 3: Detect rotation ───────────────────────────────────
        logger.info("Step 3/6 – Detecting sector rotation …")
        rotation_pairs = self.detector.detect(sector_metrics, self.previous_metrics)

        # ── Step 4: Generate trade signals ────────────────────────────
        logger.info("Step 4/6 – Generating trade signals …")
        long_candidates, short_candidates = self.generator.generate(
            sector_metrics, sector_stocks
        )

        # ── Step 5: Fetch EDINET events (JP only) ─────────────────────
        edinet_events: List[EDINETEvent] = []
        if self.edinet:
            logger.info("Step 5/6 – Fetching EDINET filings …")
            edinet_events = self.edinet.fetch_events()
        else:
            logger.info("Step 5/6 – EDINET skipped (US market or no API key)")

        # ── Step 6: Fetch macro indicators (JP only) ──────────────────
        macro_indicators: List[ESTATIndicator] = []
        if self.estat:
            logger.info("Step 6/6 – Fetching e-STAT macro indicators …")
            macro_indicators = self.estat.fetch_all()
        else:
            logger.info("Step 6/6 – e-STAT skipped (US market or no app ID)")

        # ── Assemble result ───────────────────────────────────────────
        result = AnalysisResult(
            market           = self.market,
            analysis_date    = datetime.now(),
            sector_metrics   = sector_metrics,
            rotation_pairs   = rotation_pairs,
            long_candidates  = long_candidates,
            short_candidates = short_candidates,
            edinet_events    = edinet_events,
            macro_indicators = macro_indicators,
        )

        if print_report:
            self.reporter.print_report(result)

        if export_csv:
            files = self.reporter.export_csv(result)
            for name, path in files.items():
                logger.info("Exported %s → %s", name, path)

        return result
