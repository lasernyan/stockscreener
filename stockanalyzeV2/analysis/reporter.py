"""
Console and CSV reporter for analysis results.

Outputs:
  - Rich terminal tables (via tabulate if available, plain text fallback)
  - CSV exports for sector_metrics and trade_candidates
"""
from __future__ import annotations

import csv
import logging
import os
from datetime import datetime
from typing import List

from stockanalyzeV2.models.sector_models import (
    AnalysisResult,
    SectorMetrics,
    TradeCandidate,
    RotationPair,
    EDINETEvent,
    ESTATIndicator,
)

logger = logging.getLogger(__name__)

# Optional tabulate for pretty tables
try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


def _fmt(value: float, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.{decimals}f}" if value < 0 or value > 0 else f"{value:.{decimals}f}"


class Reporter:
    """
    Format and export AnalysisResult to console and CSV files.

    Parameters
    ----------
    output_dir : str
        Directory where CSV files are written.
    """

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def print_report(self, result: AnalysisResult) -> None:
        """Print a full human-readable report to stdout."""
        now_str = result.analysis_date.strftime("%Y-%m-%d %H:%M")
        print(f"\n{'='*70}")
        print(f"  Sector Rotation & Fund Flow Analysis  [{result.market}]  {now_str}")
        print(f"{'='*70}\n")

        self._print_sector_ranking(result.sector_metrics)
        self._print_rotation_pairs(result.rotation_pairs)
        self._print_trade_candidates("LONG", result.long_candidates)
        self._print_trade_candidates("SHORT", result.short_candidates)
        self._print_edinet_events(result.edinet_events)
        self._print_macro_indicators(result.macro_indicators)

    # ------------------------------------------------------------------
    def _print_sector_ranking(self, metrics: List[SectorMetrics]) -> None:
        ranked = sorted(metrics, key=lambda s: s.scs, reverse=True)
        headers = ["Rank", "Sector", "SCS", "Momentum", "Trend", "Breadth", "FundFlow", "ADX", "Breadth%", "Rotation"]
        rows = []
        for i, sm in enumerate(ranked, 1):
            rotation_symbol = {
                "Rising":  "↑ RISING",
                "Falling": "↓ FALLING",
                "Stable":  "─",
            }.get(sm.rotation_status.value, "─")

            rows.append([
                i,
                sm.sector,
                f"{sm.scs:.1f}",
                f"{sm.momentum_score:.1f}",
                f"{sm.trend_score:.1f}",
                f"{sm.breadth_score:.1f}",
                f"{sm.fund_flow_score:.1f}",
                f"{sm.adx_score:.1f}",
                f"{sm.pct_above_sma200*100:.0f}%",
                rotation_symbol,
            ])

        print("── Sector Composite Scores (SCS) ─────────────────────────────────────")
        if _HAS_TABULATE:
            print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        else:
            print("\t".join(headers))
            for row in rows:
                print("\t".join(str(c) for c in row))
        print()

    # ------------------------------------------------------------------
    def _print_rotation_pairs(self, pairs: List[RotationPair]) -> None:
        if not pairs:
            print("── Rotation Pairs ─── (no rotation detected)\n")
            return

        print("── Rotation Pairs ─────────────────────────────────────────────────────")
        for p in pairs:
            print(
                f"  [{p.market}]  {p.from_sector}  →  {p.to_sector}"
                f"   (Δ{p.from_scs_change:+.1f} / Δ{p.to_scs_change:+.1f})"
            )
        print()

    # ------------------------------------------------------------------
    def _print_trade_candidates(self, label: str, candidates: List[TradeCandidate]) -> None:
        if not candidates:
            print(f"── {label} Candidates ─── (none)\n")
            return

        print(f"── {label} Candidates ──────────────────────────────────────────────────")
        headers = ["#", "Ticker", "Sector", "Score", "SCS", "RSI", "RelVol", "ADX", "Conditions"]
        rows = []
        for i, c in enumerate(candidates, 1):
            cond_flags = []
            if c.above_sma50:       cond_flags.append("SMA50")
            if c.rsi_in_range:      cond_flags.append("RSI")
            if c.volume_elevated:   cond_flags.append("VOL")
            if c.macd_bullish:      cond_flags.append("MACD")
            if c.adx_trending:      cond_flags.append("ADX")
            if c.in_rotation_sector:cond_flags.append("ROT+")

            rows.append([
                i,
                c.ticker,
                c.sector,
                f"{c.total_score:.1f}",
                f"{c.sector_scs:.1f}",
                f"{c.rsi:.1f}",
                f"{c.rel_volume:.2f}x",
                f"{c.adx:.1f}",
                " ".join(cond_flags),
            ])

        if _HAS_TABULATE:
            print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        else:
            print("\t".join(headers))
            for row in rows:
                print("\t".join(str(c) for c in row))
        print()

    # ------------------------------------------------------------------
    def _print_edinet_events(self, events: List[EDINETEvent]) -> None:
        if not events:
            return
        print("── EDINET Filings ──────────────────────────────────────────────────────")
        for ev in events[:20]:    # show at most 20
            ticker_str = f"  [{ev.ticker}]" if ev.ticker else ""
            print(f"  {ev.filing_date}  {ev.doc_type:<20} {ev.filer_name}{ticker_str}")
        if len(events) > 20:
            print(f"  ... and {len(events) - 20} more")
        print()

    # ------------------------------------------------------------------
    def _print_macro_indicators(self, indicators: List[ESTATIndicator]) -> None:
        if not indicators:
            return
        print("── Macro Indicators (e-STAT) ───────────────────────────────────────────")
        for ind in indicators:
            change_str = f"  Δ{ind.change:+.2f}" if ind.change is not None else ""
            val_str    = f"{ind.latest_value:.2f}" if ind.latest_value is not None else "N/A"
            print(f"  {ind.series_name:<30}  {val_str}{change_str}  [{ind.reference_date}]")
        print()

    # ------------------------------------------------------------------
    def export_csv(self, result: AnalysisResult) -> dict:
        """
        Export sector metrics and trade candidates to CSV files.

        Returns
        -------
        dict : {report_name: file_path}
        """
        timestamp = result.analysis_date.strftime("%Y%m%d_%H%M")
        files = {}

        # Sector metrics
        sm_path = os.path.join(self.output_dir, f"sector_metrics_{result.market}_{timestamp}.csv")
        self._export_sector_metrics(result.sector_metrics, sm_path)
        files["sector_metrics"] = sm_path

        # Trade candidates
        all_candidates = result.long_candidates + result.short_candidates
        tc_path = os.path.join(self.output_dir, f"trade_candidates_{result.market}_{timestamp}.csv")
        self._export_trade_candidates(all_candidates, tc_path)
        files["trade_candidates"] = tc_path

        return files

    # ------------------------------------------------------------------
    @staticmethod
    def _export_sector_metrics(metrics: List[SectorMetrics], path: str) -> None:
        fieldnames = [
            "sector", "market", "scs", "momentum_score", "trend_score",
            "breadth_score", "fund_flow_score", "adx_score",
            "pct_above_sma200", "pct_above_sma50", "avg_change_pct",
            "avg_rsi", "avg_adx", "total_fund_flow", "stock_count",
            "rotation_status", "scs_change", "breadth_level",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for sm in metrics:
                writer.writerow({
                    "sector":           sm.sector,
                    "market":           sm.market,
                    "scs":              f"{sm.scs:.2f}",
                    "momentum_score":   f"{sm.momentum_score:.2f}",
                    "trend_score":      f"{sm.trend_score:.2f}",
                    "breadth_score":    f"{sm.breadth_score:.2f}",
                    "fund_flow_score":  f"{sm.fund_flow_score:.2f}",
                    "adx_score":        f"{sm.adx_score:.2f}",
                    "pct_above_sma200": f"{sm.pct_above_sma200:.4f}",
                    "pct_above_sma50":  f"{sm.pct_above_sma50:.4f}",
                    "avg_change_pct":   f"{sm.avg_change_pct:.4f}",
                    "avg_rsi":          f"{sm.avg_rsi:.2f}",
                    "avg_adx":          f"{sm.avg_adx:.2f}",
                    "total_fund_flow":  f"{sm.total_fund_flow:.2f}",
                    "stock_count":      sm.stock_count,
                    "rotation_status":  sm.rotation_status.value,
                    "scs_change":       f"{sm.scs_change:.2f}",
                    "breadth_level":    sm.breadth_level.value,
                })
        logger.info("Exported sector metrics → %s", path)

    # ------------------------------------------------------------------
    @staticmethod
    def _export_trade_candidates(candidates: List[TradeCandidate], path: str) -> None:
        fieldnames = [
            "signal_type", "ticker", "name", "sector", "market",
            "total_score", "base_score", "rotation_bonus",
            "close", "change_pct", "rsi", "rel_volume", "adx",
            "sma50", "sma200", "macd", "macd_signal",
            "above_sma50", "macd_bullish", "rsi_in_range",
            "volume_elevated", "adx_trending", "in_rotation_sector",
            "sector_scs", "sector_rotation",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in candidates:
                writer.writerow({
                    "signal_type":        c.signal_type.value,
                    "ticker":             c.ticker,
                    "name":               c.name,
                    "sector":             c.sector,
                    "market":             c.market,
                    "total_score":        f"{c.total_score:.2f}",
                    "base_score":         f"{c.base_score:.2f}",
                    "rotation_bonus":     f"{c.rotation_bonus:.2f}",
                    "close":              f"{c.close:.2f}",
                    "change_pct":         f"{c.change_pct:.4f}",
                    "rsi":                f"{c.rsi:.2f}",
                    "rel_volume":         f"{c.rel_volume:.4f}",
                    "adx":                f"{c.adx:.2f}",
                    "sma50":              f"{c.sma50:.2f}",
                    "sma200":             f"{c.sma200:.2f}",
                    "macd":               f"{c.macd:.4f}",
                    "macd_signal":        f"{c.macd_signal:.4f}",
                    "above_sma50":        c.above_sma50,
                    "macd_bullish":       c.macd_bullish,
                    "rsi_in_range":       c.rsi_in_range,
                    "volume_elevated":    c.volume_elevated,
                    "adx_trending":       c.adx_trending,
                    "in_rotation_sector": c.in_rotation_sector,
                    "sector_scs":         f"{c.sector_scs:.2f}",
                    "sector_rotation":    c.sector_rotation.value,
                })
        logger.info("Exported trade candidates → %s", path)
