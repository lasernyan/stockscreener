"""
Sector Composite Score (SCS) Calculator.

Computes a 0-100 score for each sector by aggregating five sub-scores:

  Layer        Weight  Source
  ─────────    ──────  ──────────────────────────────────────────────
  Momentum      25%    Average 1-day change % (normalised)
  Trend         20%    RSI + MACD-divergence composite
  Breadth       25%    % stocks above SMA200 (most important)
  Fund Flow     20%    Total estimated fund flow (normalised)
  ADX           10%    Average directional index (trend clarity)

All sub-scores are normalised to [0, 100] before weighting.
"""
from __future__ import annotations

import math
import statistics
from typing import List, Dict

from stockanalyzeV2.config import (
    SCSWeights,
    BREADTH_VERY_STRONG,
    BREADTH_STRONG,
    BREADTH_NEUTRAL_HI,
    BREADTH_NEUTRAL_LO,
    BREADTH_WEAK,
)
from stockanalyzeV2.models.sector_models import (
    StockData,
    SectorMetrics,
    BreadthLevel,
)


class SCSCalculator:
    """
    Compute SectorMetrics (including SCS) from a list of StockData objects.

    Parameters
    ----------
    weights : SCSWeights
        Configurable weighting for the five sub-scores.
    """

    def __init__(self, weights: SCSWeights = None):
        self.weights = weights or SCSWeights()

    # ------------------------------------------------------------------
    def compute(
        self,
        sector:   str,
        market:   str,
        stocks:   List[StockData],
    ) -> SectorMetrics:
        """
        Compute SectorMetrics for one sector.

        Parameters
        ----------
        sector : str
            Sector label (e.g. "Technology").
        market : str
            "JP" or "US".
        stocks : List[StockData]
            All stocks in this sector with pre-computed indicators.

        Returns
        -------
        SectorMetrics
        """
        metrics = SectorMetrics(
            sector      = sector,
            market      = market,
            stock_count = len(stocks),
            stocks      = [s.ticker for s in stocks],
        )

        if not stocks:
            return metrics

        # ── Raw aggregates ─────────────────────────────────────────────
        changes    = [s.change_pct   for s in stocks]
        rsis       = [s.rsi          for s in stocks]
        macd_diffs = [s.macd - s.macd_signal for s in stocks]
        adxs       = [s.adx          for s in stocks]
        fund_flows = [s.fund_flow     for s in stocks]
        volumes    = [s.volume        for s in stocks]

        metrics.avg_change_pct  = statistics.mean(changes)
        metrics.avg_rsi         = statistics.mean(rsis)
        metrics.avg_macd_diff   = statistics.mean(macd_diffs)
        metrics.avg_adx         = statistics.mean(adxs)
        metrics.total_fund_flow = sum(fund_flows)
        metrics.total_volume    = sum(volumes)

        # ── Market Breadth ─────────────────────────────────────────────
        above_sma50  = [s for s in stocks if s.close > s.sma50  and not math.isnan(s.sma50)]
        above_sma200 = [s for s in stocks if s.close > s.sma200 and not math.isnan(s.sma200)]
        metrics.pct_above_sma50  = len(above_sma50)  / len(stocks)
        metrics.pct_above_sma200 = len(above_sma200) / len(stocks)
        metrics.breadth_level    = self._breadth_level(metrics.pct_above_sma200)

        # ── Sub-scores (0-100) ─────────────────────────────────────────
        metrics.momentum_score  = self._momentum_score(metrics.avg_change_pct)
        metrics.trend_score     = self._trend_score(metrics.avg_rsi, metrics.avg_macd_diff)
        metrics.breadth_score   = metrics.pct_above_sma200 * 100.0
        metrics.fund_flow_score = self._fund_flow_score(metrics.total_fund_flow, metrics.total_volume)
        metrics.adx_score       = self._adx_score(metrics.avg_adx)

        # ── Composite SCS ──────────────────────────────────────────────
        w = self.weights
        metrics.scs = (
            metrics.momentum_score  * w.momentum  +
            metrics.trend_score     * w.trend     +
            metrics.breadth_score   * w.breadth   +
            metrics.fund_flow_score * w.fund_flow +
            metrics.adx_score       * w.adx
        )
        metrics.scs = max(0.0, min(100.0, metrics.scs))

        return metrics

    # ------------------------------------------------------------------
    def compute_all(
        self,
        sector_stocks: Dict[str, List[StockData]],
        market: str,
    ) -> List[SectorMetrics]:
        """Compute SectorMetrics for every sector in the dict."""
        return [
            self.compute(sector, market, stocks)
            for sector, stocks in sector_stocks.items()
        ]

    # ── Sub-score helpers ──────────────────────────────────────────────

    @staticmethod
    def _momentum_score(avg_change_pct: float) -> float:
        """
        Map average daily change % → 0-100 score.
        +2% → ~90, 0% → 50, -2% → ~10 (sigmoid-like clamp).
        """
        # Clamp to [-5, +5] range then scale linearly to [0, 100]
        clamped = max(-5.0, min(5.0, avg_change_pct))
        return (clamped + 5.0) / 10.0 * 100.0

    @staticmethod
    def _trend_score(avg_rsi: float, avg_macd_diff: float) -> float:
        """
        Composite trend score from RSI + MACD divergence.
          RSI 50 → 50 pts, RSI 70 → 100 pts, RSI 30 → 0 pts (linear).
          MACD diff normalised ± 0.5% around zero → ±25 pts bonus/penalty.
        """
        rsi_score  = max(0.0, min(100.0, (avg_rsi - 30.0) / 40.0 * 100.0))
        macd_score = max(0.0, min(100.0, 50.0 + avg_macd_diff * 10.0))
        return (rsi_score * 0.6 + macd_score * 0.4)

    @staticmethod
    def _fund_flow_score(total_flow: float, total_volume: float) -> float:
        """
        Normalise total fund flow to [0, 100].
        Uses flow / volume ratio to make it scale-independent.
        ratio > +0.5% → ~100, ratio < -0.5% → ~0.
        """
        if total_volume == 0:
            return 50.0
        ratio = (total_flow / total_volume) * 100.0   # expressed in %
        clamped = max(-0.5, min(0.5, ratio))
        return (clamped + 0.5) / 1.0 * 100.0

    @staticmethod
    def _adx_score(avg_adx: float) -> float:
        """
        ADX 0-50 → 0-100 score.
        Strong trend (>40) gets near-max score regardless of direction.
        """
        return max(0.0, min(100.0, avg_adx * 2.0))

    @staticmethod
    def _breadth_level(pct_above_sma200: float) -> BreadthLevel:
        """Categorise breadth into 5 levels based on SMA200 ratio."""
        if pct_above_sma200 >= BREADTH_VERY_STRONG:
            return BreadthLevel.VERY_STRONG
        if pct_above_sma200 >= BREADTH_STRONG:
            return BreadthLevel.STRONG
        if pct_above_sma200 >= BREADTH_NEUTRAL_LO:
            return BreadthLevel.NEUTRAL
        if pct_above_sma200 >= BREADTH_WEAK:
            return BreadthLevel.WEAK
        return BreadthLevel.VERY_WEAK
