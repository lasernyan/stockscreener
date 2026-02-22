"""
Trade Signal Generator.

Scans individual stocks within strong/weak sectors and scores them against
LONG / SHORT entry criteria.

LONG conditions (applied to stocks in strong sectors):
  1. Close > SMA50       (medium-term uptrend)
  2. RSI  40–70          (momentum without overheating)
  3. Rel. Volume > 1.0   (rising interest)
  4. MACD > Signal       (bullish momentum)
  5. ADX  > 20           (trend clarity)
  Rotation inflow bonus  (+BONUS_SCORE) if sector is RISING

SHORT conditions (applied to stocks in weak sectors):
  1. Close < SMA50       (medium-term downtrend)
  2. RSI  30–60          (still room to sell)
  3. MACD < Signal       (bearish momentum)
  4. ADX  > 20           (trend clarity)
  Rotation outflow bonus (+BONUS_SCORE) if sector is FALLING

Each condition adds an equal fraction to the base_score (0-100).
Rotation bonus is additive on top.
"""
from __future__ import annotations

import logging
from typing import List, Dict

from stockanalyzeV2.config import SignalConfig, RotationConfig
from stockanalyzeV2.models.sector_models import (
    StockData,
    SectorMetrics,
    TradeCandidate,
    SignalType,
    RotationStatus,
)

logger = logging.getLogger(__name__)

# Points per satisfied condition (5 conditions → 20 pts each = 100 max base)
_LONG_CONDITIONS  = 5
_SHORT_CONDITIONS = 4
_PTS_PER_LONG  = 100.0 / _LONG_CONDITIONS
_PTS_PER_SHORT = 100.0 / _SHORT_CONDITIONS


class SignalGenerator:
    """
    Generate ranked LONG and SHORT trade candidates from sector data.

    Parameters
    ----------
    signal_cfg : SignalConfig
        Threshold configuration for entry conditions.
    rotation_cfg : RotationConfig
        Bonus score when stock's sector is in rotation.
    """

    def __init__(
        self,
        signal_cfg:   SignalConfig   = None,
        rotation_cfg: RotationConfig = None,
    ):
        self.cfg  = signal_cfg   or SignalConfig()
        self.rcfg = rotation_cfg or RotationConfig()

    # ------------------------------------------------------------------
    def generate(
        self,
        sector_metrics:  List[SectorMetrics],
        sector_stocks:   Dict[str, List[StockData]],
    ) -> tuple[List[TradeCandidate], List[TradeCandidate]]:
        """
        Scan all stocks and produce LONG and SHORT candidate lists.

        Parameters
        ----------
        sector_metrics : List[SectorMetrics]
            Computed sector-level metrics (includes rotation_status).
        sector_stocks : Dict[str, List[StockData]]
            Raw per-stock data keyed by sector name.

        Returns
        -------
        (long_candidates, short_candidates)
            Both lists sorted by total_score descending, capped at top_n.
        """
        # Build lookup: sector → SectorMetrics
        metrics_map: Dict[str, SectorMetrics] = {sm.sector: sm for sm in sector_metrics}

        longs:  List[TradeCandidate] = []
        shorts: List[TradeCandidate] = []

        for sector, stocks in sector_stocks.items():
            sm = metrics_map.get(sector)
            if sm is None:
                continue

            is_strong = sm.scs >= self.cfg.strong_scs_min
            is_weak   = sm.scs <= self.cfg.weak_scs_max

            for stock in stocks:
                if is_strong:
                    candidate = self._score_long(stock, sm)
                    if candidate.base_score > 0:
                        longs.append(candidate)

                if is_weak:
                    candidate = self._score_short(stock, sm)
                    if candidate.base_score > 0:
                        shorts.append(candidate)

        longs.sort( key=lambda c: c.total_score, reverse=True)
        shorts.sort(key=lambda c: c.total_score, reverse=True)

        top_longs  = longs[: self.cfg.top_n]
        top_shorts = shorts[: self.cfg.top_n]

        logger.info(
            "Signal generation: %d LONG candidates, %d SHORT candidates",
            len(top_longs), len(top_shorts),
        )
        return top_longs, top_shorts

    # ------------------------------------------------------------------
    def _score_long(self, stock: StockData, sm: SectorMetrics) -> TradeCandidate:
        """Score a single stock against LONG entry conditions."""
        c = self.cfg

        cond_above_sma50   = stock.close > stock.sma50
        cond_rsi_range     = c.long_rsi_min <= stock.rsi <= c.long_rsi_max
        cond_rel_volume    = stock.rel_volume >= c.long_rel_vol_min
        cond_macd_bullish  = stock.macd > stock.macd_signal
        cond_adx           = stock.adx >= c.long_adx_min

        conditions = [cond_above_sma50, cond_rsi_range, cond_rel_volume, cond_macd_bullish, cond_adx]
        base_score = sum(_PTS_PER_LONG for cond in conditions if cond)

        in_rotation = sm.rotation_status == RotationStatus.RISING
        rotation_bonus = self.rcfg.bonus_score if in_rotation else 0.0
        total_score    = base_score + rotation_bonus

        return TradeCandidate(
            ticker        = stock.ticker,
            name          = stock.name,
            sector        = stock.sector,
            market        = sm.market,
            signal_type   = SignalType.LONG,
            close         = stock.close,
            change_pct    = stock.change_pct,
            volume        = stock.volume,
            rel_volume    = stock.rel_volume,
            rsi           = stock.rsi,
            sma50         = stock.sma50,
            sma200        = stock.sma200,
            macd          = stock.macd,
            macd_signal   = stock.macd_signal,
            adx           = stock.adx,
            base_score    = base_score,
            rotation_bonus= rotation_bonus,
            total_score   = total_score,
            above_sma50       = cond_above_sma50,
            macd_bullish      = cond_macd_bullish,
            rsi_in_range      = cond_rsi_range,
            volume_elevated   = cond_rel_volume,
            adx_trending      = cond_adx,
            in_rotation_sector= in_rotation,
            sector_scs        = sm.scs,
            sector_rotation   = sm.rotation_status,
        )

    # ------------------------------------------------------------------
    def _score_short(self, stock: StockData, sm: SectorMetrics) -> TradeCandidate:
        """Score a single stock against SHORT entry conditions."""
        c = self.cfg

        cond_below_sma50   = stock.close < stock.sma50
        cond_rsi_range     = c.short_rsi_min <= stock.rsi <= c.short_rsi_max
        cond_macd_bearish  = stock.macd < stock.macd_signal
        cond_adx           = stock.adx >= c.short_adx_min

        conditions = [cond_below_sma50, cond_rsi_range, cond_macd_bearish, cond_adx]
        base_score = sum(_PTS_PER_SHORT for cond in conditions if cond)

        in_rotation = sm.rotation_status == RotationStatus.FALLING
        rotation_bonus = self.rcfg.bonus_score if in_rotation else 0.0
        total_score    = base_score + rotation_bonus

        return TradeCandidate(
            ticker        = stock.ticker,
            name          = stock.name,
            sector        = stock.sector,
            market        = sm.market,
            signal_type   = SignalType.SHORT,
            close         = stock.close,
            change_pct    = stock.change_pct,
            volume        = stock.volume,
            rel_volume    = stock.rel_volume,
            rsi           = stock.rsi,
            sma50         = stock.sma50,
            sma200        = stock.sma200,
            macd          = stock.macd,
            macd_signal   = stock.macd_signal,
            adx           = stock.adx,
            base_score    = base_score,
            rotation_bonus= rotation_bonus,
            total_score   = total_score,
            above_sma50       = not cond_below_sma50,
            macd_bullish      = not cond_macd_bearish,
            rsi_in_range      = cond_rsi_range,
            volume_elevated   = False,    # not a SHORT criterion
            adx_trending      = cond_adx,
            in_rotation_sector= in_rotation,
            sector_scs        = sm.scs,
            sector_rotation   = sm.rotation_status,
        )
