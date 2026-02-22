"""
Data models for Sector Rotation & Fund Flow Analyzer
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class BreadthLevel(Enum):
    VERY_STRONG = "Very Strong"   # >80% above SMA200
    STRONG      = "Strong"        # 60-80%
    NEUTRAL     = "Neutral"       # 40-60%
    WEAK        = "Weak"          # 20-40%
    VERY_WEAK   = "Very Weak"     # <20%


class RotationStatus(Enum):
    RISING  = "Rising"
    FALLING = "Falling"
    STABLE  = "Stable"


@dataclass
class StockData:
    """Raw per-stock data fetched from yfinance."""
    ticker:         str
    name:           str
    sector:         str
    close:          float
    change_pct:     float           # 1-day change %
    volume:         float
    rel_volume:     float           # volume / avg_volume_20d
    rsi:            float
    sma50:          float
    sma200:         float
    macd:           float
    macd_signal:    float
    adx:            float
    market_cap:     float = 0.0
    fund_flow:      float = 0.0     # volume * change_pct / 100


@dataclass
class SectorMetrics:
    """Aggregated sector-level metrics from StockData list."""
    sector:         str
    market:         str             # "JP" or "US"
    snapshot_date:  datetime = field(default_factory=datetime.now)

    # Raw aggregated values
    avg_change_pct:     float = 0.0
    avg_rsi:            float = 50.0
    avg_macd_diff:      float = 0.0   # MACD - Signal average
    avg_adx:            float = 0.0
    total_fund_flow:    float = 0.0
    total_volume:       float = 0.0

    # Breadth
    pct_above_sma50:    float = 0.0   # 0-1
    pct_above_sma200:   float = 0.0   # 0-1 (most important)
    breadth_level:      BreadthLevel = BreadthLevel.NEUTRAL

    # SCS sub-scores (0-100 each)
    momentum_score:     float = 0.0
    trend_score:        float = 0.0
    breadth_score:      float = 0.0
    fund_flow_score:    float = 0.0
    adx_score:          float = 0.0

    # Final composite score
    scs:                float = 0.0   # 0-100

    # Rotation
    rotation_status:    RotationStatus = RotationStatus.STABLE
    scs_change:         float = 0.0   # vs previous snapshot

    # Stock count
    stock_count:        int = 0
    stocks:             List[str] = field(default_factory=list)


@dataclass
class TradeCandidate:
    """Individual stock flagged as a trade candidate."""
    ticker:         str
    name:           str
    sector:         str
    market:         str
    signal_type:    SignalType

    # Price data
    close:          float
    change_pct:     float
    volume:         float
    rel_volume:     float

    # Technical indicators
    rsi:            float
    sma50:          float
    sma200:         float
    macd:           float
    macd_signal:    float
    adx:            float

    # Scoring
    base_score:     float = 0.0
    rotation_bonus: float = 0.0
    total_score:    float = 0.0

    # Condition flags
    above_sma50:        bool = False
    macd_bullish:       bool = False
    rsi_in_range:       bool = False
    volume_elevated:    bool = False
    adx_trending:       bool = False
    in_rotation_sector: bool = False

    # Sector context
    sector_scs:         float = 0.0
    sector_rotation:    RotationStatus = RotationStatus.STABLE


@dataclass
class RotationPair:
    """A detected sector rotation: money flowing from one sector to another."""
    from_sector: str
    to_sector:   str
    from_scs_change: float
    to_scs_change:   float
    market:      str


@dataclass
class EDINETEvent:
    """Event parsed from EDINET filing."""
    doc_type:       str       # "tender_offer" | "large_shareholding" | "share_repurchase"
    filer_name:     str
    filing_date:    str
    ticker:         Optional[str] = None
    description:    str = ""
    doc_id:         str = ""


@dataclass
class ESTATIndicator:
    """Macro indicator from e-STAT."""
    series_name:    str
    series_id:      str
    latest_value:   Optional[float] = None
    prev_value:     Optional[float] = None
    change:         Optional[float] = None
    unit:           str = ""
    reference_date: str = ""


@dataclass
class AnalysisResult:
    """Top-level result object returned by the full analysis pipeline."""
    market:             str
    analysis_date:      datetime
    sector_metrics:     List[SectorMetrics]
    rotation_pairs:     List[RotationPair]
    long_candidates:    List[TradeCandidate]
    short_candidates:   List[TradeCandidate]
    edinet_events:      List[EDINETEvent]
    macro_indicators:   List[ESTATIndicator]

    def top_sectors(self, n: int = 3) -> List[SectorMetrics]:
        return sorted(self.sector_metrics, key=lambda s: s.scs, reverse=True)[:n]

    def weak_sectors(self, n: int = 3) -> List[SectorMetrics]:
        return sorted(self.sector_metrics, key=lambda s: s.scs)[:n]
