"""
Configuration for Sector Rotation & Fund Flow Analyzer
"""
from dataclasses import dataclass, field
from typing import List, Dict


# ─── API Keys (set via environment variables or edit here) ───────────────────
EDINET_API_KEY = ""       # Set via env: EDINET_API_KEY
ESTAT_API_KEY  = ""       # Set via env: ESTAT_APP_ID


# ─── Japan Sector → Ticker mapping (TOPIX-based ETFs or representative stocks) ─
JP_SECTOR_TICKERS: Dict[str, List[str]] = {
    "Technology":       ["6758.T", "6861.T", "6954.T", "4063.T", "9984.T"],
    "Financials":       ["8306.T", "8316.T", "8411.T", "8001.T", "8031.T"],
    "Auto & Transport": ["7203.T", "7267.T", "7201.T", "9101.T", "9202.T"],
    "Healthcare":       ["4502.T", "4519.T", "4523.T", "4568.T", "4578.T"],
    "Industrials":      ["6301.T", "6302.T", "7011.T", "7013.T", "6501.T"],
    "Consumer Disc":    ["9983.T", "2802.T", "2914.T", "7751.T", "4901.T"],
    "Real Estate":      ["8801.T", "8802.T", "8803.T", "3003.T", "3283.T"],
    "Energy":           ["5020.T", "5019.T", "1605.T"],
    "Materials":        ["5401.T", "5411.T", "3407.T", "4004.T", "4005.T"],
    "Utilities":        ["9501.T", "9502.T", "9503.T"],
    "Communication":    ["9432.T", "9433.T", "9434.T", "4755.T"],
}

# ─── US Sector → Ticker mapping ───────────────────────────────────────────────
US_SECTOR_TICKERS: Dict[str, List[str]] = {
    "Technology":       ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"],
    "Financials":       ["JPM", "BAC", "WFC", "GS", "MS"],
    "Healthcare":       ["UNH", "JNJ", "LLY", "ABBV", "MRK"],
    "Consumer Disc":    ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "Industrials":      ["GE", "CAT", "HON", "UPS", "DE"],
    "Communication":    ["GOOGL", "META", "NFLX", "DIS", "T"],
    "Consumer Staples": ["WMT", "PG", "KO", "PEP", "COST"],
    "Energy":           ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Materials":        ["LIN", "APD", "ECL", "NEM", "FCX"],
    "Real Estate":      ["PLD", "AMT", "EQIX", "SPG", "WELL"],
    "Utilities":        ["NEE", "DUK", "SO", "D", "AEP"],
}


# ─── SCS Weight configuration ─────────────────────────────────────────────────
@dataclass
class SCSWeights:
    momentum: float = 0.25
    trend:    float = 0.20
    breadth:  float = 0.25
    fund_flow: float = 0.20
    adx:      float = 0.10


# ─── Rotation detection thresholds ────────────────────────────────────────────
@dataclass
class RotationConfig:
    rising_threshold:  float = 3.0   # SCS change >= +3 → Rising
    falling_threshold: float = -3.0  # SCS change <= -3 → Falling
    bonus_score:       float = 5.0   # Bonus added to signal score for rotation stocks


# ─── Trade signal thresholds ──────────────────────────────────────────────────
@dataclass
class SignalConfig:
    # LONG conditions
    long_rsi_min:     float = 40.0
    long_rsi_max:     float = 70.0
    long_rel_vol_min: float = 1.0
    long_adx_min:     float = 20.0

    # SHORT conditions
    short_rsi_min:    float = 30.0
    short_rsi_max:    float = 60.0
    short_adx_min:    float = 20.0

    # SCS threshold to qualify as strong/weak sector
    strong_scs_min:   float = 60.0
    weak_scs_max:     float = 40.0

    # Top N candidates to return
    top_n:            int   = 10


# ─── Breadth thresholds ───────────────────────────────────────────────────────
BREADTH_VERY_STRONG = 0.80   # >80% above SMA200
BREADTH_STRONG      = 0.60
BREADTH_NEUTRAL_HI  = 0.60
BREADTH_NEUTRAL_LO  = 0.40
BREADTH_WEAK        = 0.20   # <20% above SMA200

# ─── e-STAT series IDs of interest ───────────────────────────────────────────
# These are examples – replace with real statsDataId values from e-Stat
ESTAT_SERIES = {
    "IIP_Manufacturing":        "0003191203",  # 鉱工業生産指数（製造工業）
    "Machinery_Orders":         "0003191418",  # 機械受注統計
    "Consumer_Confidence":      "0003212935",  # 消費者態度指数
    "Unemployment_Rate":        "0003031118",  # 完全失業率
}

# ─── EDINET document type codes ───────────────────────────────────────────────
EDINET_DOC_TYPES = {
    "tender_offer":         "030",    # 公開買付届出書
    "large_shareholding":   "040",    # 大量保有報告書
    "share_repurchase":     "020",    # 自己株式取得状況報告書
}
