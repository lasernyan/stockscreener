"""
Sector Rotation Analyzer — Data Fetchers
=========================================
yfinance / EDINET API / e-STAT API の3つのデータソースを統合するパッケージ。

Modules:
    yfinance_fetcher  — 株価・テクニカル指標を yfinance から取得
    edinet_fetcher    — EDINET API からイベント情報（TOB・大量保有・自社株買い）を取得
    estat_fetcher     — e-STAT API から景気動向・生産指数などのマクロ統計を取得
"""

from .yfinance_fetcher import YFinanceFetcher, JP_SECTOR_TICKERS, US_SECTOR_TICKERS
from .edinet_fetcher import EdinetFetcher, EdinetEvent
from .estat_fetcher import EStatFetcher, MacroIndicator

__all__ = [
    "YFinanceFetcher",
    "JP_SECTOR_TICKERS",
    "US_SECTOR_TICKERS",
    "EdinetFetcher",
    "EdinetEvent",
    "EStatFetcher",
    "MacroIndicator",
]
