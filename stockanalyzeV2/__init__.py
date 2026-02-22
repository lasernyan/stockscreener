"""
stockanalyzeV2 — Sector Rotation & Fund Flow Analyzer

Top-down market analysis:
  Market → Sector Strength → Fund Flow → Rotation → Trade Signals

Quick start:
    from stockanalyzeV2.analyzer import SectorAnalyzer
    result = SectorAnalyzer(market="JP").run()
"""
from stockanalyzeV2.analyzer import SectorAnalyzer

__all__ = ["SectorAnalyzer"]
__version__ = "2.0.0"
