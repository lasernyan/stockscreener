"""
yfinance-based stock data fetcher.

Fetches OHLCV, technical indicators, and fundamental data for a list of tickers.
All technical indicators (RSI, MACD, SMA, ADX) are computed locally from
price/volume history rather than relying on third-party indicator libraries,
so the only hard dependency is yfinance + pandas.
"""
from __future__ import annotations

import logging
import time
from typing import List, Dict, Optional

import pandas as pd

from stockanalyzeV2.models.sector_models import StockData

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    yf = None  # type: ignore
    _YF_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─── Indicator helpers ────────────────────────────────────────────────────────

def _rsi(close: pd.Series, period: int = 14) -> float:
    """Wilder RSI (last value)."""
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_series = 100 - 100 / (1 + rs)
    return float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Return (macd_line, signal_line) last values."""
    ema_fast   = close.ewm(span=fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """Average Directional Index (last value)."""
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    dm_plus  = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    # Zero out where the other is bigger
    dm_plus  = dm_plus.where(dm_plus  > dm_minus, 0.0)
    dm_minus = dm_minus.where(dm_minus > dm_plus,  0.0)

    atr     = tr.ewm(alpha=1 / period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm( alpha=1 / period, adjust=False).mean() / atr
    di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / atr

    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, float("nan"))
    adx_series = dx.ewm(alpha=1 / period, adjust=False).mean()
    return float(adx_series.iloc[-1]) if not adx_series.empty else 0.0


def _relative_volume(volume: pd.Series, window: int = 20) -> float:
    """Today's volume / 20-day average volume."""
    if len(volume) < window + 1:
        return 1.0
    avg = volume.iloc[-(window + 1):-1].mean()
    return float(volume.iloc[-1] / avg) if avg > 0 else 1.0


# ─── Main fetcher ─────────────────────────────────────────────────────────────

class YFinanceFetcher:
    """
    Fetch stock data for a list of tickers using yfinance.

    Parameters
    ----------
    lookback_days : int
        History window (days) used to compute indicators.
    batch_size : int
        Number of tickers to download in a single yf.download() call.
    delay : float
        Seconds to wait between batches to avoid rate limiting.
    """

    def __init__(
        self,
        lookback_days: int = 252,
        batch_size:    int  = 50,
        delay:         float = 1.0,
    ):
        self.lookback_days = lookback_days
        self.batch_size    = batch_size
        self.delay         = delay

    # ------------------------------------------------------------------
    def fetch_sector(
        self,
        sector:  str,
        tickers: List[str],
        market:  str = "JP",
    ) -> List[StockData]:
        """
        Fetch and compute indicators for every ticker in a sector.
        Skips tickers that fail to download or have insufficient history.
        """
        stocks: List[StockData] = []

        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i : i + self.batch_size]
            stocks.extend(self._fetch_batch(batch, sector))
            if i + self.batch_size < len(tickers):
                time.sleep(self.delay)

        logger.info("Sector %s (%s): fetched %d/%d stocks", sector, market, len(stocks), len(tickers))
        return stocks

    # ------------------------------------------------------------------
    def fetch_all_sectors(
        self,
        sector_map: Dict[str, List[str]],
        market:     str = "JP",
    ) -> Dict[str, List[StockData]]:
        """
        Fetch all sectors defined in sector_map.
        Returns {sector_name: [StockData, ...]}
        """
        result: Dict[str, List[StockData]] = {}
        for sector, tickers in sector_map.items():
            result[sector] = self.fetch_sector(sector, tickers, market)
        return result

    # ------------------------------------------------------------------
    def _fetch_batch(self, tickers: List[str], sector: str) -> List[StockData]:
        """Download OHLCV for a batch and compute per-ticker indicators."""
        if not _YF_AVAILABLE:
            raise ImportError(
                "yfinance is required for stock data fetching. "
                "Install it with: pip install yfinance"
            )
        period = f"{self.lookback_days}d"
        try:
            raw = yf.download(
                tickers,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
        except Exception as exc:
            logger.warning("yfinance download failed for batch %s: %s", tickers, exc)
            return []

        if raw.empty:
            return []

        # yfinance returns multi-level columns when multiple tickers are requested
        is_multi = isinstance(raw.columns, pd.MultiIndex)
        stocks: List[StockData] = []

        for ticker in tickers:
            try:
                stock = self._parse_ticker(raw, ticker, sector, is_multi)
                if stock is not None:
                    stocks.append(stock)
            except Exception as exc:
                logger.debug("Failed to parse %s: %s", ticker, exc)

        return stocks

    # ------------------------------------------------------------------
    def _parse_ticker(
        self,
        raw:       pd.DataFrame,
        ticker:    str,
        sector:    str,
        is_multi:  bool,
    ) -> Optional[StockData]:
        """Extract a single ticker from a (possibly multi-index) DataFrame."""
        if is_multi:
            if ticker not in raw.columns.get_level_values(1):
                return None
            df = raw.xs(ticker, axis=1, level=1).dropna(how="all")
        else:
            df = raw.dropna(how="all")

        if len(df) < 30:
            logger.debug("%s: insufficient history (%d rows)", ticker, len(df))
            return None

        close  = df["Close"]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"]

        latest_close  = float(close.iloc[-1])
        prev_close    = float(close.iloc[-2]) if len(close) > 1 else latest_close
        change_pct    = (latest_close - prev_close) / prev_close * 100 if prev_close != 0 else 0.0

        sma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else latest_close
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else latest_close

        rsi_val        = _rsi(close)
        macd_val, sig_val = _macd(close)
        adx_val        = _adx(high, low, close)
        rel_vol        = _relative_volume(volume)
        latest_vol     = float(volume.iloc[-1])

        # Try to get a human-readable name via yf.Ticker (best-effort)
        name = ticker
        if _YF_AVAILABLE:
            try:
                info = yf.Ticker(ticker).fast_info
                name = getattr(info, "long_name", ticker) or ticker
            except Exception:
                pass

        fund_flow = latest_vol * change_pct / 100.0

        return StockData(
            ticker      = ticker,
            name        = name,
            sector      = sector,
            close       = latest_close,
            change_pct  = change_pct,
            volume      = latest_vol,
            rel_volume  = rel_vol,
            rsi         = rsi_val,
            sma50       = sma50,
            sma200      = sma200,
            macd        = macd_val,
            macd_signal = sig_val,
            adx         = adx_val,
            fund_flow   = fund_flow,
        )
