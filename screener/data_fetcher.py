#!/usr/bin/env python3
"""
Multi-Asset API Data Fetcher
============================
yfinance を使って株式・ETF・暗号資産・コモディティ・FX のデータを
TradingView CSV 互換フォーマットで screener/snapshots/ に保存する。

Usage:
  python3 screener/data_fetcher.py                  # 全アセット取得
  python3 screener/data_fetcher.py --market US       # 米国株のみ
  python3 screener/data_fetcher.py --market CRYPTO   # 暗号資産のみ
  python3 screener/data_fetcher.py --market ETF      # ETFのみ
  python3 screener/data_fetcher.py --market ALL      # 全アセット (default)
  python3 screener/data_fetcher.py --list-markets    # 利用可能市場一覧
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

# ============================================================
# アセット定義
# ============================================================

# 米国セクターETF (SPDR)
US_SECTOR_ETFS = {
    "XLK":  ("Technology Select Sector SPDR",       "Technology",          "Technology ETF"),
    "XLF":  ("Financial Select Sector SPDR",         "Financial Services",  "Financial ETF"),
    "XLV":  ("Health Care Select Sector SPDR",       "Healthcare",          "Healthcare ETF"),
    "XLE":  ("Energy Select Sector SPDR",            "Energy",              "Energy ETF"),
    "XLI":  ("Industrial Select Sector SPDR",        "Industrials",         "Industrial ETF"),
    "XLY":  ("Consumer Discret Select Sector SPDR",  "Consumer Cyclical",   "Consumer ETF"),
    "XLP":  ("Consumer Staples Select Sector SPDR",  "Consumer Defensive",  "Consumer ETF"),
    "XLU":  ("Utilities Select Sector SPDR",         "Utilities",           "Utilities ETF"),
    "XLRE": ("Real Estate Select Sector SPDR",       "Real Estate",         "REIT ETF"),
    "XLB":  ("Materials Select Sector SPDR",         "Basic Materials",     "Materials ETF"),
    "XLC":  ("Comm Services Select Sector SPDR",     "Communication Services", "Telecom ETF"),
    # ブロード市場
    "SPY":  ("SPDR S&P500 ETF",                     "Broad Market",        "Index ETF"),
    "QQQ":  ("Invesco QQQ Trust",                    "Technology",          "Index ETF"),
    "IWM":  ("iShares Russell 2000 ETF",             "Broad Market",        "Index ETF"),
    "DIA":  ("SPDR Dow Jones Industrial Average ETF","Broad Market",        "Index ETF"),
    "VTI":  ("Vanguard Total Stock Market ETF",      "Broad Market",        "Index ETF"),
    # 国際
    "EFA":  ("iShares MSCI EAFE ETF",               "International",        "International ETF"),
    "EEM":  ("iShares MSCI Emerging Markets ETF",   "International",        "Emerging Markets ETF"),
    "EWJ":  ("iShares MSCI Japan ETF",              "Japan",                "Japan ETF"),
}

# 債券ETF
BOND_ETFS = {
    "TLT":  ("iShares 20+ Year Treasury Bond ETF",  "Fixed Income",  "Long-Term Treasury"),
    "IEF":  ("iShares 7-10 Year Treasury Bond ETF", "Fixed Income",  "Mid-Term Treasury"),
    "SHY":  ("iShares 1-3 Year Treasury Bond ETF",  "Fixed Income",  "Short-Term Treasury"),
    "LQD":  ("iShares iBoxx Investment Grade Corp",  "Fixed Income",  "Corporate Bond"),
    "HYG":  ("iShares iBoxx High Yield Corp Bond",   "Fixed Income",  "High Yield Bond"),
    "BND":  ("Vanguard Total Bond Market ETF",       "Fixed Income",  "Broad Bond"),
    "EMB":  ("iShares JP Morgan EM Bond ETF",        "Fixed Income",  "Emerging Market Bond"),
    "TIP":  ("iShares TIPS Bond ETF",               "Fixed Income",  "Inflation-Protected"),
    # 金利指標
    "^TNX": ("10-Year Treasury Yield",              "Fixed Income",  "Interest Rate"),
    "^FVX": ("5-Year Treasury Yield",               "Fixed Income",  "Interest Rate"),
    "^IRX": ("13-Week Treasury Bill Rate",          "Fixed Income",  "Interest Rate"),
}

# コモディティETF & 先物
COMMODITY_TICKERS = {
    "GLD":  ("SPDR Gold Shares",                    "Commodities", "Gold ETF"),
    "SLV":  ("iShares Silver Trust",                "Commodities", "Silver ETF"),
    "GDX":  ("VanEck Gold Miners ETF",              "Commodities", "Gold Miners ETF"),
    "GDXJ": ("VanEck Junior Gold Miners ETF",       "Commodities", "Gold Miners ETF"),
    "SIL":  ("Global X Silver Miners ETF",          "Commodities", "Silver Miners ETF"),
    "USO":  ("United States Oil Fund",              "Commodities", "Crude Oil ETF"),
    "UNG":  ("United States Natural Gas Fund",      "Commodities", "Natural Gas ETF"),
    "PDBC": ("Invesco Optimum Yield Diversified Commodity","Commodities","Broad Commodity ETF"),
    "DBC":  ("Invesco DB Commodity Index Tracking", "Commodities", "Broad Commodity ETF"),
    # 先物 (yfinance suffix =F)
    "GC=F": ("Gold Futures",          "Commodities", "Gold Futures"),
    "SI=F": ("Silver Futures",        "Commodities", "Silver Futures"),
    "CL=F": ("Crude Oil WTI Futures", "Commodities", "Crude Oil Futures"),
    "NG=F": ("Natural Gas Futures",   "Commodities", "Natural Gas Futures"),
    "HG=F": ("Copper Futures",        "Commodities", "Copper Futures"),
    "ZW=F": ("Wheat Futures",         "Commodities", "Grain Futures"),
    "ZC=F": ("Corn Futures",          "Commodities", "Grain Futures"),
    "ZS=F": ("Soybean Futures",       "Commodities", "Grain Futures"),
}

# 暗号資産 (yfinance USD建て)
CRYPTO_TICKERS = {
    "BTC-USD": ("Bitcoin",        "Cryptocurrency", "Layer 1"),
    "ETH-USD": ("Ethereum",       "Cryptocurrency", "Layer 1"),
    "BNB-USD": ("BNB",            "Cryptocurrency", "Exchange Token"),
    "SOL-USD": ("Solana",         "Cryptocurrency", "Layer 1"),
    "XRP-USD": ("XRP",            "Cryptocurrency", "Payment"),
    "ADA-USD": ("Cardano",        "Cryptocurrency", "Layer 1"),
    "AVAX-USD":("Avalanche",      "Cryptocurrency", "Layer 1"),
    "DOGE-USD":("Dogecoin",       "Cryptocurrency", "Meme Coin"),
    "DOT-USD": ("Polkadot",       "Cryptocurrency", "Layer 0"),
    "LINK-USD":("Chainlink",      "Cryptocurrency", "Oracle"),
    "UNI-USD": ("Uniswap",        "Cryptocurrency", "DeFi"),
    "AAVE-USD":("Aave",           "Cryptocurrency", "DeFi"),
    "MATIC-USD":("Polygon",       "Cryptocurrency", "Layer 2"),
    "LTC-USD": ("Litecoin",       "Cryptocurrency", "Payment"),
    # 暗号資産ETF
    "IBIT":    ("iShares Bitcoin Trust", "Cryptocurrency", "Bitcoin ETF"),
    "FBTC":    ("Fidelity Wise Origin Bitcoin Fund", "Cryptocurrency", "Bitcoin ETF"),
}

# FX (yfinance =X suffix)
FX_TICKERS = {
    "USDJPY=X": ("USD/JPY", "Foreign Exchange", "Major Pair"),
    "EURUSD=X": ("EUR/USD", "Foreign Exchange", "Major Pair"),
    "GBPUSD=X": ("GBP/USD", "Foreign Exchange", "Major Pair"),
    "AUDUSD=X": ("AUD/USD", "Foreign Exchange", "Commodity Currency"),
    "USDCAD=X": ("USD/CAD", "Foreign Exchange", "Commodity Currency"),
    "USDCHF=X": ("USD/CHF", "Foreign Exchange", "Safe Haven"),
    "NZDUSD=X": ("NZD/USD", "Foreign Exchange", "Commodity Currency"),
    "USDCNH=X": ("USD/CNH", "Foreign Exchange", "EM Currency"),
    "USDINR=X": ("USD/INR", "Foreign Exchange", "EM Currency"),
    "USDBRL=X": ("USD/BRL", "Foreign Exchange", "EM Currency"),
    "DX-Y.NYB": ("US Dollar Index", "Foreign Exchange", "Dollar Index"),
}

# 全アセット群のマッピング
MARKET_CONFIGS = {
    "ETF":       US_SECTOR_ETFS,
    "BOND":      BOND_ETFS,
    "COMMODITY": COMMODITY_TICKERS,
    "CRYPTO":    CRYPTO_TICKERS,
    "FX":        FX_TICKERS,
}

# ============================================================
# テクニカル指標計算
# ============================================================

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(series: pd.Series, fast=12, slow=26, signal=9):
    """MACD と Signal line"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index"""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    dm_plus = high.diff().clip(lower=0)
    dm_minus = (-low.diff()).clip(lower=0)
    # 実際に大きい方だけ残す
    mask = dm_plus < dm_minus
    dm_plus[mask] = 0
    mask2 = dm_minus < dm_plus
    dm_minus[mask2] = 0

    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    di_plus = 100 * dm_plus.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
    dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx


def calc_indicators(hist: pd.DataFrame) -> dict:
    """OHLCV から全テクニカル指標を計算して dict で返す"""
    close = hist["Close"]
    volume = hist["Volume"]

    rsi_series = calc_rsi(close)
    macd_line, signal_line = calc_macd(close)
    adx_series = calc_adx(hist["High"], hist["Low"], close)

    sma20  = close.rolling(20).mean()
    sma50  = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()

    avg_vol_20 = volume.rolling(20).mean()
    rel_vol = volume / avg_vol_20.replace(0, np.nan)

    # 最新値を取得（NaN は None に変換）
    def last(s):
        v = s.dropna()
        return float(v.iloc[-1]) if len(v) else None

    prev_close = close.iloc[-2] if len(close) >= 2 else None
    curr_close = last(close)
    change_pct = ((curr_close / prev_close) - 1) * 100 if (curr_close and prev_close and prev_close != 0) else 0.0

    return {
        "close":       curr_close,
        "change_pct":  round(change_pct, 4),
        "volume":      int(volume.iloc[-1]) if not pd.isna(volume.iloc[-1]) else 0,
        "rel_volume":  round(last(rel_vol) or 1.0, 3),
        "rsi":         round(last(rsi_series) or 50.0, 2),
        "macd":        round(last(macd_line) or 0.0, 6),
        "macd_signal": round(last(signal_line) or 0.0, 6),
        "adx":         round(last(adx_series) or 20.0, 2),
        "sma20":       round(last(sma20) or curr_close or 0.0, 4),
        "sma50":       round(last(sma50) or curr_close or 0.0, 4),
        "sma200":      round(last(sma200) or curr_close or 0.0, 4),
    }


# ============================================================
# フェッチ & CSV 生成
# ============================================================

def fetch_asset_group(
    ticker_map: dict,
    period: str = "1y",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    ticker_map: {ticker: (name, sector, industry)}
    Returns DataFrame in TradingView-compatible format.
    """
    rows = []
    tickers = list(ticker_map.keys())

    if verbose:
        print(f"  Downloading {len(tickers)} tickers...", end="", flush=True)

    # バッチダウンロード（高速）
    try:
        data = yf.download(
            tickers,
            period=period,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as e:
        print(f"\n  [ERROR] yfinance download failed: {e}")
        return pd.DataFrame()

    if verbose:
        print(" done.")

    for ticker, (name, sector, industry) in ticker_map.items():
        try:
            # 単一 vs マルチティッカーで列構造が異なる
            if len(tickers) == 1:
                hist = data
            else:
                # MultiIndex: (column, ticker)
                try:
                    hist = data.xs(ticker, level=1, axis=1)
                except KeyError:
                    if verbose:
                        print(f"  [SKIP] {ticker}: no data")
                    continue

            if hist is None or len(hist) < 20:
                if verbose:
                    print(f"  [SKIP] {ticker}: insufficient history ({len(hist) if hist is not None else 0} rows)")
                continue

            hist = hist.dropna(subset=["Close"])
            if len(hist) < 2:
                continue

            ind = calc_indicators(hist)
            if ind["close"] is None or ind["close"] <= 0:
                continue

            # 時価総額（yfinance info から取得、失敗時はデフォルト）
            market_cap = _get_market_cap(ticker)

            rows.append({
                "Ticker":            ticker,
                "Name":              name,
                "Sector":            sector,
                "Industry":          industry,
                "Close":             ind["close"],
                "Change %":          ind["change_pct"],
                "Volume":            ind["volume"],
                "Relative Volume":   ind["rel_volume"],
                "Market Cap":        market_cap,
                "RSI (14)":          ind["rsi"],
                "MACD":              ind["macd"],
                "MACD Signal":       ind["macd_signal"],
                "ADX (14)":          ind["adx"],
                "SMA20":             ind["sma20"],
                "SMA50":             ind["sma50"],
                "SMA200":            ind["sma200"],
            })

        except Exception as e:
            if verbose:
                print(f"  [WARN] {ticker}: {e}")
            continue

    return pd.DataFrame(rows)


_market_cap_cache: dict = {}

def _get_market_cap(ticker: str) -> float:
    """yfinance info から時価総額取得（キャッシュあり）"""
    if ticker in _market_cap_cache:
        return _market_cap_cache[ticker]
    try:
        info = yf.Ticker(ticker).fast_info
        mc = getattr(info, "market_cap", None) or 0
    except Exception:
        mc = 0
    _market_cap_cache[ticker] = float(mc)
    return float(mc)


def fetch_all(
    markets: list,
    output_dir: str,
    period: str = "1y",
    verbose: bool = True,
) -> dict:
    """
    指定したマーケット群を取得して CSV に保存する。
    Returns: {market_name: filepath}
    """
    os.makedirs(output_dir, exist_ok=True)
    today_str = date.today().strftime("%Y-%m-%d")
    results = {}

    for market in markets:
        market = market.upper()
        if market not in MARKET_CONFIGS:
            print(f"[WARN] Unknown market: {market}. Use one of: {list(MARKET_CONFIGS)}")
            continue

        if verbose:
            print(f"\n[{market}] Fetching data...")

        ticker_map = MARKET_CONFIGS[market]
        df = fetch_asset_group(ticker_map, period=period, verbose=verbose)

        if df.empty:
            print(f"  [WARN] No data returned for {market}")
            continue

        # ファイル名: {MARKET}_{date}_{short_hash}.csv
        import hashlib
        short_hash = hashlib.md5(market.encode()).hexdigest()[:5]
        filename = f"{market}_{today_str}_{short_hash}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)

        if verbose:
            print(f"  [SAVED] {filepath} ({len(df)} assets)")

        results[market] = filepath

    return results


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Asset Data Fetcher (yfinance → TradingView CSV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--market",
        nargs="+",
        default=["ALL"],
        help="Market(s) to fetch: ETF BOND COMMODITY CRYPTO FX ALL (default: ALL)",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="yfinance download period (default: 1y). Valid: 1d 5d 1mo 3mo 6mo 1y 2y 5y",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "snapshots"),
        help="Directory to save CSV snapshots",
    )
    parser.add_argument(
        "--list-markets",
        action="store_true",
        help="List available markets and their tickers",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.list_markets:
        print("\nAvailable markets and tickers:")
        for market, tmap in MARKET_CONFIGS.items():
            print(f"\n  [{market}] ({len(tmap)} tickers)")
            for ticker, (name, sector, industry) in tmap.items():
                print(f"    {ticker:15s} {name} | {sector} > {industry}")
        return

    # ALL → 全マーケット
    markets = list(MARKET_CONFIGS.keys()) if "ALL" in [m.upper() for m in args.market] else args.market

    if verbose:
        print(f"Multi-Asset Fetcher — {date.today()}")
        print(f"Markets : {markets}")
        print(f"Period  : {args.period}")
        print(f"Output  : {args.output_dir}")

    results = fetch_all(
        markets=markets,
        output_dir=args.output_dir,
        period=args.period,
        verbose=verbose,
    )

    if verbose and results:
        print(f"\n{'='*50}")
        print(f"Saved {len(results)} CSV file(s):")
        for mkt, path in results.items():
            print(f"  [{mkt}] {path}")
        print("\nNext step: run the sector analyzer")
        print(f"  python3 screener/sector_flow_analyzer.py screener/snapshots/ --top 5 --export-csv --html")


if __name__ == "__main__":
    main()
