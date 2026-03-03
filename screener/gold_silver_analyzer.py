"""
Gold & Silver Dashboard Analyzer
==================================
yfinance を使って金・銀・関連ETFの価格データを取得し、
テクニカル分析とダッシュボードチャートを生成する。

Tickers fetched:
  GC=F       — Gold Futures (コメックス金先物)
  SI=F       — Silver Futures (コメックス銀先物)
  GLD        — SPDR Gold Shares ETF
  SLV        — iShares Silver Trust ETF
  GDX        — VanEck Gold Miners ETF
  SIL        — Global X Silver Miners ETF
  DX-Y.NYB   — US Dollar Index (DXY)
  SPY        — S&P 500 ETF (benchmark)

Usage:
    python3 screener/gold_silver_analyzer.py
    python3 screener/gold_silver_analyzer.py --period 6mo
    python3 screener/gold_silver_analyzer.py --no-chart --export-csv
    python3 screener/gold_silver_analyzer.py --out screener/my_dashboard.png
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path(__file__).parent          # screener/
DASHBOARD_PATH = OUTPUT_DIR / "gold_silver_dashboard.png"
CSV_PATH = OUTPUT_DIR / "gold_silver_metrics.csv"
RATIO_CSV_PATH = OUTPUT_DIR / "gold_silver_ratio_history.csv"

# Tickers to fetch (ordered for display)
TICKERS: dict[str, str] = {
    "GC=F":      "Gold Futures (金先物)",
    "SI=F":      "Silver Futures (銀先物)",
    "GLD":       "SPDR Gold ETF",
    "SLV":       "iShares Silver ETF",
    "GDX":       "Gold Miners ETF",
    "SIL":       "Silver Miners ETF",
    "DX-Y.NYB":  "US Dollar Index (DXY)",
    "SPY":       "S&P 500 ETF (Benchmark)",
}

COLORS: dict[str, str] = {
    "bullish":       "#27ae60",
    "mild_bull":     "#2ecc71",
    "neutral":       "#f39c12",
    "bearish":       "#e74c3c",
    "gold":          "#FFD700",
    "silver":        "#C0C0C0",
    "dxy":           "#3498db",
    "gdx":           "#e67e22",
    "sil":           "#9b59b6",
    "spy":           "#95a5a6",
    "macd_line":     "#3498db",
    "macd_signal":   "#e74c3c",
    "histogram_pos": "#27ae60",
    "histogram_neg": "#e74c3c",
    "sma20":         "#7fb3d8",
    "sma50":         "#f39c12",
    "sma200":        "#e74c3c",
    "bg_dark":       "#0d1117",
    "bg_panel":      "#161b22",
    "text":          "white",
    "grid":          "#30363d",
}

DEFAULT_PERIOD = "1y"
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL_PERIOD = 9

# Gold/Silver ratio reference levels
GS_RATIO_LOW = 70    # Silver-favorable range below this
GS_RATIO_MID = 80    # Neutral range
GS_RATIO_HIGH = 90   # Gold-favorable / risk-off above this


# ---------------------------------------------------------------------------
# Data Model
# ---------------------------------------------------------------------------

@dataclass
class AssetMetrics:
    """単一資産の最新テクニカル指標スナップショット"""
    ticker: str
    name: str
    current_price: float
    prev_close: float
    change_pct: float           # 直近1日変動率
    ytd_pct: float              # YTD パフォーマンス %
    sma20: float
    sma50: float
    sma200: float
    rsi: float                  # RSI(14)
    macd_line: float            # MACD line
    macd_signal: float          # MACD signal line
    macd_hist: float            # MACD histogram = line - signal
    above_sma20: bool
    above_sma50: bool
    above_sma200: bool
    trend_signal: str           # "LONG" | "SHORT" | "NEUTRAL"
    signal_reasons: list[str] = field(default_factory=list)
    rel_strength_vs_spy: float = 0.0   # relative performance vs SPY (%)


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def _flatten_yf_multiindex(
    raw: pd.DataFrame,
    tickers: list[str],
) -> dict[str, pd.DataFrame]:
    """yfinance multi-ticker download の MultiIndex DataFrame を ticker→DataFrame に変換する"""
    result: dict[str, pd.DataFrame] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                df = raw[t].dropna(how="all")
                if not df.empty:
                    result[t] = df
            except KeyError:
                pass
    else:
        # Single ticker returned as flat DataFrame
        if len(tickers) == 1 and not raw.empty:
            result[tickers[0]] = raw.dropna(how="all")
    return result


def _fetch_individually(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    """バッチ取得失敗時の個別ティッカーフォールバック"""
    import yfinance as yf
    result: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = yf.Ticker(t).history(period=period, auto_adjust=True)
            if not df.empty:
                result[t] = df
            else:
                print(f"  [Warning] {t}: 空のデータを受信 (市場休場または無効なティッカー)")
        except Exception as e:
            print(f"  [Warning] {t}: 取得失敗 — {e}")
    return result


def fetch_price_data(
    tickers: dict[str, str],
    period: str = DEFAULT_PERIOD,
) -> dict[str, pd.DataFrame]:
    """
    yfinance を使って指定ティッカーの価格データを取得する。
    Returns dict[ticker → OHLCV DataFrame]
    """
    try:
        import yfinance as yf
    except ImportError:
        print("  [Error] yfinance がインストールされていません。")
        print("  [Error] Run: pip install yfinance>=0.2.30")
        return {}

    ticker_list = list(tickers.keys())

    # Try batch download first
    try:
        raw = yf.download(
            ticker_list,
            period=period,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        result = _flatten_yf_multiindex(raw, ticker_list)

        # If batch returned mostly empty data, fall back to individual
        if len(result) < len(ticker_list) // 2:
            print("  [Info] バッチ取得が不完全です。個別取得に切り替えます...")
            missing = [t for t in ticker_list if t not in result]
            individual = _fetch_individually(missing, period)
            result.update(individual)

    except Exception as e:
        print(f"  [Warning] バッチ取得失敗、個別取得に切り替え: {e}")
        result = _fetch_individually(ticker_list, period)

    print(f"  取得完了: {len(result)}/{len(ticker_list)} ティッカー")
    return result


# ---------------------------------------------------------------------------
# Sample Data Generator (offline fallback / デモモード)
# ---------------------------------------------------------------------------

def _make_ohlcv(close: np.ndarray, dates: pd.DatetimeIndex, base_vol: float) -> pd.DataFrame:
    """Close 配列から OHLCV DataFrame を生成するヘルパー"""
    rng = np.random.default_rng(42)
    noise = rng.uniform(0.995, 1.005, len(close))
    high = close * rng.uniform(1.002, 1.015, len(close))
    low = close * rng.uniform(0.985, 0.998, len(close))
    vol = (base_vol * rng.uniform(0.7, 1.4, len(close))).astype(int)
    return pd.DataFrame(
        {"Open": close * noise, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def generate_sample_data(period: str = DEFAULT_PERIOD) -> dict[str, pd.DataFrame]:
    """
    ネットワーク接続がない環境向けのサンプルデータを生成する。
    現実的な価格レベルとトレンドを持つ合成データ (2024-2026 年相場を参考)。
    """
    rng = np.random.default_rng(2024)

    # Date range — weekdays only
    end_date = pd.Timestamp("2026-03-03")
    if period in ("3mo", "3m"):
        start_date = end_date - pd.DateOffset(months=3)
    elif period in ("6mo", "6m"):
        start_date = end_date - pd.DateOffset(months=6)
    elif period in ("ytd",):
        start_date = pd.Timestamp(f"{end_date.year}-01-01")
    else:  # default: 1y
        start_date = end_date - pd.DateOffset(years=1)

    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)

    def _gbm(s0: float, mu: float, sigma: float) -> np.ndarray:
        """幾何ブラウン運動で価格系列を生成する"""
        dt = 1 / 252
        returns = rng.normal(mu * dt, sigma * np.sqrt(dt), n)
        return s0 * np.exp(np.cumsum(returns))

    # Realistic 2025-2026 price trajectories
    gold_close   = _gbm(2400.0, 0.20, 0.12)   # ~$2400→$2900 (strong bull)
    silver_close = _gbm(28.0,   0.18, 0.22)    # ~$28→$32 (volatile bull)
    gld_close    = gold_close * 0.0933          # GLD ≈ GC/10.7
    slv_close    = silver_close * 0.98          # SLV ≈ SI
    gdx_close    = _gbm(30.0,   0.22, 0.28)    # Gold miners ETF
    sil_close    = _gbm(16.0,   0.20, 0.30)    # Silver miners ETF
    dxy_close    = _gbm(104.0, -0.03, 0.06)    # DXY: mild decline (gold positive)
    spy_close    = _gbm(480.0,  0.15, 0.15)    # S&P500

    data: dict[str, pd.DataFrame] = {
        "GC=F":      _make_ohlcv(gold_close,   dates, 250_000),
        "SI=F":      _make_ohlcv(silver_close, dates, 80_000),
        "GLD":       _make_ohlcv(gld_close,    dates, 8_000_000),
        "SLV":       _make_ohlcv(slv_close,    dates, 12_000_000),
        "GDX":       _make_ohlcv(gdx_close,    dates, 20_000_000),
        "SIL":       _make_ohlcv(sil_close,    dates, 2_000_000),
        "DX-Y.NYB":  _make_ohlcv(dxy_close,    dates, 0),
        "SPY":       _make_ohlcv(spy_close,    dates, 80_000_000),
    }

    print("  [Demo] サンプルデータを使用しています (ネットワーク接続なし)")
    print("  [Demo] Using synthetic sample data (no network access)")
    return data


# ---------------------------------------------------------------------------
# Technical Indicators (pandas / numpy only)
# ---------------------------------------------------------------------------

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """単純移動平均 (SMA)"""
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI(14) — Wilder's smoothing (EWM alpha=1/period)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL_PERIOD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram を返す"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_ytd(series: pd.Series) -> float:
    """年初来パフォーマンス % を計算する"""
    current_year = datetime.now().year
    ytd_data = series[series.index.year == current_year]
    if ytd_data.empty:
        return 0.0
    first_price = ytd_data.iloc[0]
    last_price = series.iloc[-1]
    if first_price == 0 or pd.isna(first_price):
        return 0.0
    return float((last_price / first_price - 1) * 100)


def compute_rel_strength(asset_series: pd.Series, spy_series: pd.Series) -> float:
    """SPY に対する相対パフォーマンス % を計算する"""
    if spy_series.empty or asset_series.empty:
        return 0.0
    # Align on common dates
    combined = pd.concat([asset_series, spy_series], axis=1).dropna()
    if combined.empty or len(combined) < 2:
        return 0.0
    a_norm = combined.iloc[:, 0] / combined.iloc[0, 0]
    s_norm = combined.iloc[:, 1] / combined.iloc[0, 1]
    rel = (a_norm / s_norm - 1) * 100
    return float(rel.iloc[-1])


# ---------------------------------------------------------------------------
# Signal Generation
# ---------------------------------------------------------------------------

def determine_trend_signal(
    rsi: float,
    macd_hist: float,
    above_sma50: bool,
    above_sma200: bool,
    above_sma20: bool,
    rel_strength: float,
) -> tuple[str, list[str]]:
    """テクニカル指標からトレンドシグナルを判定する"""
    score = 0
    reasons: list[str] = []

    if above_sma200:
        score += 25
        reasons.append("Close > SMA200 (長期上昇トレンド)")
    else:
        score -= 10
        reasons.append("Close < SMA200 (長期下降トレンド)")

    if above_sma50:
        score += 20
        reasons.append("Close > SMA50 (中期上昇トレンド)")
    else:
        score -= 5
        reasons.append("Close < SMA50 (中期下降トレンド)")

    if above_sma20:
        score += 10
        reasons.append("Close > SMA20 (短期上昇トレンド)")

    if 40 < rsi < 70:
        score += 15
        reasons.append(f"RSI={rsi:.1f} (適正レンジ)")
    elif rsi >= 70:
        score -= 5
        reasons.append(f"RSI={rsi:.1f} (過熱注意 ⚠)")
    elif rsi <= 30:
        score -= 5
        reasons.append(f"RSI={rsi:.1f} (売られ過ぎ)")

    if macd_hist > 0:
        score += 15
        reasons.append("MACD > Signal (上昇モメンタム)")
    else:
        score -= 5
        reasons.append("MACD < Signal (下降モメンタム)")

    if rel_strength > 5:
        score += 15
        reasons.append(f"SPY比 +{rel_strength:.1f}% 超過パフォーマンス")
    elif rel_strength < -5:
        score -= 5
        reasons.append(f"SPY比 {rel_strength:.1f}% 劣後パフォーマンス")

    if score >= 50:
        signal = "LONG"
    elif score <= 20:
        signal = "SHORT"
    else:
        signal = "NEUTRAL"

    return signal, reasons


def compute_asset_metrics(
    price_data: dict[str, pd.DataFrame],
    ticker_names: dict[str, str],
) -> dict[str, AssetMetrics]:
    """全ティッカーのテクニカル指標を計算して AssetMetrics を返す"""
    spy_close = price_data.get("SPY", pd.DataFrame()).get("Close", pd.Series(dtype=float))
    metrics: dict[str, AssetMetrics] = {}

    for ticker, df in price_data.items():
        if df.empty or "Close" not in df.columns:
            continue
        try:
            close = df["Close"].dropna()
            if len(close) < 5:
                continue

            sma20 = compute_sma(close, 20)
            sma50 = compute_sma(close, 50)
            sma200 = compute_sma(close, 200)
            rsi_series = compute_rsi(close)
            macd_line_s, signal_line_s, histogram_s = compute_macd(close)

            current = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else current
            change_pct = ((current / prev) - 1) * 100 if prev != 0 else 0.0

            sma20_val = float(sma20.iloc[-1]) if sma20.notna().any() else float("nan")
            sma50_val = float(sma50.iloc[-1]) if sma50.notna().any() else float("nan")
            sma200_val = float(sma200.iloc[-1]) if sma200.notna().any() else float("nan")

            above20 = bool(current > sma20_val) if not np.isnan(sma20_val) else False
            above50 = bool(current > sma50_val) if not np.isnan(sma50_val) else False
            above200 = bool(current > sma200_val) if not np.isnan(sma200_val) else False

            rsi_val = float(rsi_series.iloc[-1])
            macd_val = float(macd_line_s.iloc[-1])
            macd_sig_val = float(signal_line_s.iloc[-1])
            macd_hist_val = float(histogram_s.iloc[-1])

            ytd_val = compute_ytd(close)
            rel_strength = compute_rel_strength(close, spy_close)

            signal, reasons = determine_trend_signal(
                rsi=rsi_val,
                macd_hist=macd_hist_val,
                above_sma50=above50,
                above_sma200=above200,
                above_sma20=above20,
                rel_strength=rel_strength,
            )

            metrics[ticker] = AssetMetrics(
                ticker=ticker,
                name=ticker_names.get(ticker, ticker),
                current_price=current,
                prev_close=prev,
                change_pct=change_pct,
                ytd_pct=ytd_val,
                sma20=sma20_val,
                sma50=sma50_val,
                sma200=sma200_val,
                rsi=rsi_val,
                macd_line=macd_val,
                macd_signal=macd_sig_val,
                macd_hist=macd_hist_val,
                above_sma20=above20,
                above_sma50=above50,
                above_sma200=above200,
                trend_signal=signal,
                signal_reasons=reasons,
                rel_strength_vs_spy=rel_strength,
            )
        except Exception as e:
            print(f"  [Warning] {ticker} の指標計算失敗: {e}")

    return metrics


def compute_gold_silver_ratio(
    price_data: dict[str, pd.DataFrame],
) -> pd.Series:
    """
    金銀レシオ (Gold/Silver Ratio) を計算する。
    GC=F が優先、なければ GLD を使用。同様に SI=F → SLV。
    """
    gold_close = _get_close(price_data, ["GC=F", "GLD"])
    silver_close = _get_close(price_data, ["SI=F", "SLV"])

    if gold_close is None or silver_close is None:
        return pd.Series(dtype=float)

    combined = pd.concat([gold_close, silver_close], axis=1).dropna()
    combined.columns = ["gold", "silver"]
    ratio = combined["gold"] / combined["silver"]
    return ratio


def _get_close(
    price_data: dict[str, pd.DataFrame],
    priority: list[str],
) -> Optional[pd.Series]:
    """優先順位リストから最初に利用可能な Close Series を返す"""
    for ticker in priority:
        df = price_data.get(ticker)
        if df is not None and not df.empty and "Close" in df.columns:
            return df["Close"].dropna()
    return None


# ---------------------------------------------------------------------------
# Console Report
# ---------------------------------------------------------------------------

def _signal_color_str(signal: str) -> str:
    if signal == "LONG":
        return "LONG "
    elif signal == "SHORT":
        return "SHORT"
    return "NEUT "


def _bar(value: float, width: int = 20) -> str:
    """0-100 の値を ASCII バーで表現する"""
    filled = int(max(0, min(value, 100)) * width / 100)
    return "█" * filled + "░" * (width - filled)


def print_gold_silver_report(
    metrics: dict[str, AssetMetrics],
    gs_ratio: pd.Series,
) -> None:
    """金・銀分析サマリーをコンソールに出力する"""
    today = date.today().isoformat()

    print(f"\n{'=' * 74}")
    print(f"  GOLD & SILVER DASHBOARD  /  金・銀 コモディティ分析")
    print(f"  Date: {today}")
    print(f"{'=' * 74}")

    # Asset table
    print(f"\n  {'Asset':<28} {'Price':>10} {'Chg%':>7} {'YTD%':>7} {'RSI':>5}  Signal  RSI Bar")
    print(f"  {'-' * 70}")

    display_order = ["GC=F", "SI=F", "GLD", "SLV", "GDX", "SIL", "DX-Y.NYB", "SPY"]
    for t in display_order:
        m = metrics.get(t)
        if m is None:
            continue
        sig = _signal_color_str(m.trend_signal)
        chg_str = f"{m.change_pct:+.2f}%"
        ytd_str = f"{m.ytd_pct:+.1f}%"
        rsi_bar = _bar(m.rsi, 15)
        print(
            f"  {m.name:<28} {m.current_price:>10.2f} {chg_str:>7} {ytd_str:>7} "
            f"{m.rsi:>5.1f}  [{sig}]  {rsi_bar}"
        )

    # Gold/Silver Ratio section
    print(f"\n  {'-' * 74}")
    print(f"  GOLD/SILVER RATIO  /  金銀レシオ")
    print(f"  {'-' * 74}")
    if not gs_ratio.empty:
        current_ratio = float(gs_ratio.iloc[-1])
        ratio_sma20 = float(compute_sma(gs_ratio, 20).iloc[-1]) if len(gs_ratio) >= 20 else float("nan")
        ratio_sma50 = float(compute_sma(gs_ratio, 50).iloc[-1]) if len(gs_ratio) >= 50 else float("nan")

        sma20_str = f"{ratio_sma20:.1f}" if not np.isnan(ratio_sma20) else "N/A"
        sma50_str = f"{ratio_sma50:.1f}" if not np.isnan(ratio_sma50) else "N/A"

        print(f"  Current Ratio: {current_ratio:.1f}  (SMA20: {sma20_str}  SMA50: {sma50_str})")

        if current_ratio > GS_RATIO_HIGH:
            interp = f"レシオ高水準 ({current_ratio:.1f} > {GS_RATIO_HIGH}) → ゴールドが相対優位 / リスクオフ局面"
        elif current_ratio > GS_RATIO_MID:
            interp = f"レシオ中程度 ({current_ratio:.1f}) → 均衡状態、方向性を確認"
        else:
            interp = f"レシオ低水準 ({current_ratio:.1f} < {GS_RATIO_MID}) → シルバーが相対優位 / リスクオン局面"
        print(f"  Interpretation: {interp}")
        print(f"  Reference:  < {GS_RATIO_LOW} = Silver強い  |  {GS_RATIO_LOW}-{GS_RATIO_HIGH} = 中立  |  > {GS_RATIO_HIGH} = Gold強い")
    else:
        print("  データなし (金または銀の価格取得に失敗)")

    # DXY section
    print(f"\n  {'-' * 74}")
    print(f"  DXY (US Dollar Index)  /  米ドル指数")
    print(f"  {'-' * 74}")
    m_dxy = metrics.get("DX-Y.NYB")
    if m_dxy:
        trend_note = "DXY 下落トレンド → 金・銀に追い風" if m_dxy.change_pct < 0 else "DXY 上昇トレンド → 金・銀に逆風"
        print(f"  Current: {m_dxy.current_price:.2f}  Change: {m_dxy.change_pct:+.2f}%  RSI: {m_dxy.rsi:.1f}")
        print(f"  Signal: [{_signal_color_str(m_dxy.trend_signal)}]  Note: {trend_note}")
    else:
        print("  DXY データなし")

    # Key Observations
    print(f"\n  {'-' * 74}")
    print(f"  KEY OBSERVATIONS  /  主要観察事項")
    print(f"  {'-' * 74}")

    observations = []
    for t in ["GC=F", "SI=F"]:
        m = metrics.get(t)
        if m is None:
            continue
        if m.above_sma200:
            observations.append(f"{m.name}: SMA200 上 → 長期上昇トレンド継続")
        else:
            observations.append(f"{m.name}: SMA200 下 → 長期下降トレンド注意")
        if m.rsi > 65:
            observations.append(f"{m.name}: RSI={m.rsi:.1f} — 過熱気味、押し目に注意")
        elif m.rsi < 35:
            observations.append(f"{m.name}: RSI={m.rsi:.1f} — 売られ過ぎ、反発に注意")

    if not gs_ratio.empty:
        current_ratio = float(gs_ratio.iloc[-1])
        if current_ratio > 90:
            observations.append(f"金銀レシオ={current_ratio:.1f}: 歴史的高水準 → 銀の相対的割安感あり")
        elif current_ratio < 70:
            observations.append(f"金銀レシオ={current_ratio:.1f}: 低水準 → リスクオン環境を示唆")

    if not observations:
        observations.append("特記事項なし")

    for obs in observations:
        print(f"  [!] {obs}")

    print(f"\n{'=' * 74}\n")


# ---------------------------------------------------------------------------
# Matplotlib Dashboard
# ---------------------------------------------------------------------------

def plot_gold_silver_dashboard(
    price_data: dict[str, pd.DataFrame],
    metrics: dict[str, AssetMetrics],
    gs_ratio: pd.Series,
    output_path: Optional[str] = None,
    is_sample: bool = False,
) -> None:
    """8パネルの金・銀ダッシュボードを PNG として保存する"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as mgridspec
        try:
            import matplotlib_fontja  # noqa: F401
        except ImportError:
            pass
    except ImportError:
        print("  [Warning] matplotlib がインストールされていません。チャート生成をスキップします。")
        return

    gold_df = _get_df(price_data, ["GC=F", "GLD"])
    silver_df = _get_df(price_data, ["SI=F", "SLV"])
    dxy_df = _get_df(price_data, ["DX-Y.NYB"])
    spy_df = _get_df(price_data, ["SPY"])

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor(COLORS["bg_dark"])
    today_str = date.today().isoformat()
    demo_tag = "  [SAMPLE DATA / サンプルデータ]" if is_sample else ""
    fig.suptitle(
        f"Gold & Silver Dashboard  /  金・銀 コモディティ分析     {today_str}{demo_tag}",
        fontsize=15, fontweight="bold", color=COLORS["text"], y=0.995
    )

    gs = fig.add_gridspec(
        4, 2,
        hspace=0.40,
        wspace=0.28,
        left=0.07,
        right=0.96,
        top=0.97,
        bottom=0.04,
        height_ratios=[1, 1, 1, 0.75],
    )

    # ── Panel 1: Gold Price ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_price_panel(ax1, gold_df, COLORS["gold"], "Gold Price (GC=F)  /  金先物", "Gold")

    # ── Panel 2: Silver Price ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_price_panel(ax2, silver_df, COLORS["silver"], "Silver Price (SI=F)  /  銀先物", "Silver")

    # ── Panel 3: Gold/Silver Ratio ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_ratio_panel(ax3, gs_ratio)

    # ── Panel 4: RSI Comparison ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_rsi_panel(ax4, gold_df, silver_df)

    # ── Panel 5: Gold MACD ───────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    _plot_macd_panel(ax5, gold_df, COLORS["gold"], "Gold MACD (12,26,9)")

    # ── Panel 6: Silver MACD ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    _plot_macd_panel(ax6, silver_df, COLORS["silver"], "Silver MACD (12,26,9)")

    # ── Panel 7: DXY ─────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, 0])
    _plot_dxy_panel(ax7, dxy_df)

    # ── Panel 8: YTD Performance Bar Chart ───────────────────────────────
    ax8 = fig.add_subplot(gs[3, 1])
    _plot_performance_panel(ax8, metrics)

    out = output_path or str(DASHBOARD_PATH)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Dashboard saved to: {out}")
    print(f"  ダッシュボード保存完了: {out}")


def _get_df(
    price_data: dict[str, pd.DataFrame],
    priority: list[str],
) -> pd.DataFrame:
    """優先リストから最初に利用可能な DataFrame を返す"""
    for t in priority:
        df = price_data.get(t)
        if df is not None and not df.empty:
            return df
    return pd.DataFrame()


def _style_ax(ax) -> None:
    """共通のダークスタイルをパネルに適用する"""
    ax.set_facecolor(COLORS["bg_panel"])
    ax.tick_params(colors=COLORS["text"], labelsize=7)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.4, alpha=0.5)


def _plot_price_panel(
    ax,
    df: pd.DataFrame,
    color: str,
    title: str,
    label: str,
) -> None:
    """価格 + SMA チャートを描画する"""
    _style_ax(ax)
    if df.empty or "Close" not in df.columns:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                color=COLORS["text"], transform=ax.transAxes)
        ax.set_title(title, fontweight="bold")
        return

    close = df["Close"].dropna()
    dates = close.index

    sma20 = compute_sma(close, 20)
    sma50 = compute_sma(close, 50)
    sma200 = compute_sma(close, 200)

    ax.plot(dates, close, color=color, linewidth=2.0, label=label, zorder=3)

    if sma20.notna().sum() >= 5:
        ax.plot(dates, sma20, color=COLORS["sma20"], linewidth=1.0,
                linestyle="--", alpha=0.8, label="SMA20")
    if sma50.notna().sum() >= 5:
        ax.plot(dates, sma50, color=COLORS["sma50"], linewidth=1.0,
                linestyle="--", alpha=0.8, label="SMA50")
    if sma200.notna().sum() >= 10:
        ax.plot(dates, sma200, color=COLORS["sma200"], linewidth=1.2,
                linestyle="--", alpha=0.8, label="SMA200")

    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", facecolor=COLORS["bg_panel"],
              edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    ax.set_ylabel("Price", color=COLORS["text"])

    # Rotate date labels
    for label_obj in ax.get_xticklabels():
        label_obj.set_rotation(30)
        label_obj.set_ha("right")


def _plot_ratio_panel(ax, gs_ratio: pd.Series) -> None:
    """Gold/Silver Ratio チャートを描画する"""
    _style_ax(ax)
    ax.set_title("Gold/Silver Ratio  /  金銀レシオ", fontweight="bold")

    if gs_ratio.empty:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                color=COLORS["text"], transform=ax.transAxes)
        return

    dates = gs_ratio.index
    ratio_sma20 = compute_sma(gs_ratio, 20)
    ratio_sma50 = compute_sma(gs_ratio, 50)

    ax.plot(dates, gs_ratio, color="white", linewidth=1.8, label="G/S Ratio", zorder=3)
    if ratio_sma20.notna().sum() >= 5:
        ax.plot(dates, ratio_sma20, color=COLORS["gold"], linewidth=1.0,
                linestyle="--", alpha=0.8, label="SMA20")
    if ratio_sma50.notna().sum() >= 5:
        ax.plot(dates, ratio_sma50, color=COLORS["neutral"], linewidth=1.0,
                linestyle="--", alpha=0.8, label="SMA50")

    # Reference zones
    ax.axhline(GS_RATIO_LOW, color="#27ae60", linestyle=":", alpha=0.6,
               label=f"{GS_RATIO_LOW} (Silver優位)")
    ax.axhline(GS_RATIO_MID, color=COLORS["neutral"], linestyle=":", alpha=0.6,
               label=f"{GS_RATIO_MID} (中立)")
    ax.axhline(GS_RATIO_HIGH, color="#e74c3c", linestyle=":", alpha=0.6,
               label=f"{GS_RATIO_HIGH} (Gold優位)")

    ymin = max(0, float(gs_ratio.min()) - 5)
    ymax = float(gs_ratio.max()) + 5
    ax.fill_between(dates, GS_RATIO_LOW, GS_RATIO_MID, alpha=0.06, color="#27ae60")
    ax.fill_between(dates, GS_RATIO_HIGH, ymax, alpha=0.06, color="#e74c3c")
    ax.set_ylim(ymin, ymax)

    ax.legend(fontsize=6, loc="upper left", facecolor=COLORS["bg_panel"],
              edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    ax.set_ylabel("Ratio", color=COLORS["text"])

    for label_obj in ax.get_xticklabels():
        label_obj.set_rotation(30)
        label_obj.set_ha("right")


def _plot_rsi_panel(ax, gold_df: pd.DataFrame, silver_df: pd.DataFrame) -> None:
    """RSI 比較チャートを描画する"""
    _style_ax(ax)
    ax.set_title("RSI(14) Comparison  /  RSI 比較", fontweight="bold")

    has_data = False
    if not gold_df.empty and "Close" in gold_df.columns:
        gold_close = gold_df["Close"].dropna()
        rsi_gold = compute_rsi(gold_close)
        ax.plot(gold_close.index, rsi_gold, color=COLORS["gold"],
                linewidth=1.5, label="Gold RSI")
        has_data = True

    if not silver_df.empty and "Close" in silver_df.columns:
        silver_close = silver_df["Close"].dropna()
        rsi_silver = compute_rsi(silver_close)
        ax.plot(silver_close.index, rsi_silver, color=COLORS["silver"],
                linewidth=1.5, linestyle="--", label="Silver RSI")
        has_data = True

    if not has_data:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                color=COLORS["text"], transform=ax.transAxes)
        return

    ax.axhline(70, color="#e74c3c", linestyle=":", alpha=0.7, linewidth=1.0)
    ax.axhline(50, color="gray",    linestyle=":", alpha=0.4, linewidth=0.8)
    ax.axhline(30, color="#27ae60", linestyle=":", alpha=0.7, linewidth=1.0)

    ax.fill_between(ax.get_xlim(), 70, 100, alpha=0.06, color="#e74c3c",
                    transform=ax.get_xaxis_transform())
    ax.fill_between(ax.get_xlim(), 0, 30, alpha=0.06, color="#27ae60",
                    transform=ax.get_xaxis_transform())

    ax.set_ylim(0, 100)
    ax.set_ylabel("RSI", color=COLORS["text"])
    ax.legend(fontsize=6, loc="upper left", facecolor=COLORS["bg_panel"],
              edgecolor=COLORS["grid"], labelcolor=COLORS["text"])

    # Overbought/oversold labels
    ax.text(ax.get_xlim()[0], 72, "過熱 (70)", fontsize=6, color="#e74c3c", alpha=0.8)
    ax.text(ax.get_xlim()[0], 25, "売られ過ぎ (30)", fontsize=6, color="#27ae60", alpha=0.8)

    for label_obj in ax.get_xticklabels():
        label_obj.set_rotation(30)
        label_obj.set_ha("right")


def _plot_macd_panel(ax, df: pd.DataFrame, line_color: str, title: str) -> None:
    """MACD チャートを描画する"""
    _style_ax(ax)
    ax.set_title(title, fontweight="bold")

    if df.empty or "Close" not in df.columns:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                color=COLORS["text"], transform=ax.transAxes)
        return

    close = df["Close"].dropna()
    macd_line, signal_line, histogram = compute_macd(close)
    dates = close.index

    # Histogram bars with conditional coloring
    hist_colors = [
        COLORS["histogram_pos"] if v >= 0 else COLORS["histogram_neg"]
        for v in histogram.fillna(0)
    ]
    ax.bar(dates, histogram, color=hist_colors, alpha=0.5, width=1.5, label="Histogram")
    ax.plot(dates, macd_line, color=COLORS["macd_line"], linewidth=1.2, label="MACD")
    ax.plot(dates, signal_line, color=COLORS["macd_signal"], linewidth=1.0,
            linestyle="--", label="Signal")
    ax.axhline(0, color="gray", linewidth=0.6, alpha=0.5)

    ax.legend(fontsize=6, loc="upper left", facecolor=COLORS["bg_panel"],
              edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    ax.set_ylabel("MACD", color=COLORS["text"])

    for label_obj in ax.get_xticklabels():
        label_obj.set_rotation(30)
        label_obj.set_ha("right")


def _plot_dxy_panel(ax, dxy_df: pd.DataFrame) -> None:
    """DXY (US Dollar Index) チャートを描画する"""
    _style_ax(ax)
    ax.set_title("DXY — US Dollar Index  /  米ドル指数", fontweight="bold")

    if dxy_df.empty or "Close" not in dxy_df.columns:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                color=COLORS["text"], transform=ax.transAxes)
        return

    close = dxy_df["Close"].dropna()
    sma20 = compute_sma(close, 20)
    dates = close.index

    ax.plot(dates, close, color=COLORS["dxy"], linewidth=1.8, label="DXY")
    if sma20.notna().sum() >= 5:
        ax.plot(dates, sma20, color="white", linewidth=1.0,
                linestyle="--", alpha=0.6, label="SMA20")

    ax.legend(fontsize=6, loc="upper left", facecolor=COLORS["bg_panel"],
              edgecolor=COLORS["grid"], labelcolor=COLORS["text"])
    ax.set_ylabel("DXY", color=COLORS["text"])

    # Annotation
    latest = float(close.iloc[-1])
    ax.annotate(
        f"DXY: {latest:.2f}\n(金と逆相関 / negatively correlated with Gold)",
        xy=(0.02, 0.05),
        xycoords="axes fraction",
        fontsize=7,
        color=COLORS["text"],
        alpha=0.8,
    )

    for label_obj in ax.get_xticklabels():
        label_obj.set_rotation(30)
        label_obj.set_ha("right")


def _plot_performance_panel(ax, metrics: dict[str, AssetMetrics]) -> None:
    """YTD パフォーマンス比較バーチャートを描画する"""
    _style_ax(ax)
    ax.set_title("YTD Performance  /  年初来パフォーマンス", fontweight="bold")

    display_tickers = ["GLD", "SLV", "GDX", "SIL", "GC=F", "SI=F", "SPY"]
    labels = []
    ytd_values = []

    for t in display_tickers:
        m = metrics.get(t)
        if m is not None:
            # Use short labels for chart
            short_label = t.replace("=F", "").replace("DX-Y.NYB", "DXY")
            labels.append(short_label)
            ytd_values.append(m.ytd_pct)

    if not labels:
        ax.text(0.5, 0.5, "データなし", ha="center", va="center",
                color=COLORS["text"], transform=ax.transAxes)
        return

    bar_colors = [COLORS["bullish"] if v >= 0 else COLORS["bearish"] for v in ytd_values]
    bars = ax.bar(labels, ytd_values, color=bar_colors, edgecolor=COLORS["grid"],
                  linewidth=0.5, width=0.65)
    ax.axhline(0, color="white", linewidth=0.8, alpha=0.5)

    for bar, val in zip(bars, ytd_values):
        ypos = val + 0.3 if val >= 0 else val - 1.5
        ax.text(
            bar.get_x() + bar.get_width() / 2, ypos,
            f"{val:+.1f}%",
            ha="center", va="bottom", fontsize=8,
            color=COLORS["text"], fontweight="bold",
        )

    ax.set_ylabel("%", color=COLORS["text"])
    ax.tick_params(axis="x", rotation=0)


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_metrics_csv(
    metrics: dict[str, AssetMetrics],
    gs_ratio: pd.Series,
    output_dir: Optional[str] = None,
) -> None:
    """分析指標をCSVに出力する"""
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    # 1. Asset metrics snapshot
    rows = []
    for t, m in metrics.items():
        rows.append({
            "date": date.today().isoformat(),
            "ticker": m.ticker,
            "name": m.name,
            "price": round(m.current_price, 4),
            "change_pct": round(m.change_pct, 4),
            "ytd_pct": round(m.ytd_pct, 4),
            "rsi": round(m.rsi, 2),
            "macd_line": round(m.macd_line, 6),
            "macd_signal": round(m.macd_signal, 6),
            "macd_hist": round(m.macd_hist, 6),
            "sma20": round(m.sma20, 4) if not np.isnan(m.sma20) else "",
            "sma50": round(m.sma50, 4) if not np.isnan(m.sma50) else "",
            "sma200": round(m.sma200, 4) if not np.isnan(m.sma200) else "",
            "above_sma20": m.above_sma20,
            "above_sma50": m.above_sma50,
            "above_sma200": m.above_sma200,
            "rel_strength_spy": round(m.rel_strength_vs_spy, 4),
            "trend_signal": m.trend_signal,
            "signal_reasons": " | ".join(m.signal_reasons),
        })

    if rows:
        metrics_path = out_dir / "gold_silver_metrics.csv"
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"  Metrics CSV saved: {metrics_path}")
        print(f"  メトリクス CSV 保存完了: {metrics_path}")

    # 2. Gold/Silver ratio history
    if not gs_ratio.empty:
        ratio_path = out_dir / "gold_silver_ratio_history.csv"
        ratio_df = gs_ratio.reset_index()
        ratio_df.columns = ["date", "gold_silver_ratio"]
        ratio_df["gold_silver_ratio"] = ratio_df["gold_silver_ratio"].round(4)
        ratio_df.to_csv(ratio_path, index=False)
        print(f"  Ratio history CSV saved: {ratio_path}")
        print(f"  レシオ履歴 CSV 保存完了: {ratio_path}")


# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gold & Silver Dashboard Analyzer / 金・銀 コモディティ分析ツール"
    )
    parser.add_argument(
        "--period",
        default=DEFAULT_PERIOD,
        help="yfinance データ取得期間 (例: 1y, 6mo, 3mo, ytd). Default: 1y",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="ダッシュボード出力パス (PNG). Default: screener/gold_silver_dashboard.png",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="チャート生成をスキップする",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="指標データをCSVに出力する",
    )
    parser.add_argument(
        "--use-sample",
        action="store_true",
        help="サンプルデータを使用する (ネットワーク不要 / デモモード)",
    )
    args = parser.parse_args()

    print("\n" + "=" * 74)
    print("  Gold & Silver Analyzer  /  金・銀 コモディティ分析")
    print(f"  Period: {args.period}  |  Date: {date.today().isoformat()}")
    print("=" * 74)

    # 1. Fetch data
    using_sample = args.use_sample
    if using_sample:
        print("\n  [Demo mode] サンプルデータを使用します...")
        price_data = generate_sample_data(period=args.period)
    else:
        print("\n  Fetching market data... / 市場データ取得中...")
        price_data = fetch_price_data(TICKERS, period=args.period)

        # Auto-fallback to sample data when network is unavailable
        has_gold = "GC=F" in price_data or "GLD" in price_data
        has_silver = "SI=F" in price_data or "SLV" in price_data
        if not price_data or (not has_gold and not has_silver):
            print("\n  [Fallback] ネットワーク接続なし。サンプルデータに切り替えます...")
            print("  [Fallback] Network unavailable. Switching to sample data.")
            print("  [Tip] 実データを取得するには --use-sample を外してネットワーク接続を確認してください。")
            price_data = generate_sample_data(period=args.period)
            using_sample = True

    # 2. Compute technical indicators
    print("  Computing technical indicators... / テクニカル指標計算中...")
    metrics = compute_asset_metrics(price_data, TICKERS)
    gs_ratio = compute_gold_silver_ratio(price_data)

    # 3. Print console report
    print_gold_silver_report(metrics, gs_ratio)

    # 4. Generate dashboard chart
    if not args.no_chart:
        print("  Generating dashboard chart... / ダッシュボードチャート生成中...")
        plot_gold_silver_dashboard(price_data, metrics, gs_ratio, args.out,
                                   is_sample=using_sample)

    # 5. Export CSV
    if args.export_csv:
        print("  Exporting CSV... / CSV出力中...")
        export_metrics_csv(metrics, gs_ratio)

    print("\n  Done. / 分析完了。\n")


if __name__ == "__main__":
    main()
