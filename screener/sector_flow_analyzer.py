"""
Sector Rotation & Fund Flow Analyzer
=====================================
TradingView スクリーナー CSV を入力として、セクター別の強弱・資金フロー・
マーケットブレッスを分析し、セクターローテーションに基づく売買シグナルを生成する。

戦略コンセプト:
  1. Sector Composite Score (SCS) — 複数指標の加重合成でセクター強弱を1つの数値に
  2. Estimated Fund Flow (EFF) — 出来高×価格変動率 で資金流入/流出を推定
  3. Breadth Score — セクター内で SMA200 上の銘柄比率 (内部体力)
  4. Rotation Detector — SCS の日次変化を追跡し、資金移動パターンを検出
  5. Trade Signal — 強セクター内で個別銘柄をランキングしエントリー候補を提示

Usage:
    python screener/sector_flow_analyzer.py screener/snapshots/
    python screener/sector_flow_analyzer.py screener/snapshots/ --out report.png
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class SectorMetrics:
    """1つのスナップショットにおけるセクター別集計結果"""
    date: str
    sector: str
    # Composite Score 構成要素
    avg_change_pct: float        # 平均変動率
    avg_rsi: float               # 平均RSI
    avg_adx: float               # 平均ADX (トレンド強度)
    avg_macd_diff: float         # 平均 MACD - Signal
    breadth_sma20: float         # SMA20上の銘柄比率
    breadth_sma50: float         # SMA50上の銘柄比率
    breadth_sma200: float        # SMA200上の銘柄比率
    # Fund Flow
    total_fund_flow: float       # 推定資金フロー合計
    avg_relative_volume: float   # 平均相対出来高
    total_market_cap: float      # セクター時価総額合計
    # Composite
    composite_score: float = 0.0 # SCS (後で計算)
    stock_count: int = 0


@dataclass
class RotationSignal:
    """セクターローテーションシグナル"""
    date: str
    rising_sectors: list[str] = field(default_factory=list)
    falling_sectors: list[str] = field(default_factory=list)
    rotation_pairs: list[tuple[str, str, float]] = field(default_factory=list)


@dataclass
class TradeCandidate:
    """個別銘柄の売買候補"""
    ticker: str
    name: str
    sector: str
    industry: str
    signal: str          # "LONG" or "SHORT"
    strength: float      # シグナル強度 0-100
    reasons: list[str]
    close: float
    change_pct: float
    rsi: float
    volume_ratio: float


# ---------------------------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "Ticker", "Sector", "Close", "Change %", "Volume",
    "Relative Volume", "Market Cap", "SMA20", "SMA50", "SMA200",
    "RSI", "MACD.macd", "MACD.signal", "ADX",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """日本語/英語カラム名の揺らぎを正規化する"""
    # 日本語カラム名 → 英語カラム名のマッピング
    ja_col_map = {
        "シンボル": "Ticker",
        "詳細": "Name",
        "価格": "Close",
        "価格変動 % 1日": "Change %",
        "出来高 1日": "Volume",
        "相対ボリューム 1日": "Relative Volume",
        "時価総額": "Market Cap",
        "PER (株価収益率)": "P/E",
        "セクター": "Sector",
        "アナリストの評価": "Analyst Rating",
        "希薄化EPS, 直近12ヶ月": "EPS (TTM)",
        "希薄化EPS成長率 %, 直近12ヶ月前年比": "EPS Growth %",
        "配当利回り %, 直近12ヶ月": "Dividend Yield %",
        # テクニカル指標の日本語カラム
        "単純移動平均線 (20)": "SMA20",
        "単純移動平均線 (50)": "SMA50",
        "単純移動平均線 (200)": "SMA200",
        "相対力指数 (14)": "RSI",
        "MACD レベル (12, 26)": "MACD.macd",
        "MACDシグナル (12, 26)": "MACD.signal",
        "平均方向性指数 (14)": "ADX",
        "アベレージ・トゥルー・レンジ (14)": "ATR",
    }
    rename_map = {k: v for k, v in ja_col_map.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # カラム名の揺らぎを吸収 (英語カラムの表記揺れ対応)
    col_map = {}
    for c in df.columns:
        lower = c.strip().lower()
        if lower in ("change %", "change%", "chg%", "perf.d"):
            col_map[c] = "Change %"
        elif lower in ("relative volume", "rel volume", "rvol"):
            col_map[c] = "Relative Volume"
        elif lower in ("market cap", "mktcap", "market_cap"):
            col_map[c] = "Market Cap"
        elif lower in ("macd.macd", "macd", "macd level (12, 26)"):
            col_map[c] = "MACD.macd"
        elif lower in ("macd.signal", "macdsignal", "macd signal (12, 26)"):
            col_map[c] = "MACD.signal"
        elif lower in ("simple moving average (20)", "sma20"):
            col_map[c] = "SMA20"
        elif lower in ("simple moving average (50)", "sma50"):
            col_map[c] = "SMA50"
        elif lower in ("simple moving average (200)", "sma200"):
            col_map[c] = "SMA200"
        elif lower in ("relative strength index (14)", "rsi"):
            col_map[c] = "RSI"
        elif lower in ("average directional index (14)", "adx"):
            col_map[c] = "ADX"
        elif lower in ("average true range (14)", "atr"):
            col_map[c] = "ATR"
    if col_map:
        df = df.rename(columns=col_map)

    return df


def _convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """数値カラムを変換する"""
    numeric_cols = [
        "Close", "Change %", "Volume", "Relative Volume", "Market Cap",
        "SMA20", "SMA50", "SMA200", "RSI", "MACD.macd", "MACD.signal",
        "ADX", "ATR", "P/E", "EPS (TTM)",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_snapshot(filepath: str | Path, technical_filepath: str | Path | None = None) -> pd.DataFrame:
    """
    TradingView スクリーナー CSV を読み込み、正規化する。
    technical_filepath が指定された場合、テクニカル指標CSVをTickerでマージする。
    """
    df = pd.read_csv(filepath)
    df = _normalize_columns(df)

    # テクニカル指標CSVをマージ
    if technical_filepath is not None:
        tech_df = pd.read_csv(technical_filepath)
        tech_df = _normalize_columns(tech_df)
        tech_df = _convert_numeric(tech_df)

        if "Ticker" in tech_df.columns:
            # セクターCSV側に既に存在するテクニカルカラムは上書きしない
            # （テクニカルCSV側のデータを優先する）
            tech_cols = [c for c in tech_df.columns if c != "Ticker"]
            # セクターCSV側にあるがNaNだらけの同名カラムを除去してからマージ
            overlap_cols = [c for c in tech_cols if c in df.columns]
            if overlap_cols:
                df = df.drop(columns=overlap_cols)

            df = df.merge(tech_df[["Ticker"] + tech_cols], on="Ticker", how="left")
            print(f"  Merged technical data from {Path(technical_filepath).name} "
                  f"({len(tech_df)} rows, cols: {tech_cols})")
        else:
            print(f"  Warning: Technical CSV {technical_filepath} has no Ticker column, skipping merge")

    # 必須カラムチェック
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns {missing} in {filepath}")

    df = _convert_numeric(df)

    # Sector が空の行を除外
    df = df.dropna(subset=["Sector"])

    return df


def extract_date_from_filename(filepath: str | Path) -> str:
    """ファイル名から日付を推定 (例: screener_2025-01-06.csv → 2025-01-06)"""
    name = Path(filepath).stem
    # YYYY-MM-DD パターンを探す
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)
    return name


def is_technical_file(filepath: str | Path) -> bool:
    """ファイル名が Technical で始まるかどうか判定する"""
    return Path(filepath).name.lower().startswith("technical")


def pair_snapshot_files(csv_files: list[Path]) -> list[tuple[Path, Optional[Path]]]:
    """
    セクターCSVとテクニカルCSVを日付でペアリングする。

    Returns:
        list of (sector_csv, technical_csv_or_None) tuples, sorted by date.
    """
    sector_files: dict[str, list[Path]] = {}
    tech_files: dict[str, list[Path]] = {}

    for f in csv_files:
        date = extract_date_from_filename(f)
        if is_technical_file(f):
            tech_files.setdefault(date, []).append(f)
        else:
            sector_files.setdefault(date, []).append(f)

    # 日付ごとにペアリング
    pairs: list[tuple[Path, Optional[Path]]] = []
    all_dates = sorted(set(list(sector_files.keys()) + list(tech_files.keys())))

    for date in all_dates:
        s_files = sector_files.get(date, [])
        t_files = tech_files.get(date, [])

        if not s_files:
            # テクニカルのみ（セクター無し）はスキップ
            if t_files:
                print(f"  Skipping Technical-only file(s) for {date} (no matching sector CSV)")
            continue

        for sf in s_files:
            # 同日のテクニカルファイルがあればペアリング
            tf = t_files[0] if t_files else None
            pairs.append((sf, tf))

    return pairs


# ---------------------------------------------------------------------------
# Sector Metrics Computation
# ---------------------------------------------------------------------------

# SCS 加重設定
SCS_WEIGHTS = {
    "momentum":  0.25,  # Change%ベースのモメンタム
    "trend":     0.20,  # RSI + MACD の合成トレンド強度
    "breadth":   0.25,  # SMA200上の銘柄比率
    "flow":      0.20,  # 資金フロースコア
    "adx":       0.10,  # トレンド明確度
}


def compute_sector_metrics(df: pd.DataFrame, date_label: str) -> list[SectorMetrics]:
    """DataFrame から全セクターの SectorMetrics を計算する"""
    results = []

    for sector, group in df.groupby("Sector"):
        n = len(group)

        # --- 基本統計 ---
        avg_change = group["Change %"].mean() if "Change %" in group else 0.0
        avg_rsi = group["RSI"].mean() if "RSI" in group else 50.0
        avg_adx = group["ADX"].mean() if "ADX" in group else 0.0

        # MACD diff = MACD line - Signal line
        macd_diff = 0.0
        if "MACD.macd" in group and "MACD.signal" in group:
            macd_diff = (group["MACD.macd"] - group["MACD.signal"]).mean()

        # --- Breadth (MA上の銘柄比率) ---
        breadth_20 = _breadth(group, "SMA20")
        breadth_50 = _breadth(group, "SMA50")
        breadth_200 = _breadth(group, "SMA200")

        # --- 推定資金フロー ---
        # Fund Flow = Σ (Volume × Change% / 100)  出来高加重の方向性フロー
        fund_flow = 0.0
        if "Volume" in group and "Change %" in group:
            fund_flow = (group["Volume"] * group["Change %"] / 100.0).sum()

        avg_rvol = group["Relative Volume"].mean() if "Relative Volume" in group else 1.0
        total_mcap = group["Market Cap"].sum() if "Market Cap" in group else 0.0

        # --- Sector Composite Score (SCS) ---
        # 各要素を 0-100 にスケーリングしてから加重合成
        momentum_score = _clip_score((avg_change + 5) / 10 * 100)  # -5%~+5% → 0~100
        trend_score = _clip_score(
            0.5 * avg_rsi + 0.5 * (50 + macd_diff * 10)           # RSI + MACD合成
        )
        breadth_score = breadth_200 * 100                           # 0~1 → 0~100
        flow_direction = np.sign(fund_flow) if fund_flow != 0 else 0
        flow_magnitude = min(abs(fund_flow) / (total_mcap * 0.001 + 1), 1.0)
        flow_score = _clip_score(50 + flow_direction * flow_magnitude * 50)
        adx_score = _clip_score(avg_adx / 50 * 100)                # ADX 0~50+ → 0~100

        scs = (
            SCS_WEIGHTS["momentum"] * momentum_score
            + SCS_WEIGHTS["trend"] * trend_score
            + SCS_WEIGHTS["breadth"] * breadth_score
            + SCS_WEIGHTS["flow"] * flow_score
            + SCS_WEIGHTS["adx"] * adx_score
        )

        sm = SectorMetrics(
            date=date_label,
            sector=sector,
            avg_change_pct=round(avg_change, 4),
            avg_rsi=round(avg_rsi, 2),
            avg_adx=round(avg_adx, 2),
            avg_macd_diff=round(macd_diff, 4),
            breadth_sma20=round(breadth_20, 4),
            breadth_sma50=round(breadth_50, 4),
            breadth_sma200=round(breadth_200, 4),
            total_fund_flow=round(fund_flow, 2),
            avg_relative_volume=round(avg_rvol, 4),
            total_market_cap=total_mcap,
            composite_score=round(scs, 2),
            stock_count=n,
        )
        results.append(sm)

    return results


def _breadth(group: pd.DataFrame, ma_col: str) -> float:
    """Close が MA より上の銘柄の比率を返す (0-1)"""
    if ma_col not in group or "Close" not in group:
        return 0.5
    above = (group["Close"] > group[ma_col]).sum()
    return above / len(group) if len(group) > 0 else 0.5


def _clip_score(v: float) -> float:
    return max(0.0, min(100.0, v))


# ---------------------------------------------------------------------------
# Rotation Detection
# ---------------------------------------------------------------------------

def detect_rotation(
    metrics_history: list[list[SectorMetrics]],
    threshold: float = 3.0,
) -> list[RotationSignal]:
    """
    時系列の SectorMetrics から、セクターローテーション (資金移動) を検出。
    前日比で SCS が閾値以上変化したセクターを「上昇/下降」とし、
    下降→上昇のペアをローテーションペアとして出力する。
    """
    if len(metrics_history) < 2:
        return []

    signals = []

    for i in range(1, len(metrics_history)):
        prev_map = {m.sector: m for m in metrics_history[i - 1]}
        curr_map = {m.sector: m for m in metrics_history[i]}
        date = metrics_history[i][0].date if metrics_history[i] else "?"

        rising = []
        falling = []
        deltas: dict[str, float] = {}

        for sector in curr_map:
            if sector in prev_map:
                delta = curr_map[sector].composite_score - prev_map[sector].composite_score
                deltas[sector] = delta
                if delta >= threshold:
                    rising.append(sector)
                elif delta <= -threshold:
                    falling.append(sector)

        # ローテーションペア: falling → rising (資金が移動した可能性)
        pairs = []
        for f_sec in falling:
            for r_sec in rising:
                flow_strength = abs(deltas.get(r_sec, 0)) + abs(deltas.get(f_sec, 0))
                pairs.append((f_sec, r_sec, round(flow_strength, 2)))
        pairs.sort(key=lambda x: x[2], reverse=True)

        signals.append(RotationSignal(
            date=date,
            rising_sectors=rising,
            falling_sectors=falling,
            rotation_pairs=pairs,
        ))

    return signals


# ---------------------------------------------------------------------------
# Trade Signal Generator
# ---------------------------------------------------------------------------

def generate_trade_signals(
    df: pd.DataFrame,
    sector_metrics: list[SectorMetrics],
    rotation: Optional[RotationSignal] = None,
    top_n: int = 10,
) -> list[TradeCandidate]:
    """
    セクター分析結果を元に、個別銘柄の売買シグナルを生成する。

    ロジック:
    - 強セクター (SCS上位) から LONG 候補を選出
      → RSI < 70 (過熱でない) & Close > SMA50 (中期上昇トレンド)
      → Relative Volume > 1.0 (出来高伴う)
    - 弱セクター (SCS下位) から SHORT 候補を選出
      → RSI > 30 (まだ売り余地) & Close < SMA50 (中期下降トレンド)
      → Relative Volume > 1.0 (売り圧力の裏付け)
    - ローテーション中のセクターは追加ブースト
    """
    # セクターを SCS でランキング
    ranked = sorted(sector_metrics, key=lambda m: m.composite_score, reverse=True)
    n_sectors = len(ranked)
    if n_sectors == 0:
        return []

    # 上位1/3を強セクター、下位1/3を弱セクターとする
    cutoff = max(1, n_sectors // 3)
    strong_sectors = {m.sector for m in ranked[:cutoff]}
    weak_sectors = {m.sector for m in ranked[-cutoff:]}
    scs_map = {m.sector: m.composite_score for m in ranked}

    # ローテーションブースト
    rising_set = set(rotation.rising_sectors) if rotation else set()
    falling_set = set(rotation.falling_sectors) if rotation else set()

    candidates = []

    for _, row in df.iterrows():
        sector = row.get("Sector", "")
        ticker = row.get("Ticker", "")
        if not sector or not ticker:
            continue

        close = row.get("Close", 0)
        change = row.get("Change %", 0)
        rsi = row.get("RSI", 50)
        rvol = row.get("Relative Volume", 1.0)
        sma50 = row.get("SMA50", close)
        sma200 = row.get("SMA200", close)
        adx = row.get("ADX", 0)
        macd = row.get("MACD.macd", 0)
        macd_sig = row.get("MACD.signal", 0)

        reasons = []
        signal = None
        strength = 0.0

        # --- LONG 候補 ---
        if sector in strong_sectors:
            is_long = True
            reasons.append(f"強セクター (SCS={scs_map.get(sector, 0):.1f})")

            if close > sma50:
                strength += 20
                reasons.append("Close > SMA50 (中期上昇)")
            else:
                is_long = False

            if close > sma200:
                strength += 15
                reasons.append("Close > SMA200 (長期上昇)")

            if 40 < rsi < 70:
                strength += 15
                reasons.append(f"RSI={rsi:.1f} (適正レンジ)")
            elif rsi >= 70:
                strength -= 10
                reasons.append(f"RSI={rsi:.1f} (過熱注意)")

            if rvol > 1.0:
                strength += 10
                reasons.append(f"RVOL={rvol:.2f} (出来高増加)")

            if macd > macd_sig:
                strength += 10
                reasons.append("MACD > Signal (上昇モメンタム)")

            if adx > 20:
                strength += 10
                reasons.append(f"ADX={adx:.1f} (トレンド明確)")

            if sector in rising_set:
                strength += 15
                reasons.append("ローテーション流入中")

            if is_long and strength > 0:
                signal = "LONG"

        # --- SHORT 候補 ---
        elif sector in weak_sectors:
            is_short = True
            reasons.append(f"弱セクター (SCS={scs_map.get(sector, 0):.1f})")

            if close < sma50:
                strength += 20
                reasons.append("Close < SMA50 (中期下降)")
            else:
                is_short = False

            if close < sma200:
                strength += 15
                reasons.append("Close < SMA200 (長期下降)")

            if 30 < rsi < 60:
                strength += 15
                reasons.append(f"RSI={rsi:.1f} (まだ売り余地)")
            elif rsi <= 30:
                strength -= 10
                reasons.append(f"RSI={rsi:.1f} (売られ過ぎ注意)")

            if rvol > 1.0:
                strength += 10
                reasons.append(f"RVOL={rvol:.2f} (売り圧力)")

            if macd < macd_sig:
                strength += 10
                reasons.append("MACD < Signal (下降モメンタム)")

            if adx > 20:
                strength += 10
                reasons.append(f"ADX={adx:.1f} (トレンド明確)")

            if sector in falling_set:
                strength += 15
                reasons.append("ローテーション流出中")

            if is_short and strength > 0:
                signal = "SHORT"

        if signal:
            candidates.append(TradeCandidate(
                ticker=ticker,
                name=row.get("Name", ""),
                sector=sector,
                industry=row.get("Industry", ""),
                signal=signal,
                strength=min(100, strength),
                reasons=reasons,
                close=close,
                change_pct=change,
                rsi=rsi,
                volume_ratio=rvol,
            ))

    # 強度でソートして top_n
    candidates.sort(key=lambda c: c.strength, reverse=True)
    return candidates[:top_n]


# ---------------------------------------------------------------------------
# Report Printer
# ---------------------------------------------------------------------------

def print_sector_dashboard(
    all_metrics: list[list[SectorMetrics]],
    rotations: list[RotationSignal],
    candidates: list[TradeCandidate],
):
    """ターミナル向けの分析レポートを出力"""

    latest = all_metrics[-1] if all_metrics else []
    ranked = sorted(latest, key=lambda m: m.composite_score, reverse=True)

    print("\n" + "=" * 80)
    print("  SECTOR ROTATION & FUND FLOW DASHBOARD")
    print("=" * 80)

    # --- Sector Ranking ---
    if ranked:
        print(f"\n  Date: {ranked[0].date}")
    print(f"\n{'─' * 80}")
    print(f"  {'Sector':<22} {'SCS':>6} {'Chg%':>7} {'RSI':>6} {'ADX':>6} "
          f"{'Breadth200':>10} {'FundFlow':>14} {'RVOL':>6}")
    print(f"{'─' * 80}")

    for m in ranked:
        bar = _bar(m.composite_score, 100, 15)
        ff_str = _format_flow(m.total_fund_flow)
        print(
            f"  {m.sector:<22} {m.composite_score:>6.1f} {m.avg_change_pct:>+7.2f} "
            f"{m.avg_rsi:>6.1f} {m.avg_adx:>6.1f} {m.breadth_sma200:>9.0%} "
            f"{ff_str:>14} {m.avg_relative_volume:>6.2f}  {bar}"
        )

    # --- Fund Flow Summary ---
    print(f"\n{'─' * 80}")
    print("  FUND FLOW SUMMARY (推定資金フロー)")
    print(f"{'─' * 80}")
    flow_sorted = sorted(latest, key=lambda m: m.total_fund_flow, reverse=True)
    for m in flow_sorted:
        direction = "▲ 流入" if m.total_fund_flow > 0 else "▼ 流出"
        ff_str = _format_flow(m.total_fund_flow)
        print(f"  {direction} {m.sector:<22} {ff_str:>14}")

    # --- Breadth Analysis ---
    print(f"\n{'─' * 80}")
    print("  MARKET BREADTH (MA上の銘柄比率)")
    print(f"{'─' * 80}")
    print(f"  {'Sector':<22} {'> SMA20':>8} {'> SMA50':>8} {'> SMA200':>8}  Health")
    print(f"  {'─' * 60}")
    for m in ranked:
        health = _health_indicator(m.breadth_sma200, m.breadth_sma50)
        print(
            f"  {m.sector:<22} {m.breadth_sma20:>7.0%} {m.breadth_sma50:>7.0%} "
            f"{m.breadth_sma200:>7.0%}  {health}"
        )

    # --- Rotation Signals ---
    if rotations:
        print(f"\n{'─' * 80}")
        print("  SECTOR ROTATION SIGNALS")
        print(f"{'─' * 80}")
        for rot in rotations:
            print(f"\n  [{rot.date}]")
            if rot.rising_sectors:
                print(f"    ▲ 上昇: {', '.join(rot.rising_sectors)}")
            if rot.falling_sectors:
                print(f"    ▼ 下降: {', '.join(rot.falling_sectors)}")
            if rot.rotation_pairs:
                print(f"    ⟳ ローテーション検出:")
                for src, dst, stren in rot.rotation_pairs[:5]:
                    print(f"      {src} → {dst}  (強度: {stren:.1f})")

    # --- Trade Candidates ---
    print(f"\n{'─' * 80}")
    print("  TRADE CANDIDATES (売買候補)")
    print(f"{'─' * 80}")
    if not candidates:
        print("  (該当なし)")
    for i, c in enumerate(candidates, 1):
        icon = "🔼" if c.signal == "LONG" else "🔽"
        print(f"\n  {i}. {icon} {c.signal} {c.ticker} ({c.name})")
        print(f"     Sector: {c.sector} / {c.industry}")
        print(f"     Close: {c.close:.2f}  Chg: {c.change_pct:+.2f}%  "
              f"RSI: {c.rsi:.1f}  RVOL: {c.volume_ratio:.2f}")
        print(f"     Strength: {c.strength:.0f}/100")
        for r in c.reasons:
            print(f"       - {r}")

    print(f"\n{'=' * 80}\n")


def _bar(value: float, max_val: float, width: int = 15) -> str:
    filled = int(value / max_val * width)
    return "█" * filled + "░" * (width - filled)


def _format_flow(flow: float) -> str:
    abs_flow = abs(flow)
    if abs_flow >= 1e9:
        return f"{'+'if flow>=0 else '-'}${abs_flow/1e9:.2f}B"
    elif abs_flow >= 1e6:
        return f"{'+'if flow>=0 else '-'}${abs_flow/1e6:.1f}M"
    else:
        return f"{'+'if flow>=0 else '-'}${abs_flow:,.0f}"


def _health_indicator(breadth_200: float, breadth_50: float) -> str:
    if breadth_200 >= 0.8 and breadth_50 >= 0.8:
        return "◉ Very Strong"
    elif breadth_200 >= 0.6:
        return "● Strong"
    elif breadth_200 >= 0.4:
        return "◐ Neutral"
    elif breadth_200 >= 0.2:
        return "○ Weak"
    else:
        return "◌ Very Weak"


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_dashboard(
    all_metrics: list[list[SectorMetrics]],
    rotations: list[RotationSignal],
    candidates: list[TradeCandidate],
    output_path: Optional[str] = None,
):
    """matplotlibで分析ダッシュボードを描画する"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        try:
            import matplotlib_fontja  # noqa: F401 — 日本語フォント自動適用
        except ImportError:
            pass
    except ImportError:
        print("  [Warning] matplotlib not installed. Skipping chart generation.")
        return

    latest = all_metrics[-1] if all_metrics else []
    ranked = sorted(latest, key=lambda m: m.composite_score, reverse=True)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Sector Rotation & Fund Flow Dashboard", fontsize=16, fontweight="bold", y=0.98)

    # Grid: 3行 x 2列
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.93, bottom=0.04)

    sectors = [m.sector for m in ranked]
    scs_values = [m.composite_score for m in ranked]
    colors_scs = [_scs_color(v) for v in scs_values]

    # 1) Sector Composite Score
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.barh(sectors[::-1], scs_values[::-1], color=colors_scs[::-1], edgecolor="white", linewidth=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_title("Sector Composite Score (SCS)", fontweight="bold")
    ax1.axvline(50, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars1, scs_values[::-1]):
        ax1.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.1f}",
                 va="center", fontsize=8)

    # 2) Fund Flow
    ax2 = fig.add_subplot(gs[0, 1])
    flows = [m.total_fund_flow for m in ranked]
    flow_colors = ["#2ecc71" if f >= 0 else "#e74c3c" for f in flows]
    ax2.barh(sectors[::-1], [f / 1e6 for f in flows[::-1]],
             color=flow_colors[::-1], edgecolor="white", linewidth=0.5)
    ax2.set_title("Estimated Fund Flow ($M)", fontweight="bold")
    ax2.axvline(0, color="gray", linestyle="-", alpha=0.5)

    # 3) Breadth Analysis (stacked)
    ax3 = fig.add_subplot(gs[1, 0])
    b200 = [m.breadth_sma200 * 100 for m in ranked]
    b50 = [m.breadth_sma50 * 100 for m in ranked]
    b20 = [m.breadth_sma20 * 100 for m in ranked]
    x_pos = np.arange(len(sectors))
    w = 0.25
    ax3.bar(x_pos - w, b20[::-1] if False else b200, w, label="> SMA200", color="#2c3e50")
    ax3.bar(x_pos, b50, w, label="> SMA50", color="#2980b9")
    ax3.bar(x_pos + w, b20, w, label="> SMA20", color="#7fb3d8")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(sectors, rotation=45, ha="right", fontsize=7)
    ax3.set_ylim(0, 110)
    ax3.set_ylabel("%")
    ax3.set_title("Market Breadth (% above MA)", fontweight="bold")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.axhline(50, color="gray", linestyle="--", alpha=0.4)

    # 4) RSI vs ADX scatter
    ax4 = fig.add_subplot(gs[1, 1])
    rsival = [m.avg_rsi for m in ranked]
    adxval = [m.avg_adx for m in ranked]
    sizes = [m.total_market_cap / 1e10 for m in ranked]  # scale
    sizes = [max(50, min(s, 500)) for s in sizes]
    scatter = ax4.scatter(rsival, adxval, s=sizes, c=scs_values, cmap="RdYlGn",
                          edgecolors="gray", linewidth=0.5, alpha=0.8, vmin=30, vmax=70)
    for m in ranked:
        ax4.annotate(m.sector, (m.avg_rsi, m.avg_adx), fontsize=7,
                     textcoords="offset points", xytext=(5, 5))
    ax4.set_xlabel("Average RSI")
    ax4.set_ylabel("Average ADX")
    ax4.set_title("RSI vs ADX (bubble=MarketCap, color=SCS)", fontweight="bold")
    ax4.axvline(50, color="gray", linestyle="--", alpha=0.3)
    ax4.axhline(20, color="gray", linestyle="--", alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label="SCS", shrink=0.8)

    # 5) SCS Time-series (if multiple snapshots)
    if len(all_metrics) > 1:
        ax5 = fig.add_subplot(gs[2, 0])
        sector_series: dict[str, list[float]] = {}
        dates = []
        for snapshot in all_metrics:
            if snapshot:
                dates.append(snapshot[0].date)
            for m in snapshot:
                sector_series.setdefault(m.sector, []).append(m.composite_score)

        for sector, values in sector_series.items():
            ax5.plot(dates[:len(values)], values, marker="o", markersize=4, label=sector)
        ax5.set_title("SCS Time-Series (Sector Rotation Tracking)", fontweight="bold")
        ax5.set_ylabel("Composite Score")
        ax5.legend(fontsize=6, loc="center left", bbox_to_anchor=(1, 0.5))
        ax5.axhline(50, color="gray", linestyle="--", alpha=0.3)
    else:
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.text(0.5, 0.5, "More snapshots needed\nfor time-series view",
                 ha="center", va="center", fontsize=12, color="gray")
        ax5.set_title("SCS Time-Series", fontweight="bold")

    # 6) Trade Candidates table
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    ax6.set_title("Top Trade Candidates", fontweight="bold")
    if candidates:
        table_data = []
        for c in candidates[:8]:
            table_data.append([
                c.signal, c.ticker, c.sector[:15],
                f"{c.close:.1f}", f"{c.change_pct:+.2f}%",
                f"{c.rsi:.0f}", f"{c.strength:.0f}",
            ])
        col_labels = ["Signal", "Ticker", "Sector", "Close", "Chg%", "RSI", "Score"]
        table = ax6.table(cellText=table_data, colLabels=col_labels,
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        # 色付け
        for i, c in enumerate(candidates[:8]):
            color = "#d5f5e3" if c.signal == "LONG" else "#fadbd8"
            for j in range(len(col_labels)):
                table[(i + 1, j)].set_facecolor(color)
    else:
        ax6.text(0.5, 0.5, "No candidates", ha="center", va="center",
                 fontsize=12, color="gray")

    # Save
    out = output_path or "screener/sector_dashboard.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Dashboard saved to: {out}")


def _scs_color(score: float) -> str:
    if score >= 65:
        return "#27ae60"
    elif score >= 55:
        return "#2ecc71"
    elif score >= 45:
        return "#f39c12"
    elif score >= 35:
        return "#e67e22"
    else:
        return "#e74c3c"


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_analysis_csv(
    all_metrics: list[list[SectorMetrics]],
    candidates: list[TradeCandidate],
    output_dir: str = "screener",
):
    """分析結果をCSVとして出力する"""
    # Sector Metrics
    rows = []
    for snapshot in all_metrics:
        for m in snapshot:
            rows.append(vars(m))
    if rows:
        df_metrics = pd.DataFrame(rows)
        path = os.path.join(output_dir, "sector_metrics.csv")
        df_metrics.to_csv(path, index=False)
        print(f"  Sector metrics saved to: {path}")

    # Trade Candidates
    if candidates:
        cand_rows = []
        for c in candidates:
            cand_rows.append({
                "ticker": c.ticker,
                "name": c.name,
                "sector": c.sector,
                "industry": c.industry,
                "signal": c.signal,
                "strength": c.strength,
                "close": c.close,
                "change_pct": c.change_pct,
                "rsi": c.rsi,
                "volume_ratio": c.volume_ratio,
                "reasons": " | ".join(c.reasons),
            })
        df_cand = pd.DataFrame(cand_rows)
        path = os.path.join(output_dir, "trade_candidates.csv")
        df_cand.to_csv(path, index=False)
        print(f"  Trade candidates saved to: {path}")


# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sector Rotation & Fund Flow Analyzer for TradingView Screener CSV"
    )
    parser.add_argument(
        "snapshots_dir",
        help="Directory containing TradingView screener CSV exports",
    )
    parser.add_argument("--out", default=None, help="Output chart path (PNG)")
    parser.add_argument("--top", type=int, default=10, help="Number of trade candidates")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="SCS delta threshold for rotation detection")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    parser.add_argument("--export-csv", action="store_true", help="Export analysis to CSV")
    args = parser.parse_args()

    snap_dir = Path(args.snapshots_dir)
    if not snap_dir.is_dir():
        print(f"Error: {snap_dir} is not a directory")
        sys.exit(1)

    csv_files = sorted(snap_dir.glob("*.csv"),
                       key=lambda f: extract_date_from_filename(f))
    if not csv_files:
        print(f"Error: No CSV files found in {snap_dir}")
        sys.exit(1)

    # セクターCSV と テクニカルCSV をペアリング
    pairs = pair_snapshot_files(csv_files)
    sector_count = len(pairs)
    tech_count = sum(1 for _, t in pairs if t is not None)
    print(f"\n  Found {len(csv_files)} CSV file(s) in {snap_dir}")
    print(f"  Paired: {sector_count} sector file(s), {tech_count} technical file(s)")

    # 全スナップショットを処理
    all_metrics: list[list[SectorMetrics]] = []
    latest_df: Optional[pd.DataFrame] = None

    for sector_csv, tech_csv in pairs:
        date_label = extract_date_from_filename(sector_csv)
        tech_info = f" + {tech_csv.name}" if tech_csv else ""
        print(f"  Processing: {sector_csv.name}{tech_info} ({date_label})")
        df = load_snapshot(sector_csv, technical_filepath=tech_csv)
        metrics = compute_sector_metrics(df, date_label)
        all_metrics.append(metrics)
        latest_df = df

    # ローテーション検出
    rotations = detect_rotation(all_metrics, threshold=args.threshold)

    # 売買候補生成
    latest_rotation = rotations[-1] if rotations else None
    candidates = generate_trade_signals(
        latest_df, all_metrics[-1], latest_rotation, top_n=args.top
    )

    # レポート出力
    print_sector_dashboard(all_metrics, rotations, candidates)

    # チャート
    if not args.no_chart:
        plot_dashboard(all_metrics, rotations, candidates, args.out)

    # CSV 出力
    if args.export_csv:
        export_analysis_csv(all_metrics, candidates)


if __name__ == "__main__":
    main()
