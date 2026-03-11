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
import json
import os
import sys
from dataclasses import asdict, dataclass, field
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


@dataclass
class Alert:
    """異常検知アラート"""
    date: str
    level: str           # "CRITICAL", "WARNING", "INFO"
    category: str        # "FLOW_ANOMALY", "BROAD_SELLOFF", "BROAD_RALLY", "SCS_SPIKE"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class DiffEntry:
    """日次差分データ"""
    sector: str
    scs_now: float
    scs_prev: float
    scs_delta: float
    flow_now: float
    flow_prev: float
    flow_reversed: bool      # 前日と方向が反転
    breadth200_now: float
    breadth200_prev: float
    breadth200_delta: float
    rvol_now: float


# ---------------------------------------------------------------------------
# Cache (Incremental Processing)
# ---------------------------------------------------------------------------

CACHE_FILE = "screener/.sector_cache.json"


def _load_cache(cache_path: str = CACHE_FILE) -> dict:
    """キャッシュファイルを読み込む"""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_cache(cache: dict, cache_path: str = CACHE_FILE) -> None:
    """キャッシュファイルに書き込む"""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def _metrics_to_cache_entry(metrics: list[SectorMetrics]) -> list[dict]:
    """SectorMetrics リストをキャッシュ用 dict リストに変換"""
    return [asdict(m) for m in metrics]


def _cache_entry_to_metrics(entries: list[dict]) -> list[SectorMetrics]:
    """キャッシュ用 dict リストを SectorMetrics リストに復元"""
    result = []
    for e in entries:
        result.append(SectorMetrics(**e))
    return result


# ---------------------------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------------------------

REQUIRED_COLS = [
    "Ticker", "Sector", "Close", "Change %", "Volume",
    "Relative Volume", "Market Cap", "SMA20", "SMA50", "SMA200",
    "RSI", "MACD.macd", "ADX",
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
        "価格 - 通貨": "Currency",
        "時価総額 - 通貨": "Market Cap Currency",
        # テクニカル指標の日本語カラム (直接指標値)
        "単純移動平均線 (20)": "SMA20",
        "単純移動平均線 (50)": "SMA50",
        "単純移動平均線 (200)": "SMA200",
        "相対力指数 (14)": "RSI",
        "MACD レベル (12, 26)": "MACD.macd",
        "MACDシグナル (12, 26)": "MACD.signal",
        "平均方向性指数 (14)": "ADX",
        "アベレージ・トゥルー・レンジ (14)": "ATR",
        # TradingView テクニカル分析 CSV の日本語カラム (評価+オシレーター)
        "テクニカルの評価 1日": "Technical Rating",
        "移動平均線の評価 1日": "MA Rating",
        "オシレーターの評価 1日": "Oscillator Rating",
        "ローソク足パターン 1日": "Candlestick Pattern",
    }
    rename_map = {k: v for k, v in ja_col_map.items() if k in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    # カラム名の揺らぎを吸収 (英語/日本語カラムの表記揺れ対応)
    import re as _re
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
        # TradingView スクリーナーCSVの「〜 1日」付きカラム名に対応
        elif "sma" in lower and "単純移動平均" in c and "(20)" in c:
            col_map[c] = "SMA20"
        elif "sma" in lower and "単純移動平均" in c and "(50)" in c:
            col_map[c] = "SMA50"
        elif "sma" in lower and "単純移動平均" in c and "(200)" in c:
            col_map[c] = "SMA200"
        elif "rsi" in lower and "相対力指数" in c:
            col_map[c] = "RSI"
        elif "adx" in lower and "平均方向性指数" in c:
            col_map[c] = "ADX"
        elif "macd" in lower and "移動平均収束拡散" in c and "level" in lower:
            col_map[c] = "MACD.macd"
        elif "macd" in lower and "移動平均収束拡散" in c and "signal" in lower:
            col_map[c] = "MACD.signal"
        elif _re.match(r"mom\s*\(モメンタム\)\s*\(10\)", lower):
            col_map[c] = "Momentum"
        elif _re.match(r"ao\s*\(オーサム・オシレーター\)", lower):
            col_map[c] = "Awesome Oscillator"
        elif _re.match(r"cci\s*\(商品チャネル指数\)\s*\(20\)", lower):
            col_map[c] = "CCI"
        elif "stoch" in lower and "ストキャスティクス" in c and "%k" in lower:
            col_map[c] = "Stoch %K"
        elif "stoch" in lower and "ストキャスティクス" in c and "%d" in lower:
            col_map[c] = "Stoch %D"
    if col_map:
        df = df.rename(columns=col_map)

    return df


def _convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """数値カラムを変換する"""
    numeric_cols = [
        "Close", "Change %", "Volume", "Relative Volume", "Market Cap",
        "SMA20", "SMA50", "SMA200", "RSI", "MACD.macd", "MACD.signal",
        "ADX", "ATR", "P/E", "EPS (TTM)",
        "Momentum", "Awesome Oscillator", "CCI", "Stoch %K", "Stoch %D",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _currency_to_market(currency: str) -> str:
    """通貨コード → マーケットコード"""
    c = str(currency).strip().upper()
    if c == "JPY":
        return "JP"
    elif c == "USD":
        return "US"
    elif c in ("", "NAN"):
        return "US"
    else:
        return "OTHER"


def detect_market(df: pd.DataFrame) -> str:
    """
    DataFrameの通貨カラムから最頻マーケットを判別する (ファイル単位)。
    Returns: "US", "JP", or "OTHER"
    """
    if "Currency" not in df.columns:
        return "US"
    currencies = df["Currency"].dropna().str.upper()
    if currencies.empty:
        return "US"
    most_common = currencies.mode().iloc[0]
    return _currency_to_market(most_common)


def assign_market_per_row(df: pd.DataFrame) -> pd.DataFrame:
    """各行の通貨カラムからマーケットを判別し _market 列に設定する。"""
    if "Currency" in df.columns:
        df["_market"] = df["Currency"].fillna("USD").apply(_currency_to_market)
    else:
        df["_market"] = "US"
    return df


MARKET_LABELS = {
    "US": "US Stocks (米国株)",
    "JP": "JP Stocks (日本株)",
    "OTHER": "Other Markets",
}


def load_snapshot(filepath: str | Path) -> pd.DataFrame:
    """TradingView スクリーナー CSV を読み込み、正規化する。"""
    df = pd.read_csv(filepath)
    df = _normalize_columns(df)

    # 必須カラムチェック
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"  Warning: missing columns {missing} in {filepath}")

    df = _convert_numeric(df)

    # Sector カラムが存在しない場合は空のDataFrameを返す
    if "Sector" not in df.columns:
        print(f"  Skipping {filepath} (no Sector column)")
        return pd.DataFrame()

    # 行単位でマーケット判別
    df = assign_market_per_row(df)

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


def _deduplicate_by_date_and_market(csv_files: list[Path]) -> list[Path]:
    """
    同日・同マーケットのCSVが複数ある場合、必須カラムを最も多く含むファイルだけを残す。
    異なるマーケット (US/JP) のファイルは別々に保持する。
    """
    from collections import defaultdict

    # (date, market) → files のマッピングを構築
    key_to_files: dict[tuple[str, str], list[tuple[Path, int]]] = defaultdict(list)
    for f in csv_files:
        date = extract_date_from_filename(f)
        try:
            hdr = pd.read_csv(f, nrows=5)
            hdr = _normalize_columns(hdr)
            score = sum(1 for c in REQUIRED_COLS if c in hdr.columns)
            if "Sector" not in hdr.columns:
                score = -1
            market = detect_market(hdr)
        except Exception:
            score = -1
            market = "US"
        key_to_files[(date, market)].append((f, score))

    result: list[Path] = []
    for (date, market), entries in sorted(key_to_files.items()):
        # スコアが -1 のみのグループはスキップ
        valid = [(f, s) for f, s in entries if s >= 0]
        if not valid:
            continue
        if len(valid) == 1:
            result.append(valid[0][0])
        else:
            best_file, best_score = max(valid, key=lambda x: x[1])
            print(f"  Date {date} [{market}]: {len(entries)} files, selected {best_file.name} "
                  f"({best_score}/{len(REQUIRED_COLS)} required cols)")
            result.append(best_file)
    return result


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
        # ペニー株の極端な変動率を ±50% でクランプしてから平均
        avg_change = group["Change %"].clip(lower=-50, upper=50).mean() if "Change %" in group else 0.0
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
        # ペニー株の極端な変動率（+999,900% 等）が全体を歪めるため ±50% でクランプ
        fund_flow = 0.0
        if "Volume" in group and "Change %" in group:
            clamped_change = group["Change %"].clip(lower=-50, upper=50)
            fund_flow = (group["Volume"] * clamped_change / 100.0).sum()

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
# Daily Diff Summary
# ---------------------------------------------------------------------------

def compute_daily_diff(
    current: list[SectorMetrics],
    previous: list[SectorMetrics],
) -> list[DiffEntry]:
    """前日との差分を計算する"""
    prev_map = {m.sector: m for m in previous}
    diffs = []
    for m in current:
        p = prev_map.get(m.sector)
        if not p:
            continue
        flow_reversed = (
            (m.total_fund_flow > 0 and p.total_fund_flow < 0) or
            (m.total_fund_flow < 0 and p.total_fund_flow > 0)
        )
        diffs.append(DiffEntry(
            sector=m.sector,
            scs_now=m.composite_score,
            scs_prev=p.composite_score,
            scs_delta=round(m.composite_score - p.composite_score, 2),
            flow_now=m.total_fund_flow,
            flow_prev=p.total_fund_flow,
            flow_reversed=flow_reversed,
            breadth200_now=m.breadth_sma200,
            breadth200_prev=p.breadth_sma200,
            breadth200_delta=round(m.breadth_sma200 - p.breadth_sma200, 4),
            rvol_now=m.avg_relative_volume,
        ))
    return sorted(diffs, key=lambda d: d.scs_delta, reverse=True)


def print_daily_diff(diffs: list[DiffEntry], date: str, market_label: str = "") -> None:
    """日次差分サマリーを出力"""
    if not diffs:
        return
    header = "DAILY DIFF SUMMARY (前日比)"
    if market_label:
        header += f"  [{market_label}]"
    print(f"\n{'─' * 80}")
    print(f"  {header}  Date: {date}")
    print(f"{'─' * 80}")
    print(f"  {'Sector':<22} {'SCS':>6} {'ΔSCS':>7} {'FundFlow':>12} {'Prev Flow':>12} {'Rev':>4} "
          f"{'B200':>6} {'ΔB200':>7}")
    print(f"  {'─' * 76}")
    for d in diffs:
        ff_str = _format_flow(d.flow_now)
        pf_str = _format_flow(d.flow_prev)
        rev = "⟲" if d.flow_reversed else ""
        delta_marker = "▲" if d.scs_delta > 0 else "▼" if d.scs_delta < 0 else "─"
        print(
            f"  {d.sector:<22} {d.scs_now:>6.1f} {delta_marker}{d.scs_delta:>+6.1f} "
            f"{ff_str:>12} {pf_str:>12} {rev:>4} "
            f"{d.breadth200_now:>5.0%} {d.breadth200_delta:>+6.0%}"
        )


# ---------------------------------------------------------------------------
# Alert / Anomaly Detection
# ---------------------------------------------------------------------------

def detect_alerts(
    current: list[SectorMetrics],
    previous: Optional[list[SectorMetrics]] = None,
) -> list[Alert]:
    """異常値・注目イベントを自動検出する"""
    alerts: list[Alert] = []
    if not current:
        return alerts

    date = current[0].date

    # --- 全面安 / 全面高 検出 ---
    changes = [m.avg_change_pct for m in current if m.stock_count > 5]
    if changes:
        neg_pct = sum(1 for c in changes if c < 0) / len(changes)
        pos_pct = sum(1 for c in changes if c > 0) / len(changes)
        avg_chg = np.mean(changes)
        if neg_pct >= 0.85:
            alerts.append(Alert(
                date=date, level="CRITICAL", category="BROAD_SELLOFF",
                message=f"全面安検出: {neg_pct:.0%}のセクターが下落 (平均{avg_chg:+.2f}%)",
                details={"neg_pct": neg_pct, "avg_change": avg_chg},
            ))
        elif pos_pct >= 0.85:
            alerts.append(Alert(
                date=date, level="CRITICAL", category="BROAD_RALLY",
                message=f"全面高検出: {pos_pct:.0%}のセクターが上昇 (平均{avg_chg:+.2f}%)",
                details={"pos_pct": pos_pct, "avg_change": avg_chg},
            ))

    # --- RVOL異常値 (> 5.0) ---
    for m in current:
        if m.avg_relative_volume > 5.0:
            alerts.append(Alert(
                date=date, level="WARNING", category="FLOW_ANOMALY",
                message=f"{m.sector}: RVOL={m.avg_relative_volume:.2f} (異常な出来高)",
                details={"sector": m.sector, "rvol": m.avg_relative_volume},
            ))

    # --- FundFlow異常値 (前日比10倍超 or 方向急反転+大量) ---
    if previous:
        prev_map = {m.sector: m for m in previous}
        for m in current:
            p = prev_map.get(m.sector)
            if not p:
                continue
            # 前日比で大きな方向転換 + 絶対値が大きい
            if p.total_fund_flow != 0:
                ratio = abs(m.total_fund_flow / p.total_fund_flow) if p.total_fund_flow != 0 else 0
                reversed_dir = (m.total_fund_flow > 0) != (p.total_fund_flow > 0)
                if reversed_dir and abs(m.total_fund_flow) > 50e6:
                    alerts.append(Alert(
                        date=date, level="WARNING", category="FLOW_ANOMALY",
                        message=(f"{m.sector}: 資金フロー急反転 "
                                 f"{_format_flow(p.total_fund_flow)} → {_format_flow(m.total_fund_flow)}"),
                        details={"sector": m.sector, "prev_flow": p.total_fund_flow,
                                 "curr_flow": m.total_fund_flow},
                    ))
                elif ratio > 10 and abs(m.total_fund_flow) > 10e6:
                    alerts.append(Alert(
                        date=date, level="INFO", category="FLOW_ANOMALY",
                        message=(f"{m.sector}: 資金フロー{ratio:.0f}倍増 "
                                 f"{_format_flow(p.total_fund_flow)} → {_format_flow(m.total_fund_flow)}"),
                        details={"sector": m.sector, "ratio": ratio},
                    ))

            # SCS急変 (±15pt以上)
            scs_delta = m.composite_score - p.composite_score
            if abs(scs_delta) >= 15:
                direction = "急騰" if scs_delta > 0 else "急落"
                alerts.append(Alert(
                    date=date, level="WARNING", category="SCS_SPIKE",
                    message=f"{m.sector}: SCS {direction} {p.composite_score:.1f} → {m.composite_score:.1f} (Δ{scs_delta:+.1f})",
                    details={"sector": m.sector, "delta": scs_delta},
                ))

    # ソート: CRITICAL > WARNING > INFO
    level_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
    alerts.sort(key=lambda a: level_order.get(a.level, 9))
    return alerts


def print_alerts(alerts: list[Alert]) -> None:
    """アラートを出力する"""
    if not alerts:
        return
    print(f"\n{'─' * 80}")
    print("  ⚠ ALERTS (異常検知)")
    print(f"{'─' * 80}")
    icons = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}
    for a in alerts:
        icon = icons.get(a.level, "⚪")
        print(f"  {icon} [{a.level}] {a.message}")


# ---------------------------------------------------------------------------
# Watchlist Tracking
# ---------------------------------------------------------------------------

WATCHLIST_FILE = "screener/.watchlist.json"


def _load_watchlist(path: str = WATCHLIST_FILE) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"candidates": []}


def _save_watchlist(wl: dict, path: str = WATCHLIST_FILE) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(wl, f, ensure_ascii=False, indent=2)


def update_watchlist(
    candidates: list[TradeCandidate],
    latest_df: pd.DataFrame,
    date: str,
    market: str,
) -> dict:
    """ウォッチリストを更新し、過去候補のパフォーマンスを追跡する"""
    wl = _load_watchlist()

    # 1) 既存エントリの現在価格を更新
    ticker_price = {}
    if "Ticker" in latest_df.columns and "Close" in latest_df.columns:
        for _, row in latest_df[["Ticker", "Close"]].dropna().iterrows():
            ticker_price[row["Ticker"]] = row["Close"]

    for entry in wl["candidates"]:
        if entry.get("market") != market:
            continue
        t = entry["ticker"]
        if t in ticker_price and ticker_price[t] > 0:
            entry_price = entry.get("entry_price", 0)
            if entry_price > 0:
                current = ticker_price[t]
                pnl = ((current / entry_price) - 1) * 100
                if entry["signal"] == "SHORT":
                    pnl = -pnl
                entry["current_price"] = current
                entry["pnl_pct"] = round(pnl, 2)
                entry["last_seen"] = date
                entry["days_held"] = entry.get("days_held", 0) + 1

    # 2) 新規候補を追加（重複回避）
    existing_tickers = {(e["ticker"], e["market"]) for e in wl["candidates"]}
    for c in candidates:
        if (c.ticker, market) not in existing_tickers:
            wl["candidates"].append({
                "ticker": c.ticker,
                "name": c.name,
                "sector": c.sector,
                "signal": c.signal,
                "strength": c.strength,
                "entry_price": c.close,
                "entry_date": date,
                "current_price": c.close,
                "pnl_pct": 0.0,
                "last_seen": date,
                "days_held": 0,
                "market": market,
            })

    # 3) 古いエントリを削除 (30日以上追跡したもの)
    wl["candidates"] = [e for e in wl["candidates"] if e.get("days_held", 0) <= 30]

    _save_watchlist(wl)
    return wl


def print_watchlist(wl: dict, market: str, market_label: str = "") -> None:
    """ウォッチリストのパフォーマンスを出力"""
    entries = [e for e in wl.get("candidates", [])
               if e.get("market") == market and e.get("days_held", 0) > 0]
    if not entries:
        return

    header = "WATCHLIST PERFORMANCE (過去候補の追跡)"
    if market_label:
        header += f"  [{market_label}]"
    print(f"\n{'─' * 80}")
    print(f"  {header}")
    print(f"{'─' * 80}")

    longs = [e for e in entries if e["signal"] == "LONG"]
    shorts = [e for e in entries if e["signal"] == "SHORT"]

    for label, group in [("LONG", longs), ("SHORT", shorts)]:
        if not group:
            continue
        group_sorted = sorted(group, key=lambda e: e.get("pnl_pct", 0), reverse=True)
        print(f"\n  [{label}]")
        print(f"  {'Ticker':<10} {'Sector':<18} {'Entry':>8} {'Current':>8} {'P&L':>7} {'Days':>5} {'Entry Date'}")
        print(f"  {'─' * 72}")
        for e in group_sorted[:10]:
            pnl = e.get("pnl_pct", 0)
            pnl_str = f"{pnl:+.1f}%"
            marker = "✅" if pnl > 0 else "❌" if pnl < -5 else "➖"
            print(
                f"  {marker} {e['ticker']:<8} {e['sector']:<18} "
                f"{e['entry_price']:>8.2f} {e.get('current_price', 0):>8.2f} "
                f"{pnl_str:>7} {e.get('days_held', 0):>5} {e.get('entry_date', '')}"
            )

    # サマリー
    if entries:
        avg_pnl = np.mean([e.get("pnl_pct", 0) for e in entries])
        win_rate = sum(1 for e in entries if e.get("pnl_pct", 0) > 0) / len(entries) * 100
        print(f"\n  Summary: {len(entries)} positions | Avg P&L: {avg_pnl:+.1f}% | Win Rate: {win_rate:.0f}%")


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

    # SMA データが存在するかチェック (全銘柄の大半で SMA50 が有効か)
    has_sma = "SMA50" in df.columns and df["SMA50"].notna().mean() > 0.3
    has_macd = "MACD.macd" in df.columns and df["MACD.macd"].notna().mean() > 0.3
    has_adx = "ADX" in df.columns and df["ADX"].notna().mean() > 0.3
    # テクニカル評価カラム (Technical Rating) が利用可能か
    has_tech_rating = "Technical Rating" in df.columns
    has_momentum = "Momentum" in df.columns
    has_cci = "CCI" in df.columns
    has_stoch = "Stoch %K" in df.columns

    for _, row in df.iterrows():
        sector = row.get("Sector", "")
        ticker = row.get("Ticker", "")
        if not sector or not ticker:
            continue

        close = row.get("Close", 0)
        change = row.get("Change %", 0)
        rsi = row.get("RSI", 50)
        rvol = row.get("Relative Volume", 1.0)
        sma50 = row.get("SMA50", None)
        sma200 = row.get("SMA200", None)
        adx = row.get("ADX", 0)
        macd = row.get("MACD.macd", 0)
        macd_sig = row.get("MACD.signal", 0)

        # テクニカルCSV由来の指標
        tech_rating = str(row.get("Technical Rating", "")).strip()
        ma_rating = str(row.get("MA Rating", "")).strip()
        momentum = row.get("Momentum", 0) if has_momentum else 0
        cci = row.get("CCI", 0) if has_cci else 0
        stoch_k = row.get("Stoch %K", 50) if has_stoch else 50

        # NaN チェック
        if pd.isna(rsi):
            rsi = 50
        if pd.isna(momentum):
            momentum = 0
        if pd.isna(cci):
            cci = 0
        if pd.isna(stoch_k):
            stoch_k = 50
        if pd.isna(rvol):
            rvol = 1.0

        reasons = []
        signal = None
        strength = 0.0

        # --- LONG 候補 ---
        if sector in strong_sectors:
            is_long = True
            reasons.append(f"強セクター (SCS={scs_map.get(sector, 0):.1f})")

            # SMA ベースのトレンド判定 (利用可能な場合)
            if has_sma and sma50 is not None and not pd.isna(sma50):
                if close > sma50:
                    strength += 20
                    reasons.append("Close > SMA50 (中期上昇)")
                else:
                    is_long = False
                if sma200 is not None and not pd.isna(sma200) and close > sma200:
                    strength += 15
                    reasons.append("Close > SMA200 (長期上昇)")
            else:
                # SMA がない場合: Technical Rating / MA Rating で代替
                if has_tech_rating:
                    if tech_rating in ("買い", "強い買い"):
                        strength += 20
                        reasons.append(f"テクニカル評価: {tech_rating}")
                    elif tech_rating == "中立":
                        strength += 5
                    else:
                        is_long = False
                if ma_rating in ("買い", "強い買い"):
                    strength += 10
                    reasons.append(f"MA評価: {ma_rating}")

            # RSI
            if 40 < rsi < 70:
                strength += 15
                reasons.append(f"RSI={rsi:.1f} (適正レンジ)")
            elif rsi >= 70:
                strength -= 10
                reasons.append(f"RSI={rsi:.1f} (過熱注意)")

            # 出来高
            if rvol > 1.0:
                strength += 10
                reasons.append(f"RVOL={rvol:.2f} (出来高増加)")

            # MACD (利用可能な場合)
            if has_macd and macd > macd_sig:
                strength += 10
                reasons.append("MACD > Signal (上昇モメンタム)")

            # ADX (利用可能な場合)
            if has_adx and adx > 20:
                strength += 10
                reasons.append(f"ADX={adx:.1f} (トレンド明確)")

            # Momentum / CCI 代替指標
            if has_momentum and momentum > 0:
                strength += 5
                reasons.append(f"Momentum={momentum:.2f} (上昇)")
            if has_cci and cci > 100:
                strength += 5
                reasons.append(f"CCI={cci:.0f} (強気)")

            # ローテーション
            if sector in rising_set:
                strength += 15
                reasons.append("ローテーション流入中")

            if is_long and strength > 0:
                signal = "LONG"

        # --- SHORT 候補 ---
        elif sector in weak_sectors:
            is_short = True
            reasons.append(f"弱セクター (SCS={scs_map.get(sector, 0):.1f})")

            # SMA ベースのトレンド判定 (利用可能な場合)
            if has_sma and sma50 is not None and not pd.isna(sma50):
                if close < sma50:
                    strength += 20
                    reasons.append("Close < SMA50 (中期下降)")
                else:
                    is_short = False
                if sma200 is not None and not pd.isna(sma200) and close < sma200:
                    strength += 15
                    reasons.append("Close < SMA200 (長期下降)")
            else:
                # SMA がない場合: Technical Rating / MA Rating で代替
                if has_tech_rating:
                    if tech_rating in ("売り", "強い売り"):
                        strength += 20
                        reasons.append(f"テクニカル評価: {tech_rating}")
                    elif tech_rating == "中立":
                        strength += 5
                    else:
                        is_short = False
                if ma_rating in ("売り", "強い売り"):
                    strength += 10
                    reasons.append(f"MA評価: {ma_rating}")

            # RSI
            if 30 < rsi < 60:
                strength += 15
                reasons.append(f"RSI={rsi:.1f} (まだ売り余地)")
            elif rsi <= 30:
                strength -= 10
                reasons.append(f"RSI={rsi:.1f} (売られ過ぎ注意)")

            # 出来高
            if rvol > 1.0:
                strength += 10
                reasons.append(f"RVOL={rvol:.2f} (売り圧力)")

            # MACD (利用可能な場合)
            if has_macd and macd < macd_sig:
                strength += 10
                reasons.append("MACD < Signal (下降モメンタム)")

            # ADX (利用可能な場合)
            if has_adx and adx > 20:
                strength += 10
                reasons.append(f"ADX={adx:.1f} (トレンド明確)")

            # Momentum / CCI 代替指標
            if has_momentum and momentum < 0:
                strength += 5
                reasons.append(f"Momentum={momentum:.2f} (下降)")
            if has_cci and cci < -100:
                strength += 5
                reasons.append(f"CCI={cci:.0f} (弱気)")

            # ローテーション
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

    # LONG / SHORT を分離して各 top_n 件
    longs = sorted([c for c in candidates if c.signal == "LONG"],
                   key=lambda c: c.strength, reverse=True)[:top_n]
    shorts = sorted([c for c in candidates if c.signal == "SHORT"],
                    key=lambda c: c.strength, reverse=True)[:top_n]
    return longs + shorts


# ---------------------------------------------------------------------------
# Report Printer
# ---------------------------------------------------------------------------

def print_sector_dashboard(
    all_metrics: list[list[SectorMetrics]],
    rotations: list[RotationSignal],
    candidates: list[TradeCandidate],
    market_label: str = "",
    alerts: Optional[list[Alert]] = None,
    diffs: Optional[list[DiffEntry]] = None,
):
    """ターミナル向けの分析レポートを出力"""

    latest = all_metrics[-1] if all_metrics else []
    ranked = sorted(latest, key=lambda m: m.composite_score, reverse=True)

    title = "SECTOR ROTATION & FUND FLOW DASHBOARD"
    if market_label:
        title += f"  [{market_label}]"
    print("\n" + "=" * 80)
    print(f"  {title}")
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
    long_cands = [c for c in candidates if c.signal == "LONG"]
    short_cands = [c for c in candidates if c.signal == "SHORT"]

    for section_label, section_cands, icon in [
        ("LONG CANDIDATES (買い候補)", long_cands, "🔼"),
        ("SHORT CANDIDATES (売り候補)", short_cands, "🔽"),
    ]:
        print(f"\n{'─' * 80}")
        print(f"  {section_label}")
        print(f"{'─' * 80}")
        if not section_cands:
            print("  (該当なし)")
        for i, c in enumerate(section_cands, 1):
            print(f"\n  {i}. {icon} {c.signal} {c.ticker} ({c.name})")
            print(f"     Sector: {c.sector} / {c.industry}")
            print(f"     Close: {c.close:.2f}  Chg: {c.change_pct:+.2f}%  "
                  f"RSI: {c.rsi:.1f}  RVOL: {c.volume_ratio:.2f}")
            print(f"     Strength: {c.strength:.0f}/100")
            for r in c.reasons:
                print(f"       - {r}")

    # --- Alerts ---
    if alerts:
        print_alerts(alerts)

    # --- Daily Diff ---
    if diffs:
        date = ranked[0].date if ranked else "?"
        print_daily_diff(diffs, date, market_label=market_label)

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
    title_suffix: str = "",
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

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f"Sector Rotation & Fund Flow Dashboard{title_suffix}",
                 fontsize=16, fontweight="bold", y=0.98)

    # Grid: 4行 x 2列
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.96, top=0.94, bottom=0.03,
                          height_ratios=[1, 1, 1, 0.8])

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

    # 6) LONG Candidates table
    long_cands = [c for c in candidates if c.signal == "LONG"]
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.axis("off")
    ax6.set_title("LONG Candidates (買い候補)", fontweight="bold")
    if long_cands:
        table_data = []
        for c in long_cands[:8]:
            table_data.append([
                c.ticker, c.sector[:15],
                f"{c.close:.1f}", f"{c.change_pct:+.2f}%",
                f"{c.rsi:.0f}", f"{c.strength:.0f}",
            ])
        col_labels = ["Ticker", "Sector", "Close", "Chg%", "RSI", "Score"]
        table = ax6.table(cellText=table_data, colLabels=col_labels,
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        for i in range(len(long_cands[:8])):
            for j in range(len(col_labels)):
                table[(i + 1, j)].set_facecolor("#d5f5e3")
    else:
        ax6.text(0.5, 0.5, "No LONG candidates", ha="center", va="center",
                 fontsize=12, color="gray")

    # 7) SHORT Candidates table
    short_cands = [c for c in candidates if c.signal == "SHORT"]
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis("off")
    ax7.set_title("SHORT Candidates (売り候補)", fontweight="bold")
    if short_cands:
        table_data = []
        for c in short_cands[:8]:
            table_data.append([
                c.ticker, c.sector[:15],
                f"{c.close:.1f}", f"{c.change_pct:+.2f}%",
                f"{c.rsi:.0f}", f"{c.strength:.0f}",
            ])
        col_labels = ["Ticker", "Sector", "Close", "Chg%", "RSI", "Score"]
        table = ax7.table(cellText=table_data, colLabels=col_labels,
                          loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.4)
        for i in range(len(short_cands[:8])):
            for j in range(len(col_labels)):
                table[(i + 1, j)].set_facecolor("#fadbd8")
    else:
        ax7.text(0.5, 0.5, "No SHORT candidates", ha="center", va="center",
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
    suffix: str = "",
):
    """分析結果をCSVとして出力する"""
    # Sector Metrics
    rows = []
    for snapshot in all_metrics:
        for m in snapshot:
            rows.append(vars(m))
    if rows:
        df_metrics = pd.DataFrame(rows)
        path = os.path.join(output_dir, f"sector_metrics{suffix}.csv")
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
        path = os.path.join(output_dir, f"trade_candidates{suffix}.csv")
        df_cand.to_csv(path, index=False)
        print(f"  Trade candidates saved to: {path}")


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

def generate_html_report(
    all_metrics: list[list[SectorMetrics]],
    candidates: list[TradeCandidate],
    alerts: list[Alert],
    diffs: list[DiffEntry],
    rotations: list[RotationSignal],
    chart_path: Optional[str] = None,
    market_label: str = "",
    watchlist: Optional[dict] = None,
    market: str = "",
) -> str:
    """分析結果をHTMLレポートとして生成"""
    import base64

    latest = all_metrics[-1] if all_metrics else []
    ranked = sorted(latest, key=lambda m: m.composite_score, reverse=True)
    date = ranked[0].date if ranked else "?"

    # チャート画像をbase64エンコード
    chart_b64 = ""
    if chart_path and os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode()

    long_cands = [c for c in candidates if c.signal == "LONG"]
    short_cands = [c for c in candidates if c.signal == "SHORT"]

    def flow_html(v: float) -> str:
        color = "#27ae60" if v >= 0 else "#e74c3c"
        return f'<span style="color:{color}">{_format_flow(v)}</span>'

    def delta_html(v: float) -> str:
        color = "#27ae60" if v > 0 else "#e74c3c" if v < 0 else "#888"
        return f'<span style="color:{color}">{v:+.1f}</span>'

    # Sector table rows
    sector_rows = ""
    for m in ranked:
        sector_rows += f"""<tr>
            <td>{m.sector}</td><td><b>{m.composite_score:.1f}</b></td>
            <td>{m.avg_change_pct:+.2f}%</td><td>{m.avg_rsi:.1f}</td>
            <td>{m.breadth_sma200:.0%}</td><td>{flow_html(m.total_fund_flow)}</td>
            <td>{m.avg_relative_volume:.2f}</td></tr>"""

    # Diff rows
    diff_rows = ""
    if diffs:
        for d in diffs:
            rev = "⟲" if d.flow_reversed else ""
            diff_rows += f"""<tr>
                <td>{d.sector}</td><td>{d.scs_now:.1f}</td>
                <td>{delta_html(d.scs_delta)}</td>
                <td>{flow_html(d.flow_now)}</td><td>{flow_html(d.flow_prev)}</td>
                <td>{rev}</td><td>{d.breadth200_now:.0%}</td>
                <td>{delta_html(d.breadth200_delta * 100)}</td></tr>"""

    # Alert rows
    alert_rows = ""
    icons = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🔵"}
    for a in alerts:
        alert_rows += f"<tr><td>{icons.get(a.level, '')}</td><td>{a.level}</td><td>{a.message}</td></tr>"

    # Candidate rows
    def cand_rows(cands: list[TradeCandidate]) -> str:
        rows = ""
        for c in cands:
            rows += f"""<tr><td>{c.ticker}</td><td>{c.name}</td><td>{c.sector}</td>
                <td>{c.close:.2f}</td><td>{c.change_pct:+.2f}%</td>
                <td>{c.rsi:.1f}</td><td>{c.strength:.0f}</td></tr>"""
        return rows

    # Watchlist rows
    wl_rows = ""
    if watchlist:
        entries = [e for e in watchlist.get("candidates", [])
                   if e.get("market") == market and e.get("days_held", 0) > 0]
        for e in sorted(entries, key=lambda x: x.get("pnl_pct", 0), reverse=True):
            pnl = e.get("pnl_pct", 0)
            color = "#27ae60" if pnl > 0 else "#e74c3c" if pnl < -5 else "#888"
            wl_rows += f"""<tr><td>{e['signal']}</td><td>{e['ticker']}</td>
                <td>{e['sector']}</td><td>{e['entry_price']:.2f}</td>
                <td>{e.get('current_price', 0):.2f}</td>
                <td style="color:{color}">{pnl:+.1f}%</td>
                <td>{e.get('days_held', 0)}</td><td>{e.get('entry_date', '')}</td></tr>"""

    # Latest rotation
    latest_rot = rotations[-1] if rotations else None
    rotation_html = ""
    if latest_rot:
        if latest_rot.rising_sectors:
            rotation_html += f"<p>▲ 上昇: {', '.join(latest_rot.rising_sectors)}</p>"
        if latest_rot.falling_sectors:
            rotation_html += f"<p>▼ 下降: {', '.join(latest_rot.falling_sectors)}</p>"
        if latest_rot.rotation_pairs:
            rotation_html += "<ul>"
            for src, dst, stren in latest_rot.rotation_pairs[:5]:
                rotation_html += f"<li>{src} → {dst} (強度: {stren:.1f})</li>"
            rotation_html += "</ul>"

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sector Analysis — {market_label} — {date}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #1a1a2e; color: #e0e0e0; }}
  h1 {{ color: #00d4ff; border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }}
  h2 {{ color: #7fdbca; margin-top: 30px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }}
  th {{ background: #16213e; color: #00d4ff; padding: 8px 10px; text-align: left; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #2a2a4a; }}
  tr:hover {{ background: #16213e; }}
  .alert-box {{ background: #2a1a1a; border-left: 4px solid #e74c3c; padding: 10px; margin: 5px 0; }}
  .alert-box.warning {{ border-left-color: #f39c12; background: #2a2a1a; }}
  .alert-box.info {{ border-left-color: #3498db; background: #1a1a2a; }}
  .chart-img {{ width: 100%; max-width: 1100px; margin: 15px 0; border-radius: 8px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 768px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Sector Rotation & Fund Flow — {market_label}</h1>
<p>Date: {date} | Sectors: {len(ranked)}</p>

{"<h2>⚠ Alerts</h2>" + alert_rows if alert_rows else ""}
{('<table><tr><th></th><th>Level</th><th>Message</th></tr>' + alert_rows + '</table>') if alert_rows else ''}

<h2>Sector Ranking</h2>
<table>
<tr><th>Sector</th><th>SCS</th><th>Chg%</th><th>RSI</th><th>Breadth200</th><th>FundFlow</th><th>RVOL</th></tr>
{sector_rows}
</table>

{"<h2>Daily Diff (前日比)</h2><table><tr><th>Sector</th><th>SCS</th><th>ΔSCS</th><th>Flow</th><th>Prev Flow</th><th>Rev</th><th>B200</th><th>ΔB200</th></tr>" + diff_rows + "</table>" if diff_rows else ""}

<h2>Sector Rotation ({date})</h2>
{rotation_html if rotation_html else "<p>No significant rotation detected.</p>"}

<div class="grid">
<div>
<h2>LONG Candidates</h2>
<table><tr><th>Ticker</th><th>Name</th><th>Sector</th><th>Close</th><th>Chg%</th><th>RSI</th><th>Score</th></tr>
{cand_rows(long_cands)}
</table>
</div>
<div>
<h2>SHORT Candidates</h2>
<table><tr><th>Ticker</th><th>Name</th><th>Sector</th><th>Close</th><th>Chg%</th><th>RSI</th><th>Score</th></tr>
{cand_rows(short_cands)}
</table>
</div>
</div>

{"<h2>Watchlist Performance</h2><table><tr><th>Signal</th><th>Ticker</th><th>Sector</th><th>Entry</th><th>Current</th><th>P&L</th><th>Days</th><th>Entry Date</th></tr>" + wl_rows + "</table>" if wl_rows else ""}

{'<h2>Dashboard</h2><img class="chart-img" src="data:image/png;base64,' + chart_b64 + '"/>' if chart_b64 else ''}

<footer style="margin-top:40px;color:#666;font-size:11px;">
Generated by Sector Rotation & Fund Flow Analyzer
</footer>
</body></html>"""
    return html


# ---------------------------------------------------------------------------
# Diff Mode (--diff)
# ---------------------------------------------------------------------------

def print_diff_report(
    current_metrics: list[SectorMetrics],
    previous_metrics: list[SectorMetrics],
    diffs: list[DiffEntry],
    alerts: list[Alert],
    candidates: list[TradeCandidate],
    market_label: str = "",
) -> None:
    """直近2日の差分に特化した簡潔レポート"""
    date = current_metrics[0].date if current_metrics else "?"
    prev_date = previous_metrics[0].date if previous_metrics else "?"

    title = f"DIFF REPORT: {prev_date} → {date}"
    if market_label:
        title += f"  [{market_label}]"
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # アラート (最重要)
    if alerts:
        print_alerts(alerts)

    # SCS変動の大きいセクター TOP5 (上昇 + 下降)
    rising = [d for d in diffs if d.scs_delta > 0][:5]
    falling = [d for d in diffs if d.scs_delta < 0][-5:]

    if rising:
        print(f"\n  ▲ SCS上昇 TOP{len(rising)}:")
        for d in rising:
            print(f"    {d.sector:<22} {d.scs_prev:.1f} → {d.scs_now:.1f} ({d.scs_delta:+.1f})")

    if falling:
        print(f"\n  ▼ SCS下降 TOP{len(falling)}:")
        for d in reversed(falling):
            print(f"    {d.sector:<22} {d.scs_prev:.1f} → {d.scs_now:.1f} ({d.scs_delta:+.1f})")

    # 資金フロー反転セクター
    reversals = [d for d in diffs if d.flow_reversed]
    if reversals:
        print(f"\n  ⟲ 資金フロー反転 ({len(reversals)}セクター):")
        for d in reversals:
            print(f"    {d.sector:<22} {_format_flow(d.flow_prev):>12} → {_format_flow(d.flow_now):>12}")

    # Breadth200 急変
    breadth_changes = sorted(diffs, key=lambda d: abs(d.breadth200_delta), reverse=True)[:5]
    significant = [d for d in breadth_changes if abs(d.breadth200_delta) >= 0.05]
    if significant:
        print(f"\n  📊 Breadth200 急変:")
        for d in significant:
            print(f"    {d.sector:<22} {d.breadth200_prev:.0%} → {d.breadth200_now:.0%} ({d.breadth200_delta:+.0%})")

    # 候補サマリー (簡潔版)
    long_cands = [c for c in candidates if c.signal == "LONG"][:3]
    short_cands = [c for c in candidates if c.signal == "SHORT"][:3]

    if long_cands:
        print(f"\n  🔼 LONG TOP3:")
        for c in long_cands:
            print(f"    {c.ticker:<10} {c.sector:<18} {c.close:>8.2f}  RSI={c.rsi:.0f}  Score={c.strength:.0f}")

    if short_cands:
        print(f"\n  🔽 SHORT TOP3:")
        for c in short_cands:
            print(f"    {c.ticker:<10} {c.sector:<18} {c.close:>8.2f}  RSI={c.rsi:.0f}  Score={c.strength:.0f}")

    print(f"\n{'=' * 80}\n")


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
    parser.add_argument("--no-cache", action="store_true", help="Disable incremental cache")
    parser.add_argument("--diff", action="store_true",
                        help="Show only diff report (last 2 days)")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--no-watchlist", action="store_true", help="Disable watchlist tracking")
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

    # 同日・同マーケットのファイルが複数ある場合、最も完全なものだけを使う
    csv_files = _deduplicate_by_date_and_market(csv_files)

    print(f"\n  Found {len(csv_files)} snapshot(s) in {snap_dir}")

    # --- キャッシュ読み込み ---
    cache = {} if args.no_cache else _load_cache()
    cached_dates_by_market: dict[str, set[str]] = {}
    for key, entries in cache.items():
        # key format: "market:date"
        if ":" in key:
            m, d = key.split(":", 1)
            cached_dates_by_market.setdefault(m, set()).add(d)

    # 全スナップショットを読み込み、マーケット別に分割・分類
    from collections import defaultdict
    market_data: dict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)

    # 全CSVのdate一覧を先に収集 (キャッシュ判定に使う)
    csv_date_map: dict[str, Path] = {}
    for csvf in csv_files:
        csv_date_map[extract_date_from_filename(csvf)] = csvf

    for csvf in csv_files:
        date_label = extract_date_from_filename(csvf)
        print(f"  Loading: {csvf.name} ({date_label})")
        df = load_snapshot(csvf)
        if df.empty:
            continue

        # 1ファイル内のマーケットを行単位で分割
        for market, mdf in df.groupby("_market"):
            if not mdf.empty:
                print(f"    → {MARKET_LABELS.get(market, market)}: {len(mdf)} stocks")
                market_data[market].append((date_label, mdf))

    markets = sorted(market_data.keys())
    print(f"  Markets detected: {', '.join(MARKET_LABELS.get(m, m) for m in markets)}")

    # マーケットごとに分析
    for market in markets:
        snapshots = market_data[market]
        label = MARKET_LABELS.get(market, market)

        print(f"\n{'#' * 80}")
        print(f"  MARKET: {label}")
        print(f"{'#' * 80}")

        all_metrics: list[list[SectorMetrics]] = []
        latest_df: Optional[pd.DataFrame] = None
        cached_dates = cached_dates_by_market.get(market, set())
        cache_hits = 0

        for date_label, df in snapshots:
            cache_key = f"{market}:{date_label}"
            if cache_key in cache and not args.no_cache:
                # キャッシュから復元
                metrics = _cache_entry_to_metrics(cache[cache_key])
                cache_hits += 1
                print(f"  [CACHE] {date_label} ({len(df)} stocks)")
            else:
                # 新規計算
                print(f"  Processing: {date_label} ({len(df)} stocks)")
                metrics = compute_sector_metrics(df, date_label)
                # キャッシュに保存
                cache[cache_key] = _metrics_to_cache_entry(metrics)
            all_metrics.append(metrics)
            latest_df = df

        if cache_hits > 0:
            print(f"  Cache: {cache_hits}/{len(snapshots)} snapshots from cache "
                  f"({len(snapshots) - cache_hits} computed)")

        if not all_metrics:
            print("  No data to analyze.")
            continue

        # キャッシュ保存
        if not args.no_cache:
            _save_cache(cache)

        # ローテーション検出
        rotations = detect_rotation(all_metrics, threshold=args.threshold)

        # 売買候補生成
        latest_rotation = rotations[-1] if rotations else None
        candidates = generate_trade_signals(
            latest_df, all_metrics[-1], latest_rotation, top_n=args.top
        )

        # --- 日次差分 + アラート計算 ---
        diffs: list[DiffEntry] = []
        alerts: list[Alert] = []
        prev_metrics = all_metrics[-2] if len(all_metrics) >= 2 else None
        if prev_metrics:
            diffs = compute_daily_diff(all_metrics[-1], prev_metrics)
            alerts = detect_alerts(all_metrics[-1], prev_metrics)
        else:
            alerts = detect_alerts(all_metrics[-1])

        # --- diff モード ---
        if args.diff:
            if prev_metrics:
                print_diff_report(all_metrics[-1], prev_metrics, diffs, alerts,
                                  candidates, market_label=label)
            else:
                print("  [Warning] --diff requires at least 2 snapshots.")
            continue

        # --- 通常レポート出力 ---
        print_sector_dashboard(all_metrics, rotations, candidates,
                               market_label=label, alerts=alerts, diffs=diffs)

        # --- ウォッチリスト ---
        watchlist = None
        if not args.no_watchlist and latest_df is not None:
            date = all_metrics[-1][0].date if all_metrics[-1] else "?"
            watchlist = update_watchlist(candidates, latest_df, date, market)
            print_watchlist(watchlist, market, market_label=label)

        # チャート
        chart_path = None
        if not args.no_chart:
            suffix = f"_{market.lower()}" if len(markets) > 1 else ""
            chart_path = args.out
            if chart_path:
                base, ext = os.path.splitext(chart_path)
                chart_path = f"{base}{suffix}{ext}"
            else:
                chart_path = f"screener/sector_dashboard{suffix}.png"
            plot_dashboard(all_metrics, rotations, candidates, chart_path,
                           title_suffix=f" — {label}")

        # CSV 出力
        if args.export_csv:
            export_analysis_csv(all_metrics, candidates,
                                output_dir="screener", suffix=f"_{market.lower()}")

        # --- HTML出力 ---
        if args.html:
            suffix = f"_{market.lower()}" if len(markets) > 1 else ""
            html_path = f"screener/report{suffix}.html"
            html_content = generate_html_report(
                all_metrics, candidates, alerts, diffs, rotations,
                chart_path=chart_path, market_label=label,
                watchlist=watchlist, market=market,
            )
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"  HTML report saved to: {html_path}")


if __name__ == "__main__":
    main()
