#!/usr/bin/env python3
"""
Sector Rotation Backtester
===========================
screener/snapshots/ に蓄積された日次スナップショット CSV を使い、
SCS (Sector Composite Score) ベースのシグナルをウォークフォワードで
バックテストして統計的に検証する。

Usage:
  python3 screener/backtester.py                         # 全期間バックテスト
  python3 screener/backtester.py --top-long 3            # 上位3セクターをロング
  python3 screener/backtester.py --hold-days 5           # 5日間保有
  python3 screener/backtester.py --export-html           # HTMLレポート生成
  python3 screener/backtester.py --walk-forward 30       # 30日ウィンドウWF検証
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ============================================================
# データ構造
# ============================================================

@dataclass
class SectorScore:
    sector: str
    scs: float
    momentum: float
    breadth_sma200: float
    fund_flow: float
    avg_rsi: float
    avg_adx: float
    stock_count: int


@dataclass
class Trade:
    entry_date: date
    exit_date: Optional[date]
    direction: str          # "LONG" or "SHORT"
    sector: str
    ticker: str
    entry_price: float
    exit_price: Optional[float]
    pnl_pct: float = 0.0
    is_open: bool = True
    signal_strength: float = 0.0


@dataclass
class DailyResult:
    date: date
    long_sectors: List[str]
    short_sectors: List[str]
    portfolio_value: float
    daily_return: float
    n_open_trades: int


@dataclass
class BacktestReport:
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_pct: float
    total_trades: int
    long_trades: int
    short_trades: int
    best_trade_pct: float
    worst_trade_pct: float
    avg_hold_days: float
    benchmark_return_pct: float  # Buy & Hold SPY
    alpha_pct: float
    daily_results: List[DailyResult] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    sector_stats: Dict[str, dict] = field(default_factory=dict)


# ============================================================
# SCS 計算 (sector_flow_analyzer.py と同じロジック)
# ============================================================

SCS_WEIGHTS = {
    "momentum":  0.25,
    "trend":     0.20,
    "breadth":   0.25,
    "fund_flow": 0.20,
    "adx":       0.10,
}


def _normalize_col(df: pd.DataFrame) -> pd.DataFrame:
    """列名を統一する (sector_flow_analyzer と同じマッピング)"""
    rename_map = {
        # Change %
        "change %": "change_pct", "change": "change_pct", "chg %": "change_pct",
        "% change": "change_pct", "chg": "change_pct",
        # Close
        "close": "close", "price": "close", "last": "close", "last close": "close",
        # Volume
        "volume": "volume", "vol": "volume",
        # Relative Volume
        "relative volume": "rel_volume", "rvol": "rel_volume",
        "rel. vol.": "rel_volume", "rel vol": "rel_volume",
        # RSI
        "rsi (14)": "rsi", "rsi": "rsi",
        # MACD
        "macd": "macd", "macd line": "macd",
        "macd signal": "macd_signal", "signal": "macd_signal",
        # ADX
        "adx (14)": "adx", "adx": "adx",
        # SMA
        "sma20": "sma20", "sma (20)": "sma20",
        "sma50": "sma50", "sma (50)": "sma50",
        "sma200": "sma200", "sma (200)": "sma200",
        # Market Cap
        "market cap": "market_cap", "mkt cap": "market_cap",
        # Sector
        "sector": "sector",
        # Ticker
        "ticker": "ticker", "symbol": "ticker",
        # Name
        "name": "name", "description": "name", "company name": "name",
    }
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def compute_sector_scs(df: pd.DataFrame) -> List[SectorScore]:
    """DataFrame → セクターごとの SCS スコアリスト"""
    df = _normalize_col(df)

    required = {"sector", "close", "change_pct"}
    if not required.issubset(df.columns):
        return []

    # 数値変換
    for col in ["close", "change_pct", "volume", "rel_volume", "rsi",
                "macd", "macd_signal", "adx", "sma20", "sma50", "sma200", "market_cap"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # フィルタ: close > 0
    df = df[df["close"].fillna(0) > 0].copy()
    if df.empty:
        return []

    sector_scores = []
    for sector, grp in df.groupby("sector"):
        if str(sector).strip() in ("", "nan", "N/A"):
            continue

        n = len(grp)
        if n < 1:
            continue

        # --- Momentum (25%) ---
        avg_chg = grp["change_pct"].mean() if "change_pct" in grp else 0.0
        momentum_score = max(0, min(100, 50 + avg_chg * 5))

        # --- Trend: RSI + MACD (20%) ---
        rsi_score = 0.0
        if "rsi" in grp:
            avg_rsi = grp["rsi"].mean()
            rsi_score = max(0, min(100, avg_rsi))
        macd_score = 0.0
        if "macd" in grp and "macd_signal" in grp:
            bullish = (grp["macd"] > grp["macd_signal"]).mean()
            macd_score = bullish * 100
        trend_score = rsi_score * 0.6 + macd_score * 0.4

        # --- Breadth: % above SMA200 (25%) ---
        breadth_score = 50.0
        if "sma200" in grp and "close" in grp:
            above = (grp["close"] > grp["sma200"]).mean()
            breadth_score = above * 100

        # --- Fund Flow: Volume × Change% (20%) ---
        fund_flow_raw = 0.0
        if "volume" in grp and "change_pct" in grp:
            mc = grp.get("market_cap", pd.Series([1e9] * n))
            fund_flow_raw = (grp["volume"] * grp["change_pct"] / 100).sum()
        # Normalize to 0-100
        ff_score = max(0, min(100, 50 + np.tanh(fund_flow_raw / 1e8) * 50))

        # --- ADX: trend strength (10%) ---
        adx_score = 50.0
        if "adx" in grp:
            avg_adx = grp["adx"].mean()
            adx_score = max(0, min(100, avg_adx))

        # --- SCS composite ---
        scs = (
            momentum_score  * SCS_WEIGHTS["momentum"] +
            trend_score     * SCS_WEIGHTS["trend"] +
            breadth_score   * SCS_WEIGHTS["breadth"] +
            ff_score        * SCS_WEIGHTS["fund_flow"] +
            adx_score       * SCS_WEIGHTS["adx"]
        )

        sector_scores.append(SectorScore(
            sector=str(sector),
            scs=round(scs, 2),
            momentum=round(momentum_score, 2),
            breadth_sma200=round(breadth_score, 2),
            fund_flow=round(fund_flow_raw / 1e6, 2),  # M単位
            avg_rsi=round(rsi_score, 2),
            avg_adx=round(adx_score, 2),
            stock_count=n,
        ))

    return sorted(sector_scores, key=lambda x: x.scs, reverse=True)


# ============================================================
# スナップショットローダー
# ============================================================

def load_snapshots(snapshot_dir: str, market: str = "US") -> Dict[date, pd.DataFrame]:
    """
    snapshots/ ディレクトリから日付別に CSV を読み込む。
    market: "US" or "JP" — ticker パターンで自動分類
    """
    snapshots = {}
    csv_files = [
        f for f in os.listdir(snapshot_dir)
        if f.endswith(".csv") and not f.startswith("sector_") and not f.startswith("trade_")
    ]

    for fname in sorted(csv_files):
        # 日付を抽出 (YYYY-MM-DD)
        snap_date = None
        for part in fname.replace("_", "-").split("-"):
            pass  # iterate to find date
        import re
        m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
        if not m:
            continue
        try:
            snap_date = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            continue

        fpath = os.path.join(snapshot_dir, fname)
        try:
            df = pd.read_csv(fpath, low_memory=False)
        except Exception:
            continue

        # JP/US フィルタ
        if "Ticker" in df.columns or "ticker" in df.columns:
            ticker_col = "Ticker" if "Ticker" in df.columns else "ticker"
            tickers = df[ticker_col].astype(str)
            is_jp = tickers.str.match(r"^\d{4}(\.T)?$").mean() > 0.3
            if market == "JP" and not is_jp:
                continue
            if market == "US" and is_jp:
                continue

        # 同じ日付が複数ある場合は行数が多い方を採用
        if snap_date in snapshots:
            if len(df) > len(snapshots[snap_date]):
                snapshots[snap_date] = df
        else:
            snapshots[snap_date] = df

    return dict(sorted(snapshots.items()))


# ============================================================
# バックテストエンジン
# ============================================================

class SectorRotationBacktester:
    def __init__(
        self,
        snapshot_dir: str,
        market: str = "US",
        top_long: int = 3,
        top_short: int = 3,
        hold_days: int = 5,
        min_scs_diff: float = 5.0,  # LONG/SHORT 判断の最小SCS差
        initial_capital: float = 1_000_000,
    ):
        self.snapshot_dir = snapshot_dir
        self.market = market
        self.top_long = top_long
        self.top_short = top_short
        self.hold_days = hold_days
        self.min_scs_diff = min_scs_diff
        self.initial_capital = initial_capital

    def run(self) -> BacktestReport:
        print(f"\nLoading snapshots from {self.snapshot_dir}...")
        snapshots = load_snapshots(self.snapshot_dir, self.market)

        if len(snapshots) < 3:
            print(f"  [WARN] Not enough snapshots for backtesting ({len(snapshots)} found, need >= 3)")
            return self._empty_report()

        dates = sorted(snapshots.keys())
        print(f"  Found {len(dates)} snapshots: {dates[0]} → {dates[-1]}")

        # ウォークフォワード: 各日のシグナル生成 → 翌日のリターン計算
        trades: List[Trade] = []
        daily_results: List[DailyResult] = []
        portfolio_value = self.initial_capital
        equity_curve = [self.initial_capital]

        # 日次シグナル記録
        for i, snap_date in enumerate(dates[:-1]):  # 最終日は翌日リターンなし
            df = snapshots[snap_date]
            next_date = dates[i + 1]
            next_df = snapshots[next_date]

            scores = compute_sector_scs(df)
            if len(scores) < 2:
                continue

            # LONG: 上位 top_long セクター
            # SHORT: 下位 top_short セクター (SCS が低い)
            long_sectors = [s.sector for s in scores[:self.top_long]]
            short_sectors = [s.sector for s in scores[-self.top_short:]]

            # SCS差が小さい場合はシグナルなし
            if scores[0].scs - scores[-1].scs < self.min_scs_diff:
                long_sectors = []
                short_sectors = []

            # 各セクターのトップ/ボトム銘柄でトレード
            day_trades = self._generate_trades(
                df, next_df,
                long_sectors, short_sectors,
                snap_date, next_date,
            )
            trades.extend(day_trades)

            # 日次リターン (トレード群の平均)
            day_returns = [t.pnl_pct for t in day_trades if not t.is_open]
            avg_day_return = np.mean(day_returns) if day_returns else 0.0
            portfolio_value *= (1 + avg_day_return / 100)
            equity_curve.append(portfolio_value)

            daily_results.append(DailyResult(
                date=snap_date,
                long_sectors=long_sectors,
                short_sectors=short_sectors,
                portfolio_value=portfolio_value,
                daily_return=avg_day_return,
                n_open_trades=len(day_trades),
            ))

        # ベンチマーク (SPY 相当の単純バイ&ホールド)
        benchmark_return = self._calc_benchmark_return(snapshots, dates)

        report = self._calc_report(
            trades, daily_results, equity_curve,
            benchmark_return, dates
        )
        return report

    def _generate_trades(
        self,
        df_entry: pd.DataFrame,
        df_exit: pd.DataFrame,
        long_sectors: List[str],
        short_sectors: List[str],
        entry_date: date,
        exit_date: date,
    ) -> List[Trade]:
        """シグナルセクターの代表銘柄でトレードを生成"""
        trades = []
        df_e = _normalize_col(df_entry)
        df_x = _normalize_col(df_exit)

        if "ticker" not in df_e.columns or "close" not in df_e.columns:
            return trades

        df_e["close"] = pd.to_numeric(df_e["close"], errors="coerce")
        df_x["close"] = pd.to_numeric(df_x["close"], errors="coerce")

        # exit DataFrame を ticker でインデックス化
        if "ticker" in df_x.columns:
            exit_prices = df_x.set_index("ticker")["close"].to_dict()
        else:
            exit_prices = {}

        def get_candidates(sectors: List[str], direction: str) -> List[dict]:
            if "sector" not in df_e.columns:
                return []
            mask = df_e["sector"].isin(sectors)
            sub = df_e[mask].copy()
            if sub.empty:
                return []
            # 強さスコア (RSI + MACD ブル/ベアで並び替え)
            sub = sub[sub["close"] > 0]
            if direction == "LONG":
                if "rsi" in sub.columns:
                    sub["rsi"] = pd.to_numeric(sub["rsi"], errors="coerce")
                    # RSI 40-70 のみ (過熱排除)
                    sub = sub[(sub["rsi"] >= 40) & (sub["rsi"] <= 70)]
                if "sma50" in sub.columns:
                    sub["sma50"] = pd.to_numeric(sub["sma50"], errors="coerce")
                    sub = sub[sub["close"] > sub["sma50"]]
            else:  # SHORT
                if "rsi" in sub.columns:
                    sub["rsi"] = pd.to_numeric(sub["rsi"], errors="coerce")
                    sub = sub[sub["rsi"] < 60]
                if "sma50" in sub.columns:
                    sub["sma50"] = pd.to_numeric(sub["sma50"], errors="coerce")
                    sub = sub[sub["close"] < sub["sma50"]]

            # 各セクターからトップ1銘柄
            result = []
            for sector in sectors:
                s_sub = sub[sub["sector"] == sector] if "sector" in sub else sub
                if s_sub.empty:
                    continue
                # RVOL 順
                if "rel_volume" in s_sub.columns:
                    s_sub["rel_volume"] = pd.to_numeric(s_sub["rel_volume"], errors="coerce")
                    top = s_sub.nlargest(1, "rel_volume")
                else:
                    top = s_sub.head(1)
                for _, row in top.iterrows():
                    result.append({
                        "ticker": str(row.get("ticker", "")),
                        "sector": sector,
                        "entry_price": float(row["close"]),
                    })
            return result

        # LONGトレード
        for c in get_candidates(long_sectors, "LONG"):
            ep = c["entry_price"]
            xp = exit_prices.get(c["ticker"], ep)
            if xp and xp > 0 and ep > 0:
                pnl = (xp / ep - 1) * 100
            else:
                pnl = 0.0
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                direction="LONG",
                sector=c["sector"],
                ticker=c["ticker"],
                entry_price=ep,
                exit_price=xp if xp else ep,
                pnl_pct=round(pnl, 4),
                is_open=False,
            ))

        # SHORTトレード
        for c in get_candidates(short_sectors, "SHORT"):
            ep = c["entry_price"]
            xp = exit_prices.get(c["ticker"], ep)
            if xp and xp > 0 and ep > 0:
                pnl = (ep / xp - 1) * 100  # SHORT: 下落で利益
            else:
                pnl = 0.0
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=exit_date,
                direction="SHORT",
                sector=c["sector"],
                ticker=c["ticker"],
                entry_price=ep,
                exit_price=xp if xp else ep,
                pnl_pct=round(pnl, 4),
                is_open=False,
            ))

        return trades

    def _calc_benchmark_return(
        self,
        snapshots: Dict[date, pd.DataFrame],
        dates: List[date],
    ) -> float:
        """SPY or 市場全体の Buy&Hold リターンを推定"""
        if len(dates) < 2:
            return 0.0
        first_df = _normalize_col(snapshots[dates[0]])
        last_df  = _normalize_col(snapshots[dates[-1]])

        if "close" not in first_df.columns:
            return 0.0

        # SPY or 市場平均
        first_df["close"] = pd.to_numeric(first_df["close"], errors="coerce")
        last_df["close"]  = pd.to_numeric(last_df["close"], errors="coerce")

        avg_first = first_df["close"].median()
        avg_last  = last_df["close"].median()

        if avg_first and avg_first > 0:
            return round((avg_last / avg_first - 1) * 100, 2)
        return 0.0

    def _calc_report(
        self,
        trades: List[Trade],
        daily_results: List[DailyResult],
        equity_curve: List[float],
        benchmark_return: float,
        dates: List[date],
    ) -> BacktestReport:
        """全パフォーマンス指標を計算"""

        closed = [t for t in trades if not t.is_open]
        if not closed:
            return self._empty_report()

        pnls = [t.pnl_pct for t in closed]
        long_pnls  = [t.pnl_pct for t in closed if t.direction == "LONG"]
        short_pnls = [t.pnl_pct for t in closed if t.direction == "SHORT"]

        wins  = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100 if pnls else 0.0

        gross_profit = sum(wins)
        gross_loss   = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # エクイティカーブ
        arr = np.array(equity_curve)
        daily_returns = np.diff(arr) / arr[:-1]

        total_return = (arr[-1] / arr[0] - 1) * 100 if arr[0] > 0 else 0.0

        # 年率リターン
        n_days = (dates[-1] - dates[0]).days if len(dates) >= 2 else 1
        n_years = max(n_days / 365.25, 1 / 365.25)
        ann_return = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100

        # シャープレシオ (日次リターン, リスクフリー 0%)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0

        # ソルティノレシオ
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 1 and downside_returns.std() > 0:
            sortino = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino = 0.0

        # 最大ドローダウン
        running_max = np.maximum.accumulate(arr)
        drawdowns = (arr - running_max) / running_max * 100
        max_dd = float(drawdowns.min())

        # カルマーレシオ
        calmar = ann_return / abs(max_dd) if max_dd < 0 else float("inf")

        # セクター別統計
        sector_stats = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0})
        for t in closed:
            sector_stats[t.sector]["trades"] += 1
            sector_stats[t.sector]["total_pnl"] += t.pnl_pct
            if t.pnl_pct > 0:
                sector_stats[t.sector]["wins"] += 1

        # 保有日数
        hold_days_list = [
            (t.exit_date - t.entry_date).days
            for t in closed
            if t.exit_date
        ]
        avg_hold = np.mean(hold_days_list) if hold_days_list else 0.0

        return BacktestReport(
            total_return_pct=round(total_return, 2),
            annualized_return_pct=round(ann_return, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown_pct=round(max_dd, 2),
            calmar_ratio=round(calmar, 3) if not math.isinf(calmar) else 999.0,
            win_rate=round(win_rate, 1),
            profit_factor=round(profit_factor, 3) if not math.isinf(profit_factor) else 999.0,
            avg_trade_pct=round(np.mean(pnls), 4) if pnls else 0.0,
            total_trades=len(closed),
            long_trades=len(long_pnls),
            short_trades=len(short_pnls),
            best_trade_pct=round(max(pnls), 4) if pnls else 0.0,
            worst_trade_pct=round(min(pnls), 4) if pnls else 0.0,
            avg_hold_days=round(avg_hold, 1),
            benchmark_return_pct=benchmark_return,
            alpha_pct=round(total_return - benchmark_return, 2),
            daily_results=daily_results,
            trades=closed,
            sector_stats=dict(sector_stats),
        )

    def _empty_report(self) -> BacktestReport:
        return BacktestReport(
            total_return_pct=0, annualized_return_pct=0,
            sharpe_ratio=0, sortino_ratio=0,
            max_drawdown_pct=0, calmar_ratio=0,
            win_rate=0, profit_factor=0,
            avg_trade_pct=0, total_trades=0,
            long_trades=0, short_trades=0,
            best_trade_pct=0, worst_trade_pct=0,
            avg_hold_days=0, benchmark_return_pct=0, alpha_pct=0,
        )


# ============================================================
# ウォークフォワード分析
# ============================================================

class WalkForwardAnalyzer:
    """
    データを train/test に分割してウォークフォワード検証を行う。
    - train: パラメータ最適化 (top_long, hold_days を探索)
    - test : out-of-sample 評価
    """

    def __init__(
        self,
        snapshot_dir: str,
        market: str = "US",
        train_window: int = 20,  # 訓練に使うスナップショット数
        test_window: int = 10,   # テストに使うスナップショット数
    ):
        self.snapshot_dir = snapshot_dir
        self.market = market
        self.train_window = train_window
        self.test_window = test_window

    def run(self) -> List[dict]:
        snapshots = load_snapshots(self.snapshot_dir, self.market)
        dates = sorted(snapshots.keys())

        if len(dates) < self.train_window + self.test_window:
            print(f"  [WARN] Not enough data for walk-forward ({len(dates)} snapshots)")
            return []

        results = []
        step = max(1, self.test_window // 2)

        for start in range(0, len(dates) - self.train_window - self.test_window + 1, step):
            train_dates = dates[start: start + self.train_window]
            test_dates  = dates[start + self.train_window: start + self.train_window + self.test_window]

            train_snaps = {d: snapshots[d] for d in train_dates}
            test_snaps  = {d: snapshots[d] for d in test_dates}

            # 訓練: top_long=1,2,3 × hold=1,3,5 を総当たり
            best_sharpe = -999
            best_params = {"top_long": 2, "hold_days": 3}
            for top_l in [1, 2, 3]:
                for hold in [1, 3, 5]:
                    bt = SectorRotationBacktester(
                        snapshot_dir=self.snapshot_dir,
                        market=self.market,
                        top_long=top_l,
                        hold_days=hold,
                    )
                    # ミニバックテスト (train only)
                    bt_result = bt._run_on_snapshots(train_snaps)
                    if bt_result.sharpe_ratio > best_sharpe:
                        best_sharpe = bt_result.sharpe_ratio
                        best_params = {"top_long": top_l, "hold_days": hold}

            # テスト: 最良パラメータで out-of-sample 評価
            bt_test = SectorRotationBacktester(
                snapshot_dir=self.snapshot_dir,
                market=self.market,
                **best_params,
            )
            test_result = bt_test._run_on_snapshots(test_snaps)

            results.append({
                "period":        f"{test_dates[0]} → {test_dates[-1]}",
                "best_params":   best_params,
                "train_sharpe":  best_sharpe,
                "test_return":   test_result.total_return_pct,
                "test_sharpe":   test_result.sharpe_ratio,
                "test_win_rate": test_result.win_rate,
                "test_trades":   test_result.total_trades,
            })

        return results


# ウォークフォワード用にスナップショットを外から注入できるようにオーバーロード
def _backtester_run_on_snapshots(self, snapshots: dict) -> BacktestReport:
    if len(snapshots) < 2:
        return self._empty_report()
    dates = sorted(snapshots.keys())
    trades, daily_results, equity_curve = [], [], [self.initial_capital]
    portfolio_value = self.initial_capital

    for i, snap_date in enumerate(dates[:-1]):
        df = snapshots[snap_date]
        next_date = dates[i + 1]
        next_df = snapshots[next_date]
        scores = compute_sector_scs(df)
        if len(scores) < 2:
            continue
        long_sectors  = [s.sector for s in scores[:self.top_long]]
        short_sectors = [s.sector for s in scores[-self.top_short:]]
        day_trades = self._generate_trades(df, next_df, long_sectors, short_sectors, snap_date, next_date)
        trades.extend(day_trades)
        pnls = [t.pnl_pct for t in day_trades if not t.is_open]
        avg_r = np.mean(pnls) if pnls else 0.0
        portfolio_value *= (1 + avg_r / 100)
        equity_curve.append(portfolio_value)
        daily_results.append(DailyResult(snap_date, long_sectors, short_sectors, portfolio_value, avg_r, len(day_trades)))

    benchmark = self._calc_benchmark_return(snapshots, dates)
    return self._calc_report(trades, daily_results, equity_curve, benchmark, dates)


SectorRotationBacktester._run_on_snapshots = _backtester_run_on_snapshots


# ============================================================
# レポート出力
# ============================================================

def print_report(report: BacktestReport, walk_forward: Optional[List[dict]] = None):
    """ターミナルに整形されたバックテストレポートを出力"""
    print("\n" + "=" * 60)
    print(" BACKTEST REPORT — Sector Rotation Strategy")
    print("=" * 60)

    print(f"\n{'PERFORMANCE':}")
    print(f"  Total Return       : {report.total_return_pct:+.2f}%")
    print(f"  Annualized Return  : {report.annualized_return_pct:+.2f}%")
    print(f"  Benchmark Return   : {report.benchmark_return_pct:+.2f}%")
    print(f"  Alpha              : {report.alpha_pct:+.2f}%")

    print(f"\n{'RISK':}")
    print(f"  Sharpe Ratio       : {report.sharpe_ratio:.3f}  (>1.0 = good)")
    print(f"  Sortino Ratio      : {report.sortino_ratio:.3f}")
    print(f"  Max Drawdown       : {report.max_drawdown_pct:.2f}%")
    print(f"  Calmar Ratio       : {report.calmar_ratio:.3f}")

    print(f"\n{'TRADE STATS':}")
    print(f"  Total Trades       : {report.total_trades}  (L:{report.long_trades} / S:{report.short_trades})")
    print(f"  Win Rate           : {report.win_rate:.1f}%  (>50% = good)")
    print(f"  Profit Factor      : {report.profit_factor:.3f}  (>1.5 = good)")
    print(f"  Avg Trade          : {report.avg_trade_pct:+.4f}%")
    print(f"  Best Trade         : {report.best_trade_pct:+.4f}%")
    print(f"  Worst Trade        : {report.worst_trade_pct:+.4f}%")
    print(f"  Avg Hold Days      : {report.avg_hold_days:.1f}")

    if report.sector_stats:
        print(f"\n{'SECTOR BREAKDOWN':}")
        sorted_sectors = sorted(
            report.sector_stats.items(),
            key=lambda x: x[1]["total_pnl"],
            reverse=True
        )
        for sector, stats in sorted_sectors[:10]:
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] else 0
            print(f"  {sector:30s} trades={stats['trades']:3d}  pnl={stats['total_pnl']:+7.2f}%  wr={wr:.0f}%")

    if walk_forward:
        print(f"\n{'WALK-FORWARD SUMMARY':}")
        print(f"  {'Period':<30s} {'Params':>20s}  {'Test Ret':>8s}  {'Test SR':>7s}  {'Win%':>5s}")
        for wf in walk_forward:
            params = f"top={wf['best_params']['top_long']},hold={wf['best_params']['hold_days']}"
            print(f"  {wf['period']:<30s} {params:>20s}  {wf['test_return']:>+7.2f}%  {wf['test_sharpe']:>7.3f}  {wf['test_win_rate']:>4.1f}%")

    print("\n" + "=" * 60)

    # 評価コメント
    comments = []
    if report.sharpe_ratio >= 1.5:
        comments.append("✓ シャープレシオ良好 (≥1.5)")
    elif report.sharpe_ratio >= 1.0:
        comments.append("△ シャープレシオ合格 (≥1.0)")
    else:
        comments.append("✗ シャープレシオ不足 (<1.0) — パラメータ改善が必要")

    if report.win_rate >= 55:
        comments.append("✓ 勝率良好 (≥55%)")
    elif report.win_rate >= 50:
        comments.append("△ 勝率ギリギリ (≥50%)")
    else:
        comments.append("✗ 勝率不足 (<50%) — フィルタ強化を検討")

    if report.alpha_pct > 0:
        comments.append(f"✓ ベンチマーク超過 (+{report.alpha_pct:.2f}%)")
    else:
        comments.append(f"✗ ベンチマーク未達 ({report.alpha_pct:.2f}%)")

    if report.max_drawdown_pct < -20:
        comments.append(f"✗ 最大ドローダウン大 ({report.max_drawdown_pct:.1f}%) — リスク管理が必要")

    print("\n VERDICT:")
    for c in comments:
        print(f"  {c}")
    print()


def export_html_report(report: BacktestReport, output_path: str):
    """HTMLバックテストレポートを生成"""
    trades_html = ""
    for t in sorted(report.trades, key=lambda x: x.entry_date, reverse=True)[:100]:
        color = "#2ecc71" if t.pnl_pct > 0 else "#e74c3c"
        trades_html += f"""
        <tr>
          <td>{t.entry_date}</td>
          <td>{t.exit_date}</td>
          <td style="color:{'#2ecc71' if t.direction=='LONG' else '#e74c3c'}">{t.direction}</td>
          <td>{t.ticker}</td>
          <td>{t.sector[:20]}</td>
          <td>{t.entry_price:.2f}</td>
          <td>{t.exit_price:.2f}</td>
          <td style="color:{color}">{t.pnl_pct:+.2f}%</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>Backtest Report — Sector Rotation</title>
  <style>
    body {{ background:#0d1117; color:#c9d1d9; font-family:monospace; padding:20px; }}
    h1,h2 {{ color:#58a6ff; }} table {{ border-collapse:collapse; width:100%; }}
    th {{ background:#161b22; color:#58a6ff; padding:8px; text-align:left; }}
    td {{ padding:6px 8px; border-bottom:1px solid #21262d; }}
    .good {{ color:#2ecc71; }} .bad {{ color:#e74c3c; }} .warn {{ color:#f39c12; }}
    .metric {{ display:inline-block; background:#161b22; padding:12px 20px;
               margin:5px; border-radius:6px; min-width:160px; }}
    .metric .val {{ font-size:1.4em; font-weight:bold; }}
  </style>
</head>
<body>
  <h1>Backtest Report — Sector Rotation Strategy</h1>
  <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

  <h2>Performance Overview</h2>
  <div>
    <div class="metric">
      <div>Total Return</div>
      <div class="val {'good' if report.total_return_pct > 0 else 'bad'}">{report.total_return_pct:+.2f}%</div>
    </div>
    <div class="metric">
      <div>Annualized</div>
      <div class="val {'good' if report.annualized_return_pct > 0 else 'bad'}">{report.annualized_return_pct:+.2f}%</div>
    </div>
    <div class="metric">
      <div>Alpha vs Benchmark</div>
      <div class="val {'good' if report.alpha_pct > 0 else 'bad'}">{report.alpha_pct:+.2f}%</div>
    </div>
    <div class="metric">
      <div>Sharpe Ratio</div>
      <div class="val {'good' if report.sharpe_ratio >= 1.0 else 'bad'}">{report.sharpe_ratio:.3f}</div>
    </div>
    <div class="metric">
      <div>Max Drawdown</div>
      <div class="val {'warn' if report.max_drawdown_pct > -20 else 'bad'}">{report.max_drawdown_pct:.2f}%</div>
    </div>
    <div class="metric">
      <div>Win Rate</div>
      <div class="val {'good' if report.win_rate >= 50 else 'bad'}">{report.win_rate:.1f}%</div>
    </div>
    <div class="metric">
      <div>Profit Factor</div>
      <div class="val {'good' if report.profit_factor >= 1.5 else 'warn'}">{report.profit_factor:.3f}</div>
    </div>
    <div class="metric">
      <div>Total Trades</div>
      <div class="val">{report.total_trades}</div>
    </div>
  </div>

  <h2>Recent Trades (last 100)</h2>
  <table>
    <tr><th>Entry</th><th>Exit</th><th>Dir</th><th>Ticker</th><th>Sector</th>
        <th>Entry $</th><th>Exit $</th><th>PnL %</th></tr>
    {trades_html}
  </table>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  [SAVED] {output_path}")


# ============================================================
# メイン
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sector Rotation Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--snapshot-dir",
                        default=os.path.join(os.path.dirname(__file__), "snapshots"),
                        help="Path to snapshots directory")
    parser.add_argument("--market", default="US", choices=["US", "JP"],
                        help="Market to backtest (default: US)")
    parser.add_argument("--top-long",  type=int, default=3,
                        help="Number of top sectors to go LONG (default: 3)")
    parser.add_argument("--top-short", type=int, default=3,
                        help="Number of bottom sectors to SHORT (default: 3)")
    parser.add_argument("--hold-days", type=int, default=1,
                        help="Number of days to hold each trade (default: 1 = next snapshot)")
    parser.add_argument("--min-scs-diff", type=float, default=5.0,
                        help="Minimum SCS spread to generate signals (default: 5.0)")
    parser.add_argument("--walk-forward", type=int, default=0, metavar="TRAIN_WINDOW",
                        help="Run walk-forward validation with N-snapshot training window")
    parser.add_argument("--export-html", action="store_true",
                        help="Export HTML report")
    parser.add_argument("--export-csv",  action="store_true",
                        help="Export trades to CSV")

    args = parser.parse_args()

    print(f"Sector Rotation Backtester")
    print(f"  Market       : {args.market}")
    print(f"  Top LONG     : {args.top_long}")
    print(f"  Top SHORT    : {args.top_short}")
    print(f"  Hold Days    : {args.hold_days}")
    print(f"  Min SCS Diff : {args.min_scs_diff}")

    # メインバックテスト
    bt = SectorRotationBacktester(
        snapshot_dir=args.snapshot_dir,
        market=args.market,
        top_long=args.top_long,
        top_short=args.top_short,
        hold_days=args.hold_days,
        min_scs_diff=args.min_scs_diff,
    )
    report = bt.run()

    # ウォークフォワード
    wf_results = None
    if args.walk_forward > 0:
        print(f"\nRunning walk-forward validation (train={args.walk_forward})...")
        wf = WalkForwardAnalyzer(
            snapshot_dir=args.snapshot_dir,
            market=args.market,
            train_window=args.walk_forward,
            test_window=max(5, args.walk_forward // 2),
        )
        wf_results = wf.run()

    print_report(report, wf_results)

    # エクスポート
    out_dir = os.path.dirname(args.snapshot_dir)

    if args.export_html:
        html_path = os.path.join(out_dir, f"backtest_report_{args.market}.html")
        export_html_report(report, html_path)

    if args.export_csv and report.trades:
        csv_path = os.path.join(out_dir, f"backtest_trades_{args.market}.csv")
        trades_df = pd.DataFrame([{
            "entry_date":   t.entry_date,
            "exit_date":    t.exit_date,
            "direction":    t.direction,
            "ticker":       t.ticker,
            "sector":       t.sector,
            "entry_price":  t.entry_price,
            "exit_price":   t.exit_price,
            "pnl_pct":      t.pnl_pct,
        } for t in report.trades])
        trades_df.to_csv(csv_path, index=False)
        print(f"  [SAVED] {csv_path}")

    if wf_results:
        wf_csv = os.path.join(out_dir, f"walkforward_{args.market}.csv")
        pd.DataFrame(wf_results).to_csv(wf_csv, index=False)
        print(f"  [SAVED] {wf_csv}")


if __name__ == "__main__":
    main()
