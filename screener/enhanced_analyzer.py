"""
Enhanced Sector Rotation & Fund Flow Analyzer
=============================================
yfinance + EDINET API + e-STAT API を統合したトップダウン分析ツール。

コンセプト:
    個別銘柄ではなくセクター全体の健康状態を先に評価し、
    強いセクターからロング候補、弱いセクターからショート候補を絞り込む
    「トップダウン・アプローチ」。

    市場全体 → セクター強弱 → 資金フロー方向 → ローテーション検出 → 個別銘柄選定

5つの分析レイヤー:
    1. Sector Composite Score (SCS)
       複数指標を加重合成した 0-100 のセクター強弱スコア
       [Momentum 25% + Trend 20% + Breadth 25% + Fund Flow 20% + ADX 10%]
       + マクロ調整 (e-STAT) + イベントブースト (EDINET)

    2. Estimated Fund Flow (EFF)
       Fund Flow = Σ (Volume × Change% / 100)
       出来高を伴う価格変動だけが「本物の資金移動」

    3. Market Breadth
       セクター内で SMA200 / SMA50 / SMA20 を上回る銘柄の比率

    4. Rotation Detection
       2つ以上のスナップショットを比較し SCS 変化量が閾値を超えたセクターを検出

    5. Trade Signal Generation
       LONG: Close > SMA50, RSI 40-70, RVOL > 1, MACD > Signal, ADX > 20
       SHORT: Close < SMA50, RSI 30-60, MACD < Signal
       EDINET イベント銘柄にはボーナス加点

データソース:
    --source csv      TradingView CSV (既存の sector_flow_analyzer.py と同様)
    --source yfinance yfinance からリアルタイム取得
    --source both     CSV + yfinance を統合

Usage:
    # yfinance + EDINET + e-STAT で日本株を分析
    python3 screener/enhanced_analyzer.py --market JP --source yfinance

    # TradingView CSV で分析 (既存と同様)
    python3 screener/enhanced_analyzer.py --market JP --source csv screener/snapshots/

    # APIキーあり: EDINET・e-STAT も統合
    export EDINET_API_KEY=your_key
    export ESTAT_APP_ID=your_id
    python3 screener/enhanced_analyzer.py --market JP --source yfinance --top 10
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# 既存のアナライザーをインポート (テクニカル計算・可視化を再利用)
sys.path.insert(0, str(Path(__file__).parent))
from sector_flow_analyzer import (
    SectorMetrics,
    RotationSignal,
    TradeCandidate,
    SCS_WEIGHTS,
    compute_sector_metrics,
    detect_rotation,
    generate_trade_signals,
    print_sector_dashboard,
    plot_dashboard,
    export_analysis_csv,
    load_snapshot,
    extract_date_from_filename,
    _deduplicate_by_date_and_market,
    MARKET_LABELS,
    _clip_score,
    _breadth,
)

# 新規データフェッチャー
from data_fetchers.yfinance_fetcher import YFinanceFetcher, JP_SECTOR_TICKERS, US_SECTOR_TICKERS
from data_fetchers.edinet_fetcher import EdinetFetcher, EdinetEvent
from data_fetchers.estat_fetcher import EStatFetcher, MacroIndicator, get_simulated_indicators


# ---------------------------------------------------------------------------
# Enhanced SCS Computation
# ---------------------------------------------------------------------------

def compute_enhanced_scs(
    sector_metrics: list[SectorMetrics],
    macro_scores: dict[str, float],
    event_boosts: dict[str, float],
) -> list[SectorMetrics]:
    """
    既存の SectorMetrics に マクロスコア (e-STAT) と
    イベントブースト (EDINET) を適用して SCS を調整する。

    Parameters
    ----------
    sector_metrics : list[SectorMetrics]
        compute_sector_metrics() の出力
    macro_scores : dict[str, float]
        sector → マクロ調整スコア (-10 ~ +10)
    event_boosts : dict[str, float]
        sector → イベントブーストスコア (0 ~ +9)

    Returns
    -------
    list[SectorMetrics]
        composite_score が更新された SectorMetrics リスト
    """
    updated = []
    for m in sector_metrics:
        base_score = m.composite_score
        macro_adj = macro_scores.get(m.sector, 0.0)   # -10 ~ +10
        event_adj = event_boosts.get(m.sector, 0.0)   # 0 ~ +9

        # マクロ・イベント調整 (各最大±10%)
        adjusted = base_score + macro_adj + event_adj
        adjusted = max(0.0, min(100.0, adjusted))

        updated.append(SectorMetrics(
            date=m.date,
            sector=m.sector,
            avg_change_pct=m.avg_change_pct,
            avg_rsi=m.avg_rsi,
            avg_adx=m.avg_adx,
            avg_macd_diff=m.avg_macd_diff,
            breadth_sma20=m.breadth_sma20,
            breadth_sma50=m.breadth_sma50,
            breadth_sma200=m.breadth_sma200,
            total_fund_flow=m.total_fund_flow,
            avg_relative_volume=m.avg_relative_volume,
            total_market_cap=m.total_market_cap,
            composite_score=round(adjusted, 2),
            stock_count=m.stock_count,
        ))

    return updated


# ---------------------------------------------------------------------------
# Enhanced Trade Signal Generator (with EDINET event bonus)
# ---------------------------------------------------------------------------

def generate_enhanced_signals(
    df: pd.DataFrame,
    sector_metrics: list[SectorMetrics],
    rotation: Optional[RotationSignal],
    ticker_events: dict[str, list[EdinetEvent]],
    top_n: int = 10,
) -> list[TradeCandidate]:
    """
    EDINET イベント情報を使って売買シグナルをブーストする。

    既存の generate_trade_signals() を拡張:
    - TOB 対象銘柄 → ストレングス +30
    - 自社株買い銘柄 → ストレングス +20
    - 大量保有報告銘柄 → ストレングス +10

    Parameters
    ----------
    df : pd.DataFrame
        銘柄データ
    sector_metrics : list[SectorMetrics]
        セクターメトリクス
    rotation : RotationSignal or None
    ticker_events : dict[str, list[EdinetEvent]]
        ticker → EDINET イベントのマッピング
    top_n : int
        上位 N 件

    Returns
    -------
    list[TradeCandidate]
    """
    # まず既存のロジックでシグナル生成
    base_candidates = generate_trade_signals(df, sector_metrics, rotation, top_n=top_n * 3)

    # EDINET ボーナスを適用
    enhanced = []
    for cand in base_candidates:
        events = ticker_events.get(cand.ticker, [])
        extra_strength = 0.0
        extra_reasons = []

        for event in events:
            if event.category == "TOB":
                extra_strength += 30.0
                extra_reasons.append(f"公開買付対象 [{event.submit_date}] {event.filer_name}")
            elif event.category == "BUYBACK":
                extra_strength += 20.0
                extra_reasons.append(f"自社株買い [{event.submit_date}]")
            elif event.category == "LARGE_HOLDING":
                extra_strength += 10.0
                extra_reasons.append(f"大量保有報告 [{event.submit_date}] {event.filer_name}")
            elif event.category == "LARGE_HOLDING_CHANGE":
                extra_strength += 5.0
                extra_reasons.append(f"大量保有変更 [{event.submit_date}]")

        new_strength = min(100.0, cand.strength + extra_strength)
        new_reasons = cand.reasons + extra_reasons

        enhanced.append(TradeCandidate(
            ticker=cand.ticker,
            name=cand.name,
            sector=cand.sector,
            industry=cand.industry,
            signal=cand.signal,
            strength=new_strength,
            reasons=new_reasons,
            close=cand.close,
            change_pct=cand.change_pct,
            rsi=cand.rsi,
            volume_ratio=cand.volume_ratio,
        ))

    # LONG / SHORT を分離してソート
    longs = sorted([c for c in enhanced if c.signal == "LONG"],
                   key=lambda c: c.strength, reverse=True)[:top_n]
    shorts = sorted([c for c in enhanced if c.signal == "SHORT"],
                    key=lambda c: c.strength, reverse=True)[:top_n]
    return longs + shorts


# ---------------------------------------------------------------------------
# Data Source Handlers
# ---------------------------------------------------------------------------

def load_from_csv(
    snapshots_dir: str,
    market: Optional[str] = None,
) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    """TradingView CSV からデータを読み込む (既存ロジック)"""
    from collections import defaultdict

    snap_dir = Path(snapshots_dir)
    csv_files = sorted(snap_dir.glob("*.csv"),
                       key=lambda f: extract_date_from_filename(f))
    if not csv_files:
        print(f"Error: No CSV files found in {snap_dir}")
        return {}

    csv_files = _deduplicate_by_date_and_market(csv_files)
    print(f"\n  [CSV] {len(csv_files)} スナップショットを読み込み中...")

    market_data: dict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)

    for csvf in csv_files:
        date_label = extract_date_from_filename(csvf)
        df = load_snapshot(csvf)
        if df.empty:
            continue

        for mkt, mdf in df.groupby("_market"):
            if market and mkt != market:
                continue
            if not mdf.empty:
                market_data[mkt].append((date_label, mdf))

    return market_data


def load_from_yfinance(
    market: str = "JP",
    custom_tickers: Optional[dict[str, list[str]]] = None,
    period_days: int = 252,
    max_workers: int = 8,
) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    """yfinance から最新スナップショットを取得する"""
    from collections import defaultdict

    fetcher = YFinanceFetcher(period_days=period_days, max_workers=max_workers)
    date_label = datetime.today().strftime("%Y-%m-%d")

    print(f"\n  [yfinance] {market} 市場データを取得中 ({date_label})...")

    df = fetcher.fetch_market_snapshot(
        market=market,
        custom_tickers=custom_tickers,
        progress=True,
    )

    if df.empty:
        print(f"  [yfinance] データなし")
        return {}

    # sector_flow_analyzer.py 互換の _market 列を追加
    currency_map = {"JP": "JPY", "US": "USD"}
    if "Currency" not in df.columns:
        df["Currency"] = currency_map.get(market.upper(), "USD")

    # _market 列 (既存アナライザーとの互換性)
    from sector_flow_analyzer import assign_market_per_row
    df = assign_market_per_row(df)

    market_key = market.upper()
    market_data: dict[str, list[tuple[str, pd.DataFrame]]] = defaultdict(list)

    for mkt, mdf in df.groupby("_market"):
        market_data[mkt].append((date_label, mdf))
        print(f"  [yfinance] {MARKET_LABELS.get(mkt, mkt)}: {len(mdf)} 銘柄")

    return market_data


# ---------------------------------------------------------------------------
# Main Analysis Pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    market_data: dict[str, list[tuple[str, pd.DataFrame]]],
    edinet_events: list[EdinetEvent],
    macro_indicators: list[MacroIndicator],
    macro_scores: dict[str, float],
    args: argparse.Namespace,
) -> None:
    """マーケットごとに分析を実行してレポートを出力する"""
    estat_fetcher = EStatFetcher()
    edinet_fetcher = EdinetFetcher()

    # ティッカー → セクターのマッピング (EDINET ブースト用)
    ticker_sector_map: dict[str, str] = {}

    markets = sorted(market_data.keys())

    for market in markets:
        snapshots = market_data[market]
        label = MARKET_LABELS.get(market, market)

        print(f"\n{'#' * 80}")
        print(f"  MARKET: {label}")
        print(f"{'#' * 80}")

        all_metrics: list[list[SectorMetrics]] = []
        latest_df: Optional[pd.DataFrame] = None

        for date_label, df in snapshots:
            print(f"  分析中: {date_label} ({len(df)} 銘柄)")

            # ティッカー → セクターのマッピングを更新
            if "Ticker" in df.columns and "Sector" in df.columns:
                for _, row in df.iterrows():
                    t = str(row.get("Ticker", ""))
                    s = str(row.get("Sector", ""))
                    if t and s:
                        ticker_sector_map[t] = s

            # セクターメトリクス計算
            metrics = compute_sector_metrics(df, date_label)

            # マクロ・イベントブースト適用
            event_boosts = edinet_fetcher.get_sector_event_boosts(
                edinet_events, ticker_sector_map
            )
            metrics = compute_enhanced_scs(metrics, macro_scores, event_boosts)

            all_metrics.append(metrics)
            latest_df = df

        if not all_metrics or latest_df is None:
            print("  データなし")
            continue

        # マクロ指標サマリー表示
        if macro_indicators:
            estat_fetcher.print_summary(macro_indicators)

        # EDINET イベントサマリー表示
        if edinet_events:
            edinet_fetcher.print_summary(edinet_events)

        # ローテーション検出
        rotations = detect_rotation(all_metrics, threshold=args.threshold)

        # 売買候補生成 (EDINET ブースト込み)
        latest_rotation = rotations[-1] if rotations else None
        ticker_events = edinet_fetcher.get_ticker_event_signals(edinet_events)

        candidates = generate_enhanced_signals(
            latest_df, all_metrics[-1], latest_rotation,
            ticker_events=ticker_events,
            top_n=args.top,
        )

        # レポート出力
        print_sector_dashboard(all_metrics, rotations, candidates, market_label=label)

        # マクロスコアのセクター表示
        if macro_scores:
            _print_macro_sector_impact(macro_scores, all_metrics[-1])

        # チャート生成
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

        # CSV エクスポート
        if args.export_csv:
            export_analysis_csv(all_metrics, candidates,
                                output_dir="screener",
                                suffix=f"_{market.lower()}")


def _print_macro_sector_impact(
    macro_scores: dict[str, float],
    latest_metrics: list[SectorMetrics],
) -> None:
    """マクロ経済のセクター影響を表示する"""
    relevant = {m.sector: macro_scores.get(m.sector, 0.0) for m in latest_metrics}
    if not any(v != 0 for v in relevant.values()):
        return

    print(f"\n  {'─' * 60}")
    print("  MACRO IMPACT (マクロ経済のセクター影響)")
    print(f"  {'─' * 60}")
    for sector, score in sorted(relevant.items(), key=lambda x: -abs(x[1])):
        if abs(score) < 0.1:
            continue
        icon = "▲" if score > 0 else "▼"
        bar = "█" * int(abs(score)) + "░" * (10 - int(abs(score)))
        print(f"  {icon} {sector:<28} {score:+.1f}pt  {bar}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enhanced Sector Rotation & Fund Flow Analyzer\n"
            "yfinance + EDINET API + e-STAT API を統合した分析ツール"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- データソース ---
    parser.add_argument(
        "--source",
        choices=["csv", "yfinance", "both"],
        default="yfinance",
        help="データソース: csv=TradingView CSV, yfinance=リアルタイム取得, both=両方 (default: yfinance)",
    )
    parser.add_argument(
        "snapshots_dir",
        nargs="?",
        default="screener/snapshots",
        help="TradingView CSV ディレクトリ (--source csv/both 時に使用)",
    )

    # --- 市場・銘柄設定 ---
    parser.add_argument(
        "--market",
        choices=["JP", "US", "both"],
        default="JP",
        help="分析対象市場 (default: JP)",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=252,
        help="yfinance 価格履歴取得日数 (default: 252 = 約1年)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="yfinance 並列取得スレッド数 (default: 8)",
    )

    # --- 分析パラメータ ---
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="売買候補の表示件数 (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="ローテーション検出の SCS 変化閾値 (default: 3.0)",
    )

    # --- 外部 API ---
    parser.add_argument(
        "--edinet-key",
        default=None,
        help="EDINET API キー (環境変数 EDINET_API_KEY でも可)",
    )
    parser.add_argument(
        "--edinet-days",
        type=int,
        default=5,
        help="EDINET イベントを遡る日数 (default: 5)",
    )
    parser.add_argument(
        "--estat-id",
        default=None,
        help="e-STAT アプリケーション ID (環境変数 ESTAT_APP_ID でも可)",
    )
    parser.add_argument(
        "--simulate-macro",
        action="store_true",
        help="e-STAT API の代わりにシミュレーションデータを使用",
    )

    # --- 出力 ---
    parser.add_argument("--out", default=None, help="チャート出力パス (PNG)")
    parser.add_argument("--no-chart", action="store_true", help="チャート生成をスキップ")
    parser.add_argument("--export-csv", action="store_true", help="分析結果を CSV に出力")

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  Enhanced Sector Rotation & Fund Flow Analyzer")
    print(f"  Source: {args.source}  Market: {args.market}")
    print(f"  Date: {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ─── 1. マクロ経済データ取得 (e-STAT) ───
    print("\n[Step 1/4] マクロ経済指標取得...")
    if args.simulate_macro:
        macro_indicators = get_simulated_indicators()
        print("  シミュレーションデータを使用")
    else:
        estat_id = args.estat_id or os.environ.get("ESTAT_APP_ID", "")
        estat_fetcher = EStatFetcher(app_id=estat_id)
        macro_indicators = estat_fetcher.fetch_key_indicators()

    estat_fetcher = EStatFetcher()
    macro_scores = estat_fetcher.get_sector_macro_scores(macro_indicators)
    if macro_indicators:
        print(f"  {len(macro_indicators)} 件のマクロ指標を取得")

    # ─── 2. EDINET イベント取得 ───
    print("\n[Step 2/4] EDINET イベント取得...")
    edinet_key = args.edinet_key or os.environ.get("EDINET_API_KEY", "")
    edinet_fetcher = EdinetFetcher(api_key=edinet_key)

    if edinet_key:
        edinet_events = edinet_fetcher.fetch_recent_events(days=args.edinet_days)
        print(f"  {len(edinet_events)} 件のイベントを取得 (直近 {args.edinet_days} 日間)")
    else:
        edinet_events = []
        print("  EDINET_API_KEY 未設定: イベントデータをスキップ")
        print("  設定方法: export EDINET_API_KEY=your_key")

    # ─── 3. 株価データ取得 ───
    print("\n[Step 3/4] 株価データ取得...")

    markets_to_analyze = (
        ["JP", "US"] if args.market == "both" else [args.market]
    )

    all_market_data: dict = {}

    for mkt in markets_to_analyze:
        if args.source in ("yfinance", "both"):
            yf_data = load_from_yfinance(
                market=mkt,
                period_days=args.period,
                max_workers=args.workers,
            )
            if yf_data:
                for k, v in yf_data.items():
                    if k not in all_market_data:
                        all_market_data[k] = []
                    all_market_data[k].extend(v)

        if args.source in ("csv", "both"):
            csv_data = load_from_csv(args.snapshots_dir, market=mkt)
            for k, v in csv_data.items():
                if k not in all_market_data:
                    all_market_data[k] = []
                all_market_data[k].extend(v)

    if not all_market_data:
        print("\nError: データが取得できませんでした。")
        print("  --source yfinance の場合: ネットワーク接続を確認してください")
        print("  --source csv の場合: screener/snapshots/ にCSVファイルを配置してください")
        sys.exit(1)

    total_stocks = sum(
        len(df) for snapshots in all_market_data.values() for _, df in snapshots
    )
    print(f"\n  合計 {total_stocks} 銘柄のデータを取得")

    # ─── 4. セクター分析 ───
    print("\n[Step 4/4] セクター分析実行...")
    run_analysis(
        market_data=all_market_data,
        edinet_events=edinet_events,
        macro_indicators=macro_indicators,
        macro_scores=macro_scores,
        args=args,
    )

    print("\n分析完了。")
    if not args.no_chart:
        print("  チャート: screener/sector_dashboard_*.png")
    if args.export_csv:
        print("  CSV: screener/sector_metrics_*.csv, screener/trade_candidates_*.csv")


if __name__ == "__main__":
    main()
