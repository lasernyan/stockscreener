# Project: Sector Rotation & Fund Flow Analyzer

## Auto-Analyze Rule

When the user uploads a CSV file (TradingView screener export), automatically perform the following steps without waiting for additional instructions:

1. **Detect**: Identify any new CSV file uploaded to the workspace (typically matching pattern `*_YYYY-MM-DD_*.csv` or `Stock_*.csv`)
2. **Place**: Move/copy the file into `screener/snapshots/`
3. **Analyze**: Run the sector flow analyzer:
   ```bash
   python3 screener/sector_flow_analyzer.py screener/snapshots/ --top 5 --export-csv --html
   ```
4. **Report**: Display key findings to the user:
   - ALERTS (異常検知 — 全面安/全面高、RVOL異常、資金フロー急反転、SCS急変)
   - DAILY DIFF (前日比 — SCS変動、フロー反転、Breadth変化)
   - LONG candidates (top picks)
   - SHORT candidates (top picks)
   - Notable sector rotation signals
   - WATCHLIST (過去候補のP&L追跡)
5. **Charts**: Show the generated dashboard images (`screener/sector_dashboard_*.png`)

## Project Structure

### Core
- `screener/sector_flow_analyzer.py` — Main analyzer (sector metrics, rotation detection, trade signals, charts, alerts, diff, watchlist, HTML)
- `screener/snapshots/` — CSV data directory (TradingView exports + API-fetched data)
- `screener/sector_dashboard_jp.png` — Japan market dashboard
- `screener/sector_dashboard_us.png` — US market dashboard
- `screener/sector_metrics*.csv` — Exported sector metrics
- `screener/trade_candidates*.csv` — Exported trade candidates
- `screener/report_jp.html` — Japan market HTML report
- `screener/report_us.html` — US market HTML report
- `screener/.sector_cache.json` — Incremental processing cache (auto-generated)
- `screener/.watchlist.json` — Watchlist tracking data (auto-generated)

### New Modules (API化・バックテスト・AI分析)
- `screener/data_fetcher.py` — Multi-asset API data fetcher (ETF/債券/暗号資産/コモディティ/FX)
- `screener/backtester.py` — Walk-forward backtesting framework (SCSシグナル検証)
- `screener/claude_analyst.py` — Claude Opus 4.6 による AI 市場分析

### Backtest Outputs
- `screener/backtest_report_*.html` — バックテストHTMLレポート
- `screener/backtest_trades_*.csv` — トレード履歴
- `screener/walkforward_*.csv` — ウォークフォワード検証結果

### AI Analysis Outputs
- `screener/ai_analysis_*.html` — Claude AI分析レポート

## Quick Analyze Script

For users who cannot upload files directly, use the one-command script:

```bash
# CSVファイルを指定して取り込み＆分析
./screener/analyze.sh ~/Downloads/Stock_2026-02-13_*.csv

# 複数ファイルをまとめて取り込み
./screener/analyze.sh file1.csv file2.csv file3.csv

# 既存データで再分析（引数なし）
./screener/analyze.sh

# 差分モード (直近2日の変化だけ表示)
DIFF=1 ./screener/analyze.sh

# キャッシュ無効化 (全データ再計算)
NO_CACHE=1 ./screener/analyze.sh

# === 新機能 ===

# API から ETF/暗号資産/コモディティ/FX データを自動取得して分析
FETCH=1 ./screener/analyze.sh

# 特定マーケットのみ取得 (ETF/BOND/COMMODITY/CRYPTO/FX/ALL)
FETCH=1 MARKET=CRYPTO ./screener/analyze.sh

# バックテスト実行 (SCSシグナルの統計的検証)
BACKTEST=1 ./screener/analyze.sh

# Claude AI による市場分析 (要 ANTHROPIC_API_KEY)
AI=1 ./screener/analyze.sh

# フル分析 (データ取得 + セクター分析 + バックテスト + AI分析)
FETCH=1 BACKTEST=1 AI=1 ./screener/analyze.sh
```

When the user mentions running this script or provides a CSV file path, assist with running `screener/analyze.sh` with the appropriate arguments.

## New CLI Tools

### data_fetcher.py — Multi-Asset API Fetcher
```bash
python3 screener/data_fetcher.py                    # 全アセット取得
python3 screener/data_fetcher.py --market CRYPTO    # 暗号資産のみ
python3 screener/data_fetcher.py --market ETF BOND  # ETFと債券
python3 screener/data_fetcher.py --list-markets     # 利用可能マーケット一覧
python3 screener/data_fetcher.py --period 6mo       # 6ヶ月データ

# Supported markets: ETF BOND COMMODITY CRYPTO FX (and ALL)
# Output: screener/snapshots/{MARKET}_{date}_{hash}.csv
```

### backtester.py — Walk-Forward Backtester
```bash
python3 screener/backtester.py                          # デフォルト設定でバックテスト
python3 screener/backtester.py --market JP              # 日本市場
python3 screener/backtester.py --top-long 3 --top-short 3
python3 screener/backtester.py --walk-forward 20        # ウォークフォワード (20日訓練)
python3 screener/backtester.py --export-html --export-csv
```

**出力指標**: 総リターン / 年率リターン / シャープレシオ / ソルティノレシオ /
最大ドローダウン / カルマーレシオ / 勝率 / プロフィットファクター / アルファ

### claude_analyst.py — Claude AI 市場分析
```bash
# 要: export ANTHROPIC_API_KEY='sk-ant-...'
python3 screener/claude_analyst.py                   # 全市場 AI 分析
python3 screener/claude_analyst.py --market JP       # 日本市場のみ
python3 screener/claude_analyst.py --quick           # クイックシグナル (Haiku, 低コスト)
python3 screener/claude_analyst.py --backtest        # バックテスト結果も含めて分析
python3 screener/claude_analyst.py --export-html     # HTMLレポート生成

# Model: claude-opus-4-6 (adaptive thinking + streaming)
# Output: screener/ai_analysis_{market}_{datetime}.html
```

## CLI Options (sector_flow_analyzer.py)

```
python3 screener/sector_flow_analyzer.py screener/snapshots/ [OPTIONS]

--top N          候補数 (default: 10)
--export-csv     CSVエクスポート
--html           HTMLレポート生成
--diff           差分レポートモード (直近2日)
--no-cache       キャッシュ無効化
--no-chart       チャート生成スキップ
--no-watchlist   ウォッチリスト無効化
--threshold F    ローテーション検出閾値 (default: 3.0)
```

## CSV Format

Input CSVs are TradingView screener exports containing columns like:
Ticker, Name, Sector, Industry, Close, Change %, RSI, Volume, Relative Volume, SMA50, SMA200, MACD, ADX, etc.

Files are auto-classified as JP or US stocks based on ticker format.

**Note**: `data_fetcher.py` generates compatible CSVs from yfinance API data automatically.
