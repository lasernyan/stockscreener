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

- `screener/sector_flow_analyzer.py` — Main analyzer (sector metrics, rotation detection, trade signals, charts, alerts, diff, watchlist, HTML)
- `screener/snapshots/` — CSV data directory (TradingView screener exports)
- `screener/sector_dashboard_jp.png` — Japan market dashboard
- `screener/sector_dashboard_us.png` — US market dashboard
- `screener/sector_metrics*.csv` — Exported sector metrics
- `screener/trade_candidates*.csv` — Exported trade candidates
- `screener/report_jp.html` — Japan market HTML report
- `screener/report_us.html` — US market HTML report
- `screener/.sector_cache.json` — Incremental processing cache (auto-generated)
- `screener/.watchlist.json` — Watchlist tracking data (auto-generated)

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
```

When the user mentions running this script or provides a CSV file path, assist with running `screener/analyze.sh` with the appropriate arguments.

## CLI Options

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
