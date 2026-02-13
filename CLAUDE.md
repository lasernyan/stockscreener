# Project: Sector Rotation & Fund Flow Analyzer

## Auto-Analyze Rule

When the user uploads a CSV file (TradingView screener export), automatically perform the following steps without waiting for additional instructions:

1. **Detect**: Identify any new CSV file uploaded to the workspace (typically matching pattern `*_YYYY-MM-DD_*.csv` or `Stock_*.csv`)
2. **Place**: Move/copy the file into `screener/snapshots/`
3. **Analyze**: Run the sector flow analyzer:
   ```bash
   python3 screener/sector_flow_analyzer.py screener/snapshots/ --top 5 --export-csv
   ```
4. **Report**: Display key findings to the user:
   - LONG candidates (top picks)
   - SHORT candidates (top picks)
   - Notable sector rotation signals
   - Any significant fund flow changes
5. **Charts**: Show the generated dashboard images (`screener/sector_dashboard_*.png`)

## Project Structure

- `screener/sector_flow_analyzer.py` — Main analyzer (sector metrics, rotation detection, trade signals, charts)
- `screener/snapshots/` — CSV data directory (TradingView screener exports)
- `screener/sector_dashboard_jp.png` — Japan market dashboard
- `screener/sector_dashboard_us.png` — US market dashboard
- `screener/sector_metrics*.csv` — Exported sector metrics
- `screener/trade_candidates*.csv` — Exported trade candidates

## Quick Analyze Script

For users who cannot upload files directly, use the one-command script:

```bash
# CSVファイルを指定して取り込み＆分析
./screener/analyze.sh ~/Downloads/Stock_2026-02-13_*.csv

# 複数ファイルをまとめて取り込み
./screener/analyze.sh file1.csv file2.csv file3.csv

# 既存データで再分析（引数なし）
./screener/analyze.sh
```

When the user mentions running this script or provides a CSV file path, assist with running `screener/analyze.sh` with the appropriate arguments.

## CSV Format

Input CSVs are TradingView screener exports containing columns like:
Ticker, Name, Sector, Industry, Close, Change %, RSI, Volume, Relative Volume, SMA50, SMA200, MACD, ADX, etc.

Files are auto-classified as JP or US stocks based on ticker format.
