#!/usr/bin/env bash
# ============================================================
# Daily Sector Analysis Runner
# ============================================================
# Usage:
#   ./screener/run_analysis.sh                       # 全スナップショットを分析
#   ./screener/run_analysis.sh --export-csv           # CSV出力も併用
#   ./screener/run_analysis.sh --no-chart --top 5     # チャートなし、候補5件
#
# TradingView からの CSV エクスポート手順:
#   1. TradingView のスクリーナーを開く (https://www.tradingview.com/screener/)
#   2. フィルタを設定 (例: US市場, 時価総額 > $10B)
#   3. 必要カラムを表示: Sector, Change%, Volume, Relative Volume,
#      Market Cap, SMA20/50/200, RSI, MACD, ADX, ATR
#   4. 右上の「Export」→ CSV をダウンロード
#   5. screener/snapshots/ にファイル名 screener_YYYY-MM-DD.csv で保存
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SNAPSHOTS_DIR="${SCRIPT_DIR}/snapshots"

# スナップショットディレクトリの確認
if [ ! -d "$SNAPSHOTS_DIR" ]; then
    echo "Error: Snapshots directory not found: $SNAPSHOTS_DIR"
    echo "Create it and add TradingView CSV exports."
    exit 1
fi

# CSV ファイルの存在確認
CSV_COUNT=$(find "$SNAPSHOTS_DIR" -name "*.csv" | wc -l)
if [ "$CSV_COUNT" -eq 0 ]; then
    echo "Error: No CSV files found in $SNAPSHOTS_DIR"
    echo "Export CSV from TradingView Screener and save it there."
    exit 1
fi

echo "============================================================"
echo "  Sector Rotation & Fund Flow Analyzer"
echo "  Snapshots: $CSV_COUNT file(s)"
echo "============================================================"

# 実行
python3 "${SCRIPT_DIR}/sector_flow_analyzer.py" "$SNAPSHOTS_DIR" "$@"
