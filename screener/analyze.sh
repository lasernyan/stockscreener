#!/usr/bin/env bash
# ============================================================
# Auto-Analyze: CSV取り込み & セクター分析 ワンコマンド実行
# ============================================================
# Usage:
#   ./screener/analyze.sh <csv_file> [csv_file2 ...]
#   ./screener/analyze.sh ~/Downloads/Stock_*.csv
#   ./screener/analyze.sh   (引数なし = snapshots/ 内の既存データで再分析)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SNAPSHOTS_DIR="${SCRIPT_DIR}/snapshots"
ANALYZER="${SCRIPT_DIR}/sector_flow_analyzer.py"

mkdir -p "$SNAPSHOTS_DIR"

# --- CSVファイルの取り込み ---
if [ $# -gt 0 ]; then
    for csvfile in "$@"; do
        if [ ! -f "$csvfile" ]; then
            echo "  [SKIP] Not found: $csvfile"
            continue
        fi
        filename="$(basename "$csvfile")"
        dest="${SNAPSHOTS_DIR}/${filename}"
        if [ -f "$dest" ]; then
            echo "  [EXISTS] ${filename} (already in snapshots/)"
        else
            cp "$csvfile" "$dest"
            echo "  [ADDED] ${filename} -> snapshots/"
        fi
    done
    echo ""
fi

# --- 分析実行 ---
echo "Running sector flow analysis..."
echo ""
python3 "$ANALYZER" "$SNAPSHOTS_DIR" --top 5 --export-csv

echo ""
echo "Done. Check screener/sector_dashboard_*.png for charts."
