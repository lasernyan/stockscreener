#!/usr/bin/env bash
# ============================================================
# Auto-Analyze: CSV取り込み & セクター分析 ワンコマンド実行
# ============================================================
# Usage:
#   ./screener/analyze.sh <csv_file> [csv_file2 ...]
#   ./screener/analyze.sh ~/Downloads/Stock_*.csv
#   ./screener/analyze.sh   (引数なし = snapshots/ 内の既存データで再分析)
#
# Options (environment variables):
#   DIFF=1 ./screener/analyze.sh         # 差分モード (直近2日)
#   HTML=1 ./screener/analyze.sh         # HTML出力付き
#   NO_CACHE=1 ./screener/analyze.sh     # キャッシュ無効
#   TOP=10 ./screener/analyze.sh         # 候補数を変更
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SNAPSHOTS_DIR="${SCRIPT_DIR}/snapshots"
ANALYZER="${SCRIPT_DIR}/sector_flow_analyzer.py"

mkdir -p "$SNAPSHOTS_DIR"

# --- CSVファイルの取り込み ---
csv_args=()
other_args=()
for arg in "$@"; do
    if [ -f "$arg" ]; then
        csv_args+=("$arg")
    fi
done

if [ ${#csv_args[@]} -gt 0 ]; then
    for csvfile in "${csv_args[@]}"; do
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

# --- オプション構築 ---
EXTRA_OPTS="--top ${TOP:-5} --export-csv"

if [ "${DIFF:-0}" = "1" ]; then
    EXTRA_OPTS="$EXTRA_OPTS --diff"
fi

if [ "${HTML:-1}" = "1" ]; then
    EXTRA_OPTS="$EXTRA_OPTS --html"
fi

if [ "${NO_CACHE:-0}" = "1" ]; then
    EXTRA_OPTS="$EXTRA_OPTS --no-cache"
fi

# --- 分析実行 ---
echo "Running sector flow analysis..."
echo ""
python3 "$ANALYZER" "$SNAPSHOTS_DIR" $EXTRA_OPTS

echo ""
echo "Done. Check:"
echo "  screener/sector_dashboard_*.png  (charts)"
echo "  screener/report_*.html           (HTML reports)"
echo "  screener/.watchlist.json         (watchlist tracking)"
