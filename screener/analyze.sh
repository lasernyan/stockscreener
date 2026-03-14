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
#   AI=1 ./screener/analyze.sh           # Claude AI による市場分析 (要 ANTHROPIC_API_KEY)
#   FETCH=1 ./screener/analyze.sh        # API から自動データ取得 (ETF/CRYPTO/FX/COMMODITYも取得)
#   BACKTEST=1 ./screener/analyze.sh     # バックテスト実行
#   MARKET=ETF ./screener/analyze.sh     # FETCH時のマーケット指定 (ETF/BOND/COMMODITY/CRYPTO/FX/ALL)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SNAPSHOTS_DIR="${SCRIPT_DIR}/snapshots"
ANALYZER="${SCRIPT_DIR}/sector_flow_analyzer.py"
FETCHER="${SCRIPT_DIR}/data_fetcher.py"
BACKTESTER="${SCRIPT_DIR}/backtester.py"
CLAUDE_ANALYST="${SCRIPT_DIR}/claude_analyst.py"

mkdir -p "$SNAPSHOTS_DIR"

# --- Step 0: API からデータ自動取得 (FETCH=1) ---
if [ "${FETCH:-0}" = "1" ]; then
    echo "=== [STEP 0] Fetching multi-asset data via API ==="
    FETCH_MARKET="${MARKET:-ALL}"
    python3 "$FETCHER" --market $FETCH_MARKET --output-dir "$SNAPSHOTS_DIR"
    echo ""
fi

# --- CSVファイルの取り込み ---
csv_args=()
for arg in "$@"; do
    if [ -f "$arg" ]; then
        csv_args+=("$arg")
    fi
done

if [ ${#csv_args[@]} -gt 0 ]; then
    echo "=== Importing CSV files ==="
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

# --- Step 1: セクターフロー分析 ---
echo "=== [STEP 1] Sector flow analysis ==="
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

python3 "$ANALYZER" "$SNAPSHOTS_DIR" $EXTRA_OPTS
echo ""

# --- Step 2: バックテスト (BACKTEST=1) ---
if [ "${BACKTEST:-0}" = "1" ]; then
    echo "=== [STEP 2] Backtesting ==="
    BT_MARKET="${BT_MARKET:-US}"
    python3 "$BACKTESTER" \
        --snapshot-dir "$SNAPSHOTS_DIR" \
        --market "$BT_MARKET" \
        --top-long 3 \
        --export-html \
        --export-csv
    echo ""
fi

# --- Step 3: Claude AI 分析 (AI=1 かつ ANTHROPIC_API_KEY が設定済み) ---
if [ "${AI:-0}" = "1" ]; then
    echo "=== [STEP 3] Claude AI Analysis ==="
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "  [WARN] ANTHROPIC_API_KEY が未設定です。AI分析をスキップします。"
        echo "  export ANTHROPIC_API_KEY='your-key' を実行してから再試行してください。"
    else
        AI_MARKET="${AI_MARKET:-ALL}"
        AI_OPTS="--market $AI_MARKET"
        if [ "${BACKTEST:-0}" = "1" ]; then
            AI_OPTS="$AI_OPTS --backtest"
        fi
        if [ "${HTML:-1}" = "1" ]; then
            AI_OPTS="$AI_OPTS --export-html"
        fi
        python3 "$CLAUDE_ANALYST" --screener-dir "$SCRIPT_DIR" $AI_OPTS
    fi
    echo ""
fi

echo "=== Done ==="
echo "Generated files:"
echo "  screener/sector_dashboard_*.png       (sector charts)"
echo "  screener/report_*.html                (HTML reports)"
if [ "${BACKTEST:-0}" = "1" ]; then
    echo "  screener/backtest_report_*.html       (backtest report)"
    echo "  screener/backtest_trades_*.csv        (trade history)"
fi
if [ "${AI:-0}" = "1" ]; then
    echo "  screener/ai_analysis_*.html           (Claude AI analysis)"
fi
echo "  screener/.watchlist.json              (watchlist tracking)"
