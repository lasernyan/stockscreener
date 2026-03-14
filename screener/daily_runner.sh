#!/usr/bin/env bash
# ============================================================
# Daily Runner — 毎日の自動分析スクリプト
# ============================================================
# cron から呼び出されることを想定した本番用ランナー。
# ログ出力・エラー通知・ロック機構を備える。
#
# セットアップ:
#   1. 実行権限を付与
#      chmod +x screener/daily_runner.sh
#
#   2. APIキーを .env ファイルに保存 (gitignore済み)
#      echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
#
#   3. cron に登録 (crontab -e で以下を追加)
#      0 21 * * * /path/to/stockscreener/screener/daily_runner.sh >> /path/to/stockscreener/logs/cron.log 2>&1
#
# 環境変数:
#   RUN_FETCH=1      # API データ取得を有効化 (default: 1)
#   RUN_BACKTEST=1   # バックテストを有効化 (default: 1)
#   RUN_AI=1         # Claude AI 分析を有効化 (Anthropic API, default: 0)
#   USE_COWORK=1     # COWORK モード: Claude Code Agent SDK で分析 (default: 1)
#                    # USE_COWORK=1 の場合 RUN_AI は無視される
#   MARKET_FETCH=ALL # 取得マーケット指定 (default: ALL)
# ============================================================

set -euo pipefail

# ============================================================
# パス設定
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="${PROJECT_DIR}/logs"
LOCK_FILE="${PROJECT_DIR}/.daily_runner.lock"
ENV_FILE="${PROJECT_DIR}/.env"

mkdir -p "$LOG_DIR"

# ============================================================
# ログ関数
# ============================================================
LOG_FILE="${LOG_DIR}/daily_$(date +%Y-%m-%d).log"

log() {
    local level="$1"
    shift
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${ts}] [${level}] ${msg}" | tee -a "$LOG_FILE"
}

log_info()  { log "INFO " "$@"; }
log_warn()  { log "WARN " "$@"; }
log_error() { log "ERROR" "$@"; }

# ============================================================
# 二重起動防止 (lockfile)
# ============================================================
if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "unknown")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        log_warn "Already running (PID=$OLD_PID). Skipping."
        exit 0
    else
        log_warn "Stale lock found (PID=$OLD_PID). Removing."
        rm -f "$LOCK_FILE"
    fi
fi

echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"; log_info "Runner exited."' EXIT

# ============================================================
# 環境変数の読み込み (.env ファイル)
# ============================================================
if [ -f "$ENV_FILE" ]; then
    # セキュアに読み込む (コマンドインジェクション防止)
    while IFS='=' read -r key value; do
        # コメント行・空行をスキップ
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        # 値のクォートを除去
        value="${value%\"}"
        value="${value#\"}"
        value="${value%\'}"
        value="${value#\'}"
        export "$key=$value"
    done < <(grep -E '^[A-Z_]+=.' "$ENV_FILE" || true)
    log_info "Loaded .env from $ENV_FILE"
fi

# ============================================================
# 設定
# ============================================================
RUN_FETCH="${RUN_FETCH:-1}"
RUN_BACKTEST="${RUN_BACKTEST:-1}"
RUN_AI="${RUN_AI:-0}"
USE_COWORK="${USE_COWORK:-1}"   # デフォルトは COWORK モード
MARKET_FETCH="${MARKET_FETCH:-ALL}"

log_info "========================================"
log_info "Daily Runner started"
log_info "  FETCH=$RUN_FETCH  BACKTEST=$RUN_BACKTEST"
log_info "  COWORK=$USE_COWORK  AI=$RUN_AI"
log_info "  MARKET=$MARKET_FETCH"
log_info "========================================"

START_TIME=$(date +%s)
ERRORS=()

# ============================================================
# Step 0: API データ取得
# ============================================================
if [ "$RUN_FETCH" = "1" ]; then
    log_info "[STEP 0] Fetching multi-asset data..."
    if FETCH=1 MARKET="$MARKET_FETCH" \
        python3 "${SCRIPT_DIR}/data_fetcher.py" \
            --market "$MARKET_FETCH" \
            --output-dir "${SCRIPT_DIR}/snapshots" \
            >> "$LOG_FILE" 2>&1; then
        log_info "[STEP 0] Fetch completed."
    else
        log_warn "[STEP 0] Fetch failed or partially failed (continuing)."
        ERRORS+=("data_fetcher")
    fi
fi

# ============================================================
# Step 1: セクターフロー分析
# ============================================================
log_info "[STEP 1] Running sector flow analysis..."
if python3 "${SCRIPT_DIR}/sector_flow_analyzer.py" \
        "${SCRIPT_DIR}/snapshots" \
        --top 5 --export-csv --html \
        >> "$LOG_FILE" 2>&1; then
    log_info "[STEP 1] Sector analysis completed."
else
    log_error "[STEP 1] Sector analysis FAILED."
    ERRORS+=("sector_flow_analyzer")
fi

# ============================================================
# Step 2: バックテスト
# ============================================================
if [ "$RUN_BACKTEST" = "1" ]; then
    log_info "[STEP 2] Running backtest..."
    if python3 "${SCRIPT_DIR}/backtester.py" \
            --snapshot-dir "${SCRIPT_DIR}/snapshots" \
            --market US \
            --top-long 3 \
            --export-html --export-csv \
            >> "$LOG_FILE" 2>&1; then
        log_info "[STEP 2] Backtest completed."
    else
        log_warn "[STEP 2] Backtest failed (continuing)."
        ERRORS+=("backtester")
    fi
fi

# ============================================================
# Step 3: 分析レポート生成 (COWORK or Anthropic API)
# ============================================================
if [ "$USE_COWORK" = "1" ]; then
    # --- COWORK モード: Claude Code Agent SDK を使用 ---
    # Claude Code Pro/Max サブスクリプションで動作 (APIキー不要)
    log_info "[STEP 3] Running COWORK analysis (Claude Code Agent)..."
    COWORK_OPTS="--market ${MARKET_FETCH}"
    if [ "$RUN_FETCH" != "1" ]; then
        COWORK_OPTS="$COWORK_OPTS --no-fetch"
    fi
    if [ "$RUN_BACKTEST" != "1" ]; then
        COWORK_OPTS="$COWORK_OPTS --no-backtest"
    fi
    if python3 "${SCRIPT_DIR}/cowork_runner.py" \
            --project-dir "$PROJECT_DIR" \
            $COWORK_OPTS \
            >> "$LOG_FILE" 2>&1; then
        log_info "[STEP 3] COWORK analysis completed."
    else
        log_warn "[STEP 3] COWORK analysis failed (continuing)."
        log_warn "  Claude Code CLI がインストール・ログイン済みか確認してください"
        ERRORS+=("cowork_runner")
    fi

elif [ "$RUN_AI" = "1" ]; then
    # --- Anthropic API モード (USE_COWORK=0 かつ RUN_AI=1 のとき) ---
    log_info "[STEP 3] Running Claude API analysis..."
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        log_warn "[STEP 3] ANTHROPIC_API_KEY not set. Skipping AI analysis."
        log_warn "  Set it in ${ENV_FILE}: ANTHROPIC_API_KEY=sk-ant-..."
    else
        if python3 "${SCRIPT_DIR}/claude_analyst.py" \
                --screener-dir "$SCRIPT_DIR" \
                --market ALL \
                --export-html \
                $([ "$RUN_BACKTEST" = "1" ] && echo "--backtest" || echo "") \
                >> "$LOG_FILE" 2>&1; then
            log_info "[STEP 3] AI analysis completed."
        else
            log_warn "[STEP 3] AI analysis failed (continuing)."
            ERRORS+=("claude_analyst")
        fi
    fi
fi

# ============================================================
# 完了サマリー
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_FMT="$((ELAPSED / 60))m $((ELAPSED % 60))s"

log_info "========================================"
if [ ${#ERRORS[@]} -eq 0 ]; then
    log_info "All steps completed successfully in ${ELAPSED_FMT}"
else
    log_warn "Completed with errors in ${ELAPSED_FMT}: ${ERRORS[*]}"
fi
log_info "Log: ${LOG_FILE}"
log_info "========================================"

# エラーがあった場合は exit 1 (cron がメール通知するよう設定可能)
[ ${#ERRORS[@]} -eq 0 ] && exit 0 || exit 1
