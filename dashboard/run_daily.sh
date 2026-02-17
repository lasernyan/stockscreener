#!/usr/bin/env bash
#
# Gold & Silver Dashboard - Daily Update Script
#
# Usage:
#   ./dashboard/run_daily.sh           # Run once (manual)
#   ./dashboard/run_daily.sh --cron    # Install as daily cron job
#   ./dashboard/run_daily.sh --remove  # Remove cron job
#
# Cron runs daily at 07:00 UTC (after Asian markets, before EU open)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$SCRIPT_DIR/data/daily_update.log"
PYTHON="${PYTHON:-python3}"
CRON_SCHEDULE="0 7 * * *"
CRON_TAG="gold_silver_dashboard"

mkdir -p "$SCRIPT_DIR/data"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_update() {
    log "=== Starting daily dashboard update ==="

    # Step 1: Fetch data
    log "Fetching market data..."
    if $PYTHON "$SCRIPT_DIR/fetch_data.py" >> "$LOG_FILE" 2>&1; then
        log "Data fetch completed."
    else
        log "ERROR: Data fetch failed (exit code: $?)"
        return 1
    fi

    # Step 2: Generate HTML dashboard
    log "Generating HTML dashboard..."
    if $PYTHON "$SCRIPT_DIR/generate_dashboard.py" >> "$LOG_FILE" 2>&1; then
        log "Dashboard generated."
    else
        log "ERROR: Dashboard generation failed (exit code: $?)"
        return 1
    fi

    # Step 3: Show summary
    if [ -f "$SCRIPT_DIR/gold_silver_dashboard.html" ]; then
        SIZE=$(wc -c < "$SCRIPT_DIR/gold_silver_dashboard.html")
        log "Output: $SCRIPT_DIR/gold_silver_dashboard.html ($SIZE bytes)"
    fi

    log "=== Daily update complete ==="
}

install_cron() {
    local CRON_CMD="$CRON_SCHEDULE cd $PROJECT_DIR && $SCRIPT_DIR/run_daily.sh >> $LOG_FILE 2>&1 # $CRON_TAG"

    # Remove existing entry first
    (crontab -l 2>/dev/null | grep -v "$CRON_TAG") | crontab - 2>/dev/null || true

    # Add new entry
    (crontab -l 2>/dev/null; echo "$CRON_CMD") | crontab -
    echo "Cron job installed: runs daily at 07:00 UTC"
    echo "  Schedule: $CRON_SCHEDULE"
    echo "  Log: $LOG_FILE"
    echo ""
    echo "Current crontab:"
    crontab -l 2>/dev/null | grep "$CRON_TAG" || echo "  (none found)"
}

remove_cron() {
    (crontab -l 2>/dev/null | grep -v "$CRON_TAG") | crontab - 2>/dev/null || true
    echo "Cron job removed."
}

# Parse arguments
case "${1:-}" in
    --cron)
        install_cron
        ;;
    --remove)
        remove_cron
        ;;
    --help|-h)
        echo "Usage: $0 [--cron|--remove|--help]"
        echo ""
        echo "  (no args)  Run data fetch + dashboard generation once"
        echo "  --cron     Install as daily cron job (07:00 UTC)"
        echo "  --remove   Remove cron job"
        ;;
    *)
        run_update
        ;;
esac
