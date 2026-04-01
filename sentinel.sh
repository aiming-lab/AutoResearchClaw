#!/usr/bin/env bash
# sentinel.sh — Watchdog for AutoResearchClaw pipeline process.
#
# Monitors the pipeline heartbeat file and auto-restarts on crash.
# Inspired by Sibyl's sentinel watchdog design.
#
# Usage:
#   ./sentinel.sh <run_dir> [--python <python_path>]
#
# The pipeline runner writes heartbeat.json after each stage. If the
# heartbeat goes stale (>5 min) and the PID is dead, sentinel restarts.
#
# Configuration via environment:
#   SENTINEL_CHECK_INTERVAL  — seconds between checks (default: 60)
#   SENTINEL_STALE_THRESHOLD — seconds before heartbeat is stale (default: 300)
#   SENTINEL_MAX_RETRIES     — max restart attempts (default: 5)
#   SENTINEL_COOLDOWN        — seconds to wait after 3 consecutive failures (default: 360)

set -euo pipefail

# --- Arguments ---
RUN_DIR="${1:?Usage: sentinel.sh <run_dir> [--python <path>]}"
PYTHON_PATH="python"
shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            PYTHON_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# --- Configuration ---
CHECK_INTERVAL="${SENTINEL_CHECK_INTERVAL:-60}"
STALE_THRESHOLD="${SENTINEL_STALE_THRESHOLD:-300}"
MAX_RETRIES="${SENTINEL_MAX_RETRIES:-5}"
COOLDOWN="${SENTINEL_COOLDOWN:-360}"

HEARTBEAT_FILE="${RUN_DIR}/heartbeat.json"
RECOVERY_LOG="${RUN_DIR}/sentinel_recovery.log"
FAILED_LOG="${RUN_DIR}/sentinel_failed.log"

retry_count=0
consecutive_failures=0

log() {
    local msg="[sentinel $(date '+%Y-%m-%dT%H:%M:%S')] $1"
    echo "$msg"
    echo "$msg" >> "$RECOVERY_LOG"
}

# --- Check if heartbeat is stale ---
is_stale() {
    if [[ ! -f "$HEARTBEAT_FILE" ]]; then
        return 0  # No heartbeat = stale
    fi

    local now
    now=$(date +%s)

    # Extract timestamp from heartbeat.json
    local hb_ts
    hb_ts=$(python3 -c "
import json, sys
try:
    data = json.load(open('${HEARTBEAT_FILE}'))
    from datetime import datetime
    ts = datetime.fromisoformat(data['timestamp'])
    print(int(ts.timestamp()))
except Exception:
    print(0)
" 2>/dev/null || echo 0)

    local age=$(( now - hb_ts ))
    [[ $age -gt $STALE_THRESHOLD ]]
}

# --- Check if PID is alive ---
pid_alive() {
    local pid_file="${RUN_DIR}/pipeline.pid"
    if [[ ! -f "$pid_file" ]]; then
        return 1
    fi
    local pid
    pid=$(cat "$pid_file" 2>/dev/null || echo "")
    if [[ -z "$pid" ]]; then
        return 1
    fi
    kill -0 "$pid" 2>/dev/null
}

# --- Check for active subprocesses ---
has_active_children() {
    local pid_file="${RUN_DIR}/pipeline.pid"
    if [[ ! -f "$pid_file" ]]; then
        return 1
    fi
    local pid
    pid=$(cat "$pid_file" 2>/dev/null || echo "")
    if [[ -z "$pid" ]]; then
        return 1
    fi
    # Check if any child processes exist
    pgrep -P "$pid" > /dev/null 2>&1
}

# --- Restart pipeline ---
restart_pipeline() {
    log "Attempting pipeline restart (attempt $((retry_count + 1))/${MAX_RETRIES})"

    $PYTHON_PATH -m researchclaw run --resume --output "$RUN_DIR" &
    local new_pid=$!
    echo "$new_pid" > "${RUN_DIR}/pipeline.pid"

    log "Pipeline restarted with PID ${new_pid}"
    retry_count=$((retry_count + 1))
}

# --- Main loop ---
log "Sentinel started for ${RUN_DIR}"
log "Check interval: ${CHECK_INTERVAL}s, Stale threshold: ${STALE_THRESHOLD}s"
log "Max retries: ${MAX_RETRIES}, Cooldown: ${COOLDOWN}s"

while true; do
    sleep "$CHECK_INTERVAL"

    # If PID is alive, reset failure counter
    if pid_alive; then
        consecutive_failures=0
        continue
    fi

    # PID is dead — check if heartbeat is stale
    if ! is_stale; then
        # Heartbeat is fresh but PID is gone — might have just exited normally
        continue
    fi

    # Don't interrupt active subprocesses
    if has_active_children; then
        log "Active subprocesses detected — skipping restart"
        continue
    fi

    # Check retry limit
    if [[ $retry_count -ge $MAX_RETRIES ]]; then
        log "Max retries (${MAX_RETRIES}) reached — sentinel giving up"
        echo "Sentinel failed after ${MAX_RETRIES} retries at $(date)" >> "$FAILED_LOG"
        exit 1
    fi

    # Cooldown after consecutive failures
    consecutive_failures=$((consecutive_failures + 1))
    if [[ $consecutive_failures -ge 3 ]]; then
        log "3 consecutive failures — cooling down for ${COOLDOWN}s"
        sleep "$COOLDOWN"
        consecutive_failures=0
    fi

    restart_pipeline
done
