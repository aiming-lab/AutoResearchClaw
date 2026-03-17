#!/bin/bash
# Start MetaClaw proxy for AutoResearchClaw integration.
#
# Usage:
#   ./scripts/metaclaw_start.sh              # skills_only mode (default)
#   ./scripts/metaclaw_start.sh madmax       # madmax mode (with RL training)
#   ./scripts/metaclaw_start.sh skills_only  # skills_only mode (explicit)

set -e

MODE="${1:-skills_only}"
PORT="${2:-30000}"

METACLAW_DIR="/home/jqliu/projects/MetaClaw"
VENV="$METACLAW_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "ERROR: MetaClaw venv not found at $VENV"
    echo "Run: cd $METACLAW_DIR && python -m venv .venv && source .venv/bin/activate && pip install -e '.[evolve,embedding]'"
    exit 1
fi

echo "Starting MetaClaw in ${MODE} mode on port ${PORT}..."

# Activate venv and start
source "$VENV/bin/activate"
exec metaclaw start --mode "$MODE" --port "$PORT"
