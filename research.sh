#!/bin/bash
# AutoResearchClaw launcher
# Usage: ./research.sh "Your research topic here"
#        ./research.sh "Your topic" --no-auto-approve  (stops at gates for human review)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$1" ]; then
  echo "Usage: ./research.sh \"Your research topic\""
  exit 1
fi

TOPIC="$1"
shift
EXTRA_ARGS="$@"

# Set your API key via environment variable or a local .env file (not committed)
# export OPENAI_API_KEY="sk-proj-..."
# Or load from .env: [ -f .env ] && source .env

source .venv/bin/activate

echo "🔬 Starting AutoResearchClaw..."
echo "📝 Topic: $TOPIC"
echo ""

researchclaw run \
  --config config.arc.yaml \
  --topic "$TOPIC" \
  --auto-approve \
  $EXTRA_ARGS
