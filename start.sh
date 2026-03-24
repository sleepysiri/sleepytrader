#!/usr/bin/env bash
# SleepyTrader - ABM Stock Market Simulator
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "╔══════════════════════════════════════════╗"
echo "║          SLEEPY TRADER  💤📈              ║"
echo "║       ABM Stock Market Simulator         ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.9+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "[INFO] Python $PYTHON_VERSION detected."

# Create venv if needed
if [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/.venv"
fi

# Activate venv
source "$PROJECT_DIR/.venv/bin/activate"

# Install dependencies
echo "[INFO] Installing dependencies..."
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet -r "$PROJECT_DIR/requirements.txt"

echo ""
echo "[INFO] Dependencies installed."
echo ""

# Load .env if present
if [ -f "$PROJECT_DIR/.env" ]; then
    echo "[INFO] Loading API keys from .env file..."
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

# Optional: check for API keys
if [ -n "$GROQ_API_KEY" ]; then
    echo "[INFO] GROQ_API_KEY detected - AI agent thoughts enabled via Groq."
elif [ -n "$TOGETHER_API_KEY" ]; then
    echo "[INFO] TOGETHER_API_KEY detected - AI agent thoughts enabled via Together AI."
elif [ -n "$OPENROUTER_API_KEY" ]; then
    echo "[INFO] OPENROUTER_API_KEY detected - AI agent thoughts enabled via OpenRouter."
else
    echo "[INFO] No LLM API key found. Agents will use rule-based thoughts."
    echo "[INFO] Set GROQ_API_KEY, TOGETHER_API_KEY, or OPENROUTER_API_KEY to enable AI thoughts."
fi

echo ""
echo "[INFO] Starting simulation server on http://localhost:8000"
echo "[INFO] Open your browser and navigate to: http://localhost:8000"
echo "[INFO] Press Ctrl+C to stop."
echo ""

# Run the server
cd "$PROJECT_DIR/backend"
python3 -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info
