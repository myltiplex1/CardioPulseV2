#!/usr/bin/env bash
# CardioPulse launcher for macOS and Linux. Run with:  bash start_cardiopulse.sh
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
export TFHUB_CACHE_DIR="$ROOT/.tfhub_cache"

if [ ! -x "$PY" ]; then
    echo "First run: setting up the Python environment (needs internet, 5 to 15 minutes)."
    if command -v python3.12 >/dev/null 2>&1; then python3.12 -m venv "$VENV"; else python3 -m venv "$VENV"; fi
    "$PY" -m pip install --upgrade pip
    "$PY" -m pip install -r "$ROOT/requirements.txt"
fi

if [ ! -d "$TFHUB_CACHE_DIR" ]; then
    echo "Downloading the YAMNet model once, so the dashboard can run without internet..."
    "$PY" -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1'); print('YAMNet saved.')"
fi

echo
echo "Switch on the device and join the Wi-Fi network HeartMonitor_ESP32 (password heartbeat123)."
echo "The dashboard runs at http://localhost:8501. Press Ctrl+C to stop."
echo
cd "$ROOT/cardiopulse"
( sleep 10; (command -v xdg-open >/dev/null && xdg-open http://localhost:8501) || (command -v open >/dev/null && open http://localhost:8501) ) >/dev/null 2>&1 &
"$PY" -m streamlit run dashboard_live.py
