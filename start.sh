#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Lumin_Core Launcher  .WxE.
#  Starts luminova_core_v2_2.py as the always-running voice backend.
#
#  The terminal window minimises itself automatically after launch.
#  Logs are written to ~/luminova_logs/core.log
#
#  To start automatically on boot, add this line to /etc/rc.local:
#    su pi -c "cd /home/pi/Lumin_Core && bash start.sh" &
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd "$(dirname "$0")"   # Always run from this folder so imports work

echo "========================================"
echo "  Lumin_Core v2.2 WxE — Starting"
echo "========================================"

# ── Python check ──────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found."
    exit 1
fi

# ── Dependency check (warn, don't abort) ──────────────────────────────────────
python3 -c "import numpy" 2>/dev/null || echo "WARN: numpy missing — pip install numpy"
python3 -c "import faster_whisper" 2>/dev/null || echo "WARN: faster-whisper missing — pip install faster-whisper"
python3 -c "import sounddevice" 2>/dev/null || echo "WARN: sounddevice missing — pip install sounddevice (needs libportaudio2)"

# ── Ollama check and start ─────────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    if ! curl -s --max-time 2 http://localhost:11434/api/tags &>/dev/null; then
        echo "Starting Ollama..."
        ollama serve &>/dev/null &
        sleep 3
    else
        echo "Ollama already running."
    fi

    # Make sure our model is available
    if ! ollama list 2>/dev/null | grep -q "qwen2.5"; then
        echo "Pulling qwen2.5:1.5b (first run only, takes a few minutes)..."
        ollama pull qwen2.5:1.5b
    fi
else
    echo "WARN: ollama not found. Install from https://ollama.ai"
    echo "      The core will still start but LLM replies will fail."
fi

# ── Minimise this terminal window ─────────────────────────────────────────────
# Hides the terminal 0.8 seconds after launch so the Pi desktop stays clean.
if command -v xdotool &>/dev/null && [ -n "$DISPLAY" ]; then
    (sleep 0.8 && xdotool getactivewindow windowminimize 2>/dev/null) &
fi

# ── Start the core ─────────────────────────────────────────────────────────────
echo "Starting Luminova voice core..."
echo "Logs: ~/luminova_logs/core.log"
echo "Press Ctrl+C to stop."
echo ""

python3 luminova_core_v2_2.py

# Keep terminal open if it crashed so you can read the error
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "Core exited with error $EXIT_CODE. Check ~/luminova_logs/core.log"
    echo "Press Enter to close."
    read -r
fi
