#!/usr/bin/env bash
# deploy_friction_api.sh
# Run on EC2 (Thesis-Causal-V2) to pull latest code and start the live API.
# Usage:  bash deploy_friction_api.sh

set -e

THESIS_DIR="/home/ubuntu/thesis"
VENV_DIR="$THESIS_DIR/.venv"
API_SCRIPT="$THESIS_DIR/Scratch/Modeling/Causal_Inference/05_G_Computation_LSTMs/live_api/live_friction_api.py"
LOG_FILE="/home/ubuntu/friction_api.log"
PID_FILE="/home/ubuntu/friction_api.pid"

echo "=== Pulling latest code ==="
cd "$THESIS_DIR"
git pull

echo "=== Ensuring virtualenv ==="
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "=== Installing dependencies ==="
pip install --quiet --upgrade pip
pip install --quiet econml scikit-learn geopandas joblib numpy

echo "=== Downloading precomputed data if missing ==="
if [ ! -f "$THESIS_DIR/Data/Zoning_Cases/austin_base_geometries.fgb" ]; then
    echo "  Downloading geometries from R2..."
    curl -o "$THESIS_DIR/Data/Zoning_Cases/austin_base_geometries.fgb" "https://pub-7f58e07bff423d2120acf10aa6bf7a32.r2.dev/public/austin_base_geometries.fgb"
fi

if [ ! -f "$THESIS_DIR/Data/Zoning_Cases/inference_cache.npy" ]; then
    echo "  Downloading inference cache from R2 (390MB)..."
    curl -o "$THESIS_DIR/Data/Zoning_Cases/inference_cache.npy" "https://pub-7f58e07bff423d2120acf10aa6bf7a32.r2.dev/public/inference_cache.npy"
fi

echo "=== Stopping existing server (if running) ==="
if [ -f "$PID_FILE" ]; then
    kill "$(cat $PID_FILE)" 2>/dev/null || true
    rm -f "$PID_FILE"
fi

echo "=== Starting Live Friction API ==="
PORT=8001 \
API_ROOT="$THESIS_DIR" \
CORS_ORIGINS="*" \
nohup python "$API_SCRIPT" >> "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started with PID $(cat $PID_FILE). Logs: $LOG_FILE"
echo ""
echo "Health check in 5s..."
sleep 5
curl -s http://localhost:8001/health && echo " — API is healthy!"
