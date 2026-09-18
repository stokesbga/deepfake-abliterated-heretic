#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Run this installer on your NVIDIA machine under Linux or WSL2." >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || { echo "Install the NVIDIA driver first." >&2; exit 1; }
command -v ffmpeg >/dev/null || { echo "Install FFmpeg (e.g. sudo apt install ffmpeg)." >&2; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe is required." >&2; exit 1; }
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -e '.[gpu]'
echo "Installed. Next: source .venv/bin/activate && heretic download"

