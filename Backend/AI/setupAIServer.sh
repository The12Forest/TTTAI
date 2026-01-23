#!/bin/bash
set -e

if ! command -v python3 >/dev/null; then
  echo "Python 3 is required"
  exit 1
fi

echo "Creating venv..."
python3 -m venv venv

echo "Activating venv..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

if [ -d "models" ]; then
  echo "Starting API..."
  python AI-Server.py
fi

if [ ! -d "models" ]; then
  echo "Model not fund!"
fi

