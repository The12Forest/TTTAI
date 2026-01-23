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

if [ ! -d "model" ]; then
  echo "Creating TensorFlow model..."
  python save_model.py
fi

echo "Starting API..."
python app.py
