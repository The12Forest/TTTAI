#!/bin/bash

if command -v node &> /dev/null
then
    # Also check npm
    if command -v npm &> /dev/null
    then
    else
        echo "✗ npm is NOT installed"
        echo "Please install npm"
        exit 1
    fi
else
    echo "✗ Node.js is NOT installed"
    echo "Please install node"
    exit 1
fi

if command -v python3 &> /dev/null
then
else
    echo "✗ Pyton3 is NOT installed"
    echo "Please install pyton3"
    exit 1
fi

npm install

cd ./Backen/routes/ai
python3 -m venv .venv
source /venv/bin/activate
pip install fastapi numpy tensorflow

echo "If you dont see any errors its all good."
echo "Else ask ChatGPT to assist you."