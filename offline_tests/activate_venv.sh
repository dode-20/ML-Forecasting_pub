#!/bin/bash
# Virtual environment activation script

echo "Activating virtual environment for offline_tests..."
source venv/bin/activate
echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
echo ""
echo "Installed packages:"
pip list 