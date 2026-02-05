#!/bin/bash

# Plagiarism Detection System - Installation Script
# This script installs all required dependencies for the plagiarism detection system

set -e  # Exit on any error

echo "=========================================="
echo "Plagiarism Detection System"
echo "Installation Script"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

# Install pip dependencies
echo ""
echo "[2/5] Installing Python packages..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install --upgrade pip
    python3 -m pip install -r requirements.txt
    echo "✓ Python packages installed successfully"
else
    echo "WARNING: requirements.txt not found"
    echo "Installing packages manually..."
    python3 -m pip install --upgrade pip
    python3 -m pip install sentence-transformers textstat spacy lexical_diversity flask
fi

# Download NLTK data
echo ""
echo "[3/5] Downloading NLTK data..."
python3 -c "
import nltk
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

packages = [
    'punkt', 
    'punkt_tb', 
    'stopwords', 
    'brown', 
    'cmudict', 
    'omw-1.4', 
    'wordnet', 
    'averaged_perceptron_tagger'
]

print('Downloading NLTK packages...')
for pkg in packages:
    try:
        nltk.download(pkg, quiet=True)
        print(f'  ✓ Downloaded {pkg}')
    except Exception as e:
        print(f'  ✗ Failed to download {pkg}: {e}')
"
echo "✓ NLTK data downloaded successfully"

# Download spaCy model
echo ""
echo "[4/5] Downloading spaCy language model..."
python3 -m spacy download en_core_web_sm --quiet
echo "✓ spaCy model downloaded successfully"

# Verify installation
echo ""
echo "[5/5] Verifying installation..."
python3 -c "
import sys
print('Checking installed packages...')

packages = {
    'flask': 'Flask',
    'sentence_transformers': 'sentence-transformers',
    'textstat': 'textstat',
    'spacy': 'spacy',
    'lexical_diversity': 'lexical_diversity',
    'torch': 'PyTorch (optional)',
    'nltk': 'NLTK'
}

success = True
for import_name, display_name in packages.items():
    try:
        __import__(import_name)
        print(f'  ✓ {display_name}')
    except ImportError:
        print(f'  ✗ {display_name} NOT FOUND')
        success = False

if success:
    print('\\n✓ All required packages are installed!')
else:
    print('\\n✗ Some packages are missing. Please install them manually.')
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start the Flask server:"
echo "   python app.py"
echo ""
echo "2. Open your browser to:"
echo "   http://127.0.0.1:5000"
echo ""
echo "3. Paste text and click 'Detect Plagiarism'"
echo ""
echo "For more information, see README.md"
echo "=========================================="
