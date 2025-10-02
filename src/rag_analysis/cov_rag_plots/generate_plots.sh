#!/bin/bash
# Quick start script for Coverage Gap Analysis plots
# Usage: ./generate_plots.sh

set -e

echo "=================================="
echo "Coverage Gap Analysis Plot Generator"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found at .venv/"
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if matplotlib and numpy are installed
echo "🔍 Checking dependencies..."
python -c "import matplotlib, numpy" 2>/dev/null || {
    echo "📦 Installing matplotlib and numpy..."
    pip install matplotlib numpy
}

# Run all plots
echo ""
echo "📊 Generating all plots..."
echo ""
python src/rag_analysis/cov_rag_plots/run_all_plots.py

echo ""
echo "✅ Done! Check the plots in:"
echo "   src/rag_analysis/cov_rag_plots/"
echo ""
echo "📄 Key findings documented in:"
echo "   src/rag_analysis/cov_rag_plots/KEY_FINDINGS.md"
