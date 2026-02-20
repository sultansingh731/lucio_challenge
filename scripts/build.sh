#!/bin/bash
# Lucio Server - Build Script
# This script builds the Rust module and sets up the Python environment

set -e

echo "=========================================="
echo "🚀 Lucio Server Build Script"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for required tools
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        return 1
    else
        echo -e "${GREEN}✅ $1 found${NC}"
        return 0
    fi
}

echo ""
echo "📋 Checking prerequisites..."
check_tool "python3" || exit 1
check_tool "cargo" || { echo -e "${YELLOW}⚠️ Rust not installed. Install from https://rustup.rs${NC}"; exit 1; }
check_tool "pip" || exit 1

# Check for maturin
if ! pip show maturin &> /dev/null; then
    echo -e "${YELLOW}📦 Installing maturin...${NC}"
    pip install maturin
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo ""
echo "📂 Project directory: $PROJECT_DIR"

# Step 1: Build Rust module
echo ""
echo "=========================================="
echo "🦀 Building Rust Module"
echo "=========================================="

cd "$PROJECT_DIR/rust-core"

# Build with maturin in development mode
echo "Building with maturin..."
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Rust module built successfully${NC}"
else
    echo -e "${RED}❌ Rust build failed${NC}"
    exit 1
fi

# Step 2: Install Python dependencies
echo ""
echo "=========================================="
echo "🐍 Installing Python Dependencies"
echo "=========================================="

cd "$PROJECT_DIR"

pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠️ Some dependencies may have failed${NC}"
fi

# Step 3: Check GPU availability
echo ""
echo "=========================================="
echo "🎮 Checking GPU"
echo "=========================================="

python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU Available: {torch.cuda.get_device_name(0)}')
    print(f'   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
else:
    print('⚠️ No GPU detected - will use CPU (slower)')
"

# Step 4: Verify Rust module import
echo ""
echo "=========================================="
echo "🔗 Verifying Rust-Python Bridge"
echo "=========================================="

python3 -c "
try:
    import lucio_core
    info = lucio_core.get_system_info()
    print(f'✅ Rust module loaded successfully')
    print(f'   CPU threads: {info[\"num_cpus\"]}')
except ImportError as e:
    print(f'⚠️ Rust module not available: {e}')
    print('   Will use Python fallback')
"

echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Build Complete!${NC}"
echo "=========================================="
echo ""
echo "To run the server:"
echo "  cd $PROJECT_DIR/python-engine"
echo "  export GROQ_API_KEY=your_api_key"
echo "  python main.py"
echo ""
