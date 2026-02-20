# Lucio Master - High Performance RAG Pipeline

A blazing-fast RAG (Retrieval-Augmented Generation) system designed to ingest 200 PDFs and answer 15 complex questions in under 30 seconds.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   LUCIO MASTER ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐     ┌─────────────┐     ┌────────────┐         │
│  │   Rust     │     │   Python    │     │   Numpy    │         │
│  │  Ingestor  │ ──▶ │  Embedder   │ ──▶ │   Index    │         │
│  │(PyO3+Rayon)│     │(MiniLM-L6)  │     │  (Cosine)  │         │
│  └────────────┘     └─────────────┘     └────────────┘         │
│       ▲                   ▲                   │                  │
│       │                   │                   ▼                  │
│  ┌────┴────┐        ┌────┴────┐        ┌────────────┐          │
│  │  200    │        │  OCR    │        │   Groq     │          │
│  │  PDFs   │        │(Optional)│        │   LLM      │          │
│  └─────────┘        └─────────┘        │(llama-3.3) │          │
│                                         └────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## ⚡ Performance (Actual Benchmarks)

| Stage | Time | Description |
|-------|------|-------------|
| Parsing (Rust) | ~0.5s | Parallel PDF extraction (8 threads, 29+ PDFs/sec) |
| OCR | ~0.0s | Skipped for text PDFs, available for scanned |
| Embedding | ~23s | all-MiniLM-L6-v2 on CPU (faster on GPU) |
| Vector Search | ~0.02s | Numpy cosine similarity |
| Groq Synthesis | ~1-5s | Parallel LLM inference (depends on API) |
| **Total** | **~24-28s** | **Complete pipeline (CPU only)** |

*Tested with 16 PDFs → 1771 chunks on Apple M-series (CPU)*

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (tested on Python 3.14)
- Rust 1.70+ with `maturin` (for the native module)
- Groq API key (get one at https://console.groq.com)
- GPU optional (works on CPU)

### Installation

```bash
# Clone and enter directory
cd lucio_challenge

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Build Rust module (requires Rust + maturin)
pip install maturin
cd rust-core
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
cd ..
```

### Configuration

```bash
# Copy and edit environment file
cp .env.example .env

# Edit .env with your settings:
# GROQ_API_KEY=your_api_key_here
# LUCIO_PDF_DIR=./data/pdfs
```

### Running

```bash
# Activate virtual environment
source venv/bin/activate

# Place your PDFs in data/pdfs/ folder

# Run the pipeline
python python-engine/main.py
```

## 📁 Project Structure

```
lucio_challenge/
├── rust-core/              # Rust PDF processor (PyO3 + Rayon)
│   ├── Cargo.toml          # Rust dependencies (pyo3 0.23, rayon, lopdf)
│   └── src/
│       └── lib.rs          # Parallel PDF extraction with 8 threads
│
├── python-engine/          # Python ML pipeline
│   ├── pipeline.py         # Embeddings (MiniLM), vector search, OCR
│   └── main.py             # Async orchestrator + Groq synthesis
│
├── scripts/
│   ├── build.sh            # Build automation script
│   └── generate_test_pdfs.sh
│
├── tests/
│   └── benchmark.py        # Performance testing
│
├── data/
│   └── pdfs/               # Place your PDFs here
│
├── .env                    # Environment config (API keys)
├── .env.example            # Example environment file
├── requirements.txt        # Python dependencies
└── pyproject.toml          # Python project config
```

## 🔧 Configuration

### Environment Variables (in `.env` file)

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | (required) | Your Groq API key for LLM synthesis |
| `LUCIO_PDF_DIR` | `./data/pdfs` | Directory containing PDF files |
| `LUCIO_USE_GPU` | `true` | Enable GPU acceleration if available |
| `TORCH_DEVICE` | `cuda` | PyTorch device (cuda/cpu/mps) |

### Customizing Questions

Edit `QUESTIONS` list in `python-engine/main.py`:

```python
QUESTIONS = [
    "What are the revenue figures for Meta for Q1, Q2 and Q3?",
    "What was KFIN's revenue in 2021?",
    "What metrics helped CCI determine if the combination would be anticompetitive?",
    # Add more questions...
]
```

## 🧪 Example Output

```
============================================================
🚀 LUCIO SERVER - High Performance RAG Pipeline
============================================================
📁 Found 16 PDFs in ./data/pdfs
🦀 Using Rust ingestor with 8 threads
✅ Ingestion completed in 0.47s (1771 chunks)
✅ OCR completed in 0.00s
✅ Embedding completed in 23.25s
✅ Numpy index built in 0.00s
✅ Batch search completed in 0.02s
✅ Synthesis completed in 5.10s

📊 PERFORMANCE STATS
  Ingestion:    0.47s
  OCR:          0.00s
  Embedding:    23.25s
  Indexing:     0.00s
  Search:       0.02s
  Synthesis:    5.10s
  ─────────────────────
  TOTAL:        28.84s

🏆 SUCCESS! Completed under 30 seconds!
```

## 🔑 Key Technologies & Optimizations

### 1. Rust Parallel PDF Processing
- **PyO3 0.23** for Python-Rust interop
- **Rayon** for work-stealing parallelism (8 threads)
- **lopdf** for PDF parsing
- Processes ~30 PDFs/second

### 2. Sentence Embeddings
- **all-MiniLM-L6-v2** model (80MB, fast)
- Batch processing with transformers
- Works on CPU or GPU (CUDA/MPS)

### 3. Vector Search
- **Numpy cosine similarity** (fallback for ARM Macs)
- FAISS support available for x86 systems
- Sub-millisecond search latency

### 4. Groq LLM Synthesis
- **llama-3.3-70b-versatile** model
- `asyncio.gather()` for parallel API calls
- Exponential backoff retry with configurable timeout
- Context built from top-5 relevant chunks

## 🐛 Troubleshooting

### Rust module not found
```bash
# Ensure Rust and maturin are installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin

# Build the module
cd rust-core
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

### Python 3.14+ compatibility
The forward compatibility flag is required:
```bash
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
```

### FAISS crashes on ARM Mac
The system automatically falls back to numpy-based cosine similarity search.

### Groq API timeouts
- Default timeout is 5 seconds with 3 retries
- Increase timeout in `GroqSynthesizer` if needed
- Check your API rate limits at https://console.groq.com

### Slow embeddings
- Embeddings are CPU-bound (~23s for 1700 chunks)
- Use a GPU for 5-10x speedup
- Or use a smaller chunk count by increasing chunk size

## 📦 Dependencies

### Python (requirements.txt)
- `torch` - PyTorch for ML operations
- `transformers` - Hugging Face transformers
- `sentence-transformers` - Embedding models
- `faiss-cpu` - Vector search (optional)
- `groq` - Groq API client
- `PyPDF2` - Python PDF fallback
- `python-dotenv` - Environment variable loading

### Rust (Cargo.toml)
- `pyo3 0.23` - Python bindings
- `rayon 1.8` - Parallel processing
- `lopdf 0.31` - PDF parsing
- `serde 1.0` - Serialization

## 📄 License

MIT License - See LICENSE file for details.
