# Lucio Master - High Performance RAG Pipeline

**A production-grade document intelligence system that extracts, indexes, and answers questions from hundreds of PDFs in under 30 seconds.**

## 🎯 What is This?

Lucio is an **AI-powered document question-answering system** that:
- 📄 Ingests and processes large volumes of PDF documents (legal, financial, technical)
- 🔍 Builds a searchable knowledge base using semantic embeddings
- 💬 Answers complex questions by retrieving relevant context and synthesizing answers with LLMs
- ⚡ Completes the entire pipeline in **under 30 seconds** (target performance)

**Perfect for:** Legal document analysis, compliance reviews, financial DD, research paper analysis, or any use case requiring rapid document Q&A at scale.

## 🧠 Business Logic & How It Works

```
└─▶ PDF Documents (./data/pdfs/)
    │
    ├─▶ [1] INGEST - Rust parallel processor extracts text from all PDFs
    │       ↓ Output: Raw text chunks (by page)
    │
    ├─▶ [2] EMBED - Python ML pipeline creates semantic vectors
    │       ↓ Uses: sentence-transformers/all-MiniLM-L6-v2
    │       ↓ Output: 384-dim embeddings for each chunk
    │
    ├─▶ [3] INDEX - Builds searchable vector database
    │       ↓ Uses: Numpy cosine similarity (FAISS optional)
    │       ↓ Output: Fast similarity search index
    │
    ├─▶ [4] QUERY - User asks questions
    │       ↓ Each question is embedded and searched
    │       ↓ Top 8 most relevant chunks retrieved
    │
    └─▶ [5] SYNTHESIZE - LLM generates answers from context
            ↓ Uses: Groq API (llama-3.1-70b-versatile)
            ↓ Output: Cited answers with document references
```

**Key Innovation:** Parallel processing at every stage - Rust for I/O, GPU/MPS for ML, async for API calls.

**Key Innovation:** Parallel processing at every stage - Rust for I/O, GPU/MPS for ML, async for API calls.

## 🚀 Quick Start (3 Steps)

### 1️⃣ Setup Environment

```bash
# Clone and navigate
cd lucio_challenge

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  

# Install dependencies
pip install -r requirements.txt

# Build Rust module (one-time)
cd rust-core
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
cd ..
```

### 2️⃣ Configure API Key

```bash
# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
echo "LUCIO_PDF_DIR=./data/pdfs" >> .env
```

Get your free Groq API key: https://console.groq.com (upgrade to Dev Tier recommended)

### 3️⃣ Run the Pipeline

```bash
# Add your PDFs to data/pdfs/ folder
mkdir -p data/pdfs
# (copy your PDF files here)

# Run the complete pipeline
source venv/bin/activate
python3 python-engine/main.py
```

**That's it!** The system will:
- Extract text from all PDFs
- Build semantic search index
- Answer the preset questions (customizable in main.py)
- Display results with timing stats

## ⚡ Performance Benchmarks

**Tested on:** Apple M-series Mac (MPS GPU), 17 PDFs, 1819 chunks, 7 questions

| Stage | Time | Optimization |
|-------|------|-------------|
| **Ingestion** | 0.40s | Rust parallel processing (8 threads) |
| **OCR** | 0.00s | Skipped for text PDFs |
| **Embedding** | 8.95s | FP16 precision, batch size 256, MPS GPU |
| **Indexing** | 0.00s | In-memory numpy index |
| **Search** | 0.06s | Batch cosine similarity (7 queries) |
| **Synthesis** | 14.22s | Parallel async Groq API calls |
| **TOTAL** | **23.63s** | ✅ **Under 30s target!** |

## 🎛️ Customization

### Change Questions

Edit `python-engine/main.py` around line 415:

```python
QUESTIONS = [
    "What are the revenue figures for Meta?",
    "What was KFIN's revenue in 2021?", 
    # Add your questions here
]
```

### Change Model

Edit `python-engine/main.py` around line 50:

```python
model: str = "llama-3.1-70b-versatile",  # Or: "llama-3.1-8b-instant" for faster
```

### Change PDF Directory

Edit `.env` file:

```bash
LUCIO_PDF_DIR=/path/to/your/pdfs  # Absolute or relative path
```

## � Project Structure

```
lucio_challenge/
├── python-engine/           # Main application
│   ├── main.py             # Pipeline orchestrator + Groq synthesis
│   └── pipeline.py         # Embeddings, vector search, OCR
│
├── rust-core/              # High-performance PDF processor
│   ├── Cargo.toml          # Rust dependencies
│   └── src/lib.rs          # PyO3 bindings + parallel PDF parsing
│
├── data/pdfs/              # 📄 Place your PDF files here
│
├── .env                    # 🔑 Configuration (API keys)
├── requirements.txt        # Python dependencies
└── README.md              # You are here
```

## 🔧 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Ingestion** | Rust + PyO3 + Rayon | Parallel PDF parsing (~30 PDFs/sec) |
| **Embeddings** | sentence-transformers | Semantic vector generation |
| **GPU** | PyTorch MPS/CUDA | Hardware acceleration |
| **Search** | Numpy/FAISS | Fast cosine similarity |
| **LLM** | Groq API (Llama 3.1) | Answer synthesis |
| **Async** | Python asyncio | Parallel API calls |

## 🐛 Common Issues & Solutions

### ❌ "Rust module not found"
```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build the module
cd rust-core
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop --release
cd ..
```

### ❌ "Groq API rate limit exceeded"
- Free tier: 100K tokens/day (allows ~2 runs)
- **Solution:** Upgrade to Dev Tier ($7/month) for 500K tokens/day
- Get it at: https://console.groq.com/settings/billing

### ❌ "Exceeded 30s target"
- **Bottleneck:** Usually Groq API synthesis (~14s)
- **Solutions:**
  - Use faster model: `llama-3.1-8b-instant`
  - Reduce chunks passed to LLM (change `[:8]` to `[:5]` in main.py line 145)
  - Upgrade to Dev Tier for better API throughput

### ❌ "No GPU detected"
- System falls back to CPU automatically
- For Apple Silicon: ensure PyTorch with MPS support is installed
- For NVIDIA: install `torch` with CUDA support

## 📊 Example Output

```bash
$ python3 python-engine/main.py

============================================================
🚀 LUCIO SERVER - High Performance RAG Pipeline
============================================================
📁 Found 17 PDFs in ./data/pdfs
🚀 GPU Detected: Apple Metal (MPS)

📚 Ingesting 17 PDFs...
🦀 Using Rust ingestor with 8 threads
✅ Ingestion completed in 0.40s (1819 chunks)
📊 Filtered to 612 high-quality chunks

🧠 Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
✅ Model loaded with FP16 precision on Apple Metal (MPS)
✅ Embedding completed in 8.95s

🔨 Building index with 612 vectors...
✅ Numpy index built in 0.00s

🔍 Batch searching 7 queries...
✅ Batch search completed in 0.06s

🤖 Starting parallel synthesis for 7 questions...
✅ Synthesis completed in 14.22s

============================================================
📋 RESULTS
============================================================

❓ Q1: What metrics helped CCI determine if the combination would be anticompetitive?
💡 A: According to [doc_0001, Page 44], the metrics used include the 
      Herfindahl-Hirschman Index (HHI)...
⏱️ Latency: 1981ms | Tokens: 7360

[... more results ...]

============================================================
📊 PERFORMANCE STATS
============================================================
  Ingestion:    0.40s
  OCR:          0.00s
  Embedding:    8.95s
  Indexing:     0.00s
  Search:       0.06s
  Synthesis:    14.22s
  ─────────────────────
  TOTAL:        23.63s

🏆 SUCCESS! Completed under 30 seconds!
```

## 🔑 Key Features

✅ **Blazing Fast:** Sub-30s for complete document intelligence pipeline  
✅ **Parallel Processing:** Rust threads + GPU + async API calls  
✅ **Production Ready:** Error handling, retries, fallbacks  
✅ **GPU Accelerated:** Supports NVIDIA CUDA, Apple MPS, and CPU  
✅ **Scalable:** Handles hundreds of documents with ease  
✅ **Accurate:** High-quality embeddings + citation-backed answers  

## 📄 License

MIT License

---

**Built for speed. Optimized for scale. Ready for production.** 🚀
