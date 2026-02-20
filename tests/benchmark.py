"""
Lucio Server - Benchmark Test
=============================
Tests the full pipeline performance against the 30-second target.
"""

import os
import sys
import asyncio
import time
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent / "python-engine"))

from pipeline import LucioPipeline, DocumentChunk


def create_test_chunks(num_docs: int = 200, pages_per_doc: int = 5) -> list:
    """Create test document chunks"""
    chunks = []
    
    sample_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised.",
        "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language.",
        "Computer vision is an interdisciplinary scientific field that deals with how computers can gain high-level understanding from digital images or videos.",
        "Reinforcement learning is an area of machine learning concerned with how intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward.",
    ]
    
    for doc_idx in range(num_docs):
        for page_idx in range(pages_per_doc):
            text = sample_texts[(doc_idx + page_idx) % len(sample_texts)]
            chunks.append(DocumentChunk(
                text=text,
                doc_id=f"doc_{doc_idx:04d}",
                page_num=page_idx + 1,
                chunk_id=f"{page_idx}",
                needs_ocr=False,
                metadata={"source": f"test_doc_{doc_idx}.pdf"},
            ))
    
    return chunks


async def run_benchmark():
    """Run the full benchmark"""
    print("=" * 60)
    print("🏁 LUCIO SERVER BENCHMARK")
    print("=" * 60)
    
    # Configuration
    NUM_DOCS = 200
    PAGES_PER_DOC = 5
    NUM_QUESTIONS = 15
    
    total_start = time.time()
    timings = {}
    
    # Step 1: Create test data
    print("\n📊 Step 1: Creating test data...")
    start = time.time()
    chunks = create_test_chunks(NUM_DOCS, PAGES_PER_DOC)
    timings["data_creation"] = time.time() - start
    print(f"   Created {len(chunks)} chunks in {timings['data_creation']:.2f}s")
    
    # Step 2: Initialize pipeline
    print("\n🔧 Step 2: Initializing pipeline...")
    start = time.time()
    pipeline = LucioPipeline(use_gpu=True)
    timings["init"] = time.time() - start
    print(f"   Pipeline initialized in {timings['init']:.2f}s")
    
    # Step 3: Embed chunks
    print("\n🧠 Step 3: Embedding chunks...")
    start = time.time()
    processed_chunks, embeddings = pipeline.embed_chunks(chunks)
    timings["embedding"] = time.time() - start
    print(f"   Embedded {len(processed_chunks)} chunks in {timings['embedding']:.2f}s")
    print(f"   Throughput: {len(processed_chunks) / timings['embedding']:.0f} chunks/sec")
    
    # Step 4: Build index
    print("\n🔨 Step 4: Building FAISS index...")
    start = time.time()
    if len(embeddings) > 0:
        pipeline.build_index(processed_chunks, embeddings)
    timings["indexing"] = time.time() - start
    print(f"   Index built in {timings['indexing']:.2f}s")
    
    # Step 5: Batch search
    print("\n🔍 Step 5: Batch search...")
    questions = [
        f"Question {i}: What is the main topic discussed?"
        for i in range(NUM_QUESTIONS)
    ]
    
    start = time.time()
    results = pipeline.batch_search(questions, k=10)
    timings["search"] = time.time() - start
    print(f"   Searched {len(questions)} queries in {timings['search']:.2f}s")
    print(f"   Throughput: {len(questions) / timings['search']:.0f} queries/sec")
    
    # Calculate totals
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "=" * 60)
    print("📈 BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Data Creation:  {timings['data_creation']:>6.2f}s")
    print(f"  Initialization: {timings['init']:>6.2f}s")
    print(f"  Embedding:      {timings['embedding']:>6.2f}s")
    print(f"  Indexing:       {timings['indexing']:>6.2f}s")
    print(f"  Search:         {timings['search']:>6.2f}s")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:          {total_time:>6.2f}s")
    print()
    
    # Target breakdown
    print("📊 TARGET vs ACTUAL:")
    targets = {
        "embedding": 8.0,
        "indexing": 0.5,
        "search": 0.1,
    }
    
    for key, target in targets.items():
        actual = timings.get(key, 0)
        status = "✅" if actual <= target else "⚠️"
        print(f"  {key.capitalize():15} Target: {target:>5.1f}s | Actual: {actual:>5.2f}s {status}")
    
    print()
    
    if total_time < 30:
        print(f"🏆 SUCCESS! Completed in {total_time:.2f}s (under 30s target)")
        print(f"   Buffer remaining: {30 - total_time:.2f}s")
    else:
        print(f"⚠️ EXCEEDED target by {total_time - 30:.2f}s")
    
    return timings


if __name__ == "__main__":
    asyncio.run(run_benchmark())
