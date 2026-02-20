"""
Lucio Server - Main Entry Point
================================
Async main loop with producer-consumer pattern and parallel Groq synthesis.
"""

import os
import sys
import asyncio
import time
import glob
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import LucioPipeline, DocumentChunk

# Groq client with retry logic
import groq


@dataclass
class QueryResult:
    """Result from a single query"""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: float
    tokens_used: int


class GroqSynthesizer:
    """
    Handles parallel synthesis using Groq API with:
    - Exponential backoff retry
    - 5-second timeout
    - Parallel execution via asyncio.gather()
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",  # Updated model (llama3-70b-8192 was decommissioned)
        max_retries: int = 3,
        base_timeout: float = 5.0,
    ):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            print("⚠️ GROQ_API_KEY not set. Synthesis will use mock responses.")
        
        self.model = model
        self.max_retries = max_retries
        self.base_timeout = base_timeout
        self.client = None
        
        if self.api_key:
            self.client = groq.AsyncGroq(api_key=self.api_key)
    
    async def _call_with_retry(
        self,
        question: str,
        context: str,
    ) -> Tuple[str, int]:
        """Call Groq API with exponential backoff retry"""
        
        if not self.client:
            # Mock response for testing
            await asyncio.sleep(0.1)
            return f"Mock answer for: {question}", 0
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Be concise and accurate. If the context doesn't contain enough information, say so.
Always cite which document(s) your answer is based on."""
        
        user_prompt = f"""Context:
{context}

Question: {question}

Answer based on the context above:"""
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Calculate timeout with exponential backoff
                timeout = self.base_timeout * (2 ** attempt)
                
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.3,
                        max_tokens=500,
                    ),
                    timeout=timeout,
                )
                
                answer = response.choices[0].message.content
                tokens = response.usage.total_tokens if response.usage else 0
                
                return answer, tokens
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout}s"
                print(f"⚠️ Groq API timeout (attempt {attempt + 1}/{self.max_retries})")
                
            except Exception as e:
                last_error = str(e)
                print(f"⚠️ Groq API error (attempt {attempt + 1}/{self.max_retries}): {e}")
                
                # Exponential backoff wait
                if attempt < self.max_retries - 1:
                    wait_time = (2 ** attempt) * 0.5
                    await asyncio.sleep(wait_time)
        
        # All retries failed
        return f"Error: {last_error}", 0
    
    async def synthesize_single(
        self,
        question: str,
        relevant_chunks: List[Tuple[DocumentChunk, float]],
    ) -> QueryResult:
        """Synthesize answer for a single question"""
        start = time.time()
        
        # Build context from relevant chunks
        context_parts = []
        sources = []
        
        for chunk, score in relevant_chunks[:5]:  # Top 5 chunks
            context_parts.append(f"[{chunk.doc_id}, Page {chunk.page_num}]: {chunk.text}")
            sources.append({
                "doc_id": chunk.doc_id,
                "page_num": chunk.page_num,
                "score": score,
                "snippet": chunk.text[:200],
            })
        
        context = "\n\n".join(context_parts)
        
        # Get answer from Groq
        answer, tokens = await self._call_with_retry(question, context)
        
        latency = (time.time() - start) * 1000
        
        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            latency_ms=latency,
            tokens_used=tokens,
        )
    
    async def synthesize_parallel(
        self,
        questions: List[str],
        search_results: List[List[Tuple[DocumentChunk, float]]],
    ) -> List[QueryResult]:
        """
        Synthesize answers for all questions in parallel using asyncio.gather()
        """
        print(f"🤖 Starting parallel synthesis for {len(questions)} questions...")
        start = time.time()
        
        # Create coroutines for all questions
        tasks = [
            self.synthesize_single(q, results)
            for q, results in zip(questions, search_results)
        ]
        
        # Execute all in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(QueryResult(
                    question=questions[i],
                    answer=f"Error: {str(result)}",
                    sources=[],
                    latency_ms=0,
                    tokens_used=0,
                ))
            else:
                final_results.append(result)
        
        total_time = time.time() - start
        print(f"✅ Synthesis completed in {total_time:.2f}s")
        
        return final_results


class LucioServer:
    """
    Main orchestrator implementing producer-consumer pattern:
    1. Rust ingestor (producer) -> PDF chunks
    2. GPU pipeline (consumer/producer) -> Embeddings
    3. FAISS indexer (consumer/producer) -> Searchable index
    4. Query worker (consumer) -> Answers
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        use_gpu: bool = True,
    ):
        self.pipeline = LucioPipeline(use_gpu=use_gpu)
        self.synthesizer = GroqSynthesizer(api_key=groq_api_key)
        
        # State
        self.chunks: List[DocumentChunk] = []
        self.is_indexed = False
        
        # Stats
        self.stats = {
            "ingest_time": 0.0,
            "ocr_time": 0.0,
            "embed_time": 0.0,
            "index_time": 0.0,
            "search_time": 0.0,
            "synthesis_time": 0.0,
            "total_time": 0.0,
        }
    
    def ingest_pdfs(self, pdf_paths: List[str]) -> List[DocumentChunk]:
        """
        Ingest PDFs using Rust parallel processor
        """
        print(f"\n📚 Ingesting {len(pdf_paths)} PDFs...")
        start = time.time()
        
        try:
            # Try to import the Rust module
            import lucio_core
            
            print(f"🦀 Using Rust ingestor with {lucio_core.get_system_info()['num_cpus']} threads")
            
            # Process PDFs in parallel using Rust
            rust_chunks = lucio_core.process_pdfs_parallel(pdf_paths)
            
            # Convert to Python objects
            self.chunks = self.pipeline.process_chunks_from_rust(rust_chunks)
            
        except ImportError:
            print("⚠️ Rust module not available. Using Python fallback.")
            self.chunks = self._python_fallback_ingest(pdf_paths)
        
        self.stats["ingest_time"] = time.time() - start
        print(f"✅ Ingestion completed in {self.stats['ingest_time']:.2f}s ({len(self.chunks)} chunks)")
        
        return self.chunks
    
    def _python_fallback_ingest(self, pdf_paths: List[str]) -> List[DocumentChunk]:
        """Python fallback for PDF ingestion with parallel processing"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import PyPDF2
        
        all_chunks = []
        
        def process_single_pdf(args):
            idx, path = args
            chunks = []
            try:
                with open(path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text() or ""
                        chunks.append(DocumentChunk(
                            text=text,
                            doc_id=f"doc_{idx:04d}",
                            page_num=page_num + 1,
                            chunk_id=str(page_num),
                            needs_ocr=len(text) < 10,
                            metadata={"source_file": path},
                        ))
            except Exception as e:
                print(f"⚠️ Error processing {os.path.basename(path)}: {e}")
            return chunks
        
        # Process PDFs in parallel using thread pool
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_single_pdf, (idx, path)): idx 
                      for idx, path in enumerate(pdf_paths)}
            
            completed = 0
            for future in as_completed(futures):
                chunks = future.result()
                all_chunks.extend(chunks)
                completed += 1
                if completed % 5 == 0 or completed == len(pdf_paths):
                    print(f"  Processed {completed}/{len(pdf_paths)} PDFs...")
        
        return all_chunks
    
    def process_and_index(self):
        """
        Run OCR (if needed), embed chunks, and build index
        """
        if not self.chunks:
            raise RuntimeError("No chunks to process. Call ingest_pdfs first.")
        
        # OCR pass
        start = time.time()
        self.chunks = self.pipeline.run_ocr_batch(self.chunks)
        self.stats["ocr_time"] = time.time() - start
        
        # Embedding pass
        start = time.time()
        self.chunks, embeddings = self.pipeline.embed_chunks(self.chunks)
        self.stats["embed_time"] = time.time() - start
        
        # Build index
        if len(embeddings) > 0:
            start = time.time()
            self.pipeline.build_index(self.chunks, embeddings)
            self.stats["index_time"] = time.time() - start
            self.is_indexed = True
        else:
            print("⚠️ No embeddings to index")
    
    async def answer_questions(
        self,
        questions: List[str],
        k: int = 10,
    ) -> List[QueryResult]:
        """
        Answer multiple questions in parallel
        """
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call process_and_index first.")
        
        # Batch search for all questions
        start = time.time()
        search_results = self.pipeline.batch_search(questions, k=k)
        self.stats["search_time"] = time.time() - start
        
        # Parallel synthesis
        start = time.time()
        results = await self.synthesizer.synthesize_parallel(questions, search_results)
        self.stats["synthesis_time"] = time.time() - start
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get timing statistics"""
        self.stats["total_time"] = sum([
            self.stats["ingest_time"],
            self.stats["ocr_time"],
            self.stats["embed_time"],
            self.stats["index_time"],
            self.stats["search_time"],
            self.stats["synthesis_time"],
        ])
        return self.stats


async def main():
    """
    Main entry point - runs the complete pipeline
    """
    print("=" * 60)
    print("🚀 LUCIO SERVER - High Performance RAG Pipeline")
    print("=" * 60)
    
    total_start = time.time()
    
    # Configuration - resolve path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_pdf_dir = os.path.join(project_root, "data", "pdfs")
    PDF_DIR = os.environ.get("LUCIO_PDF_DIR", default_pdf_dir)
    
    # Expand user path (for ~/Downloads etc)
    PDF_DIR = os.path.expanduser(PDF_DIR)
    
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    USE_GPU = os.environ.get("LUCIO_USE_GPU", "true").lower() == "true"
    
    # Sample questions (replace with actual questions)
    QUESTIONS = [
        "What are the revenue figures for Meta for Q1, Q2 and Q3?",
        "What was KFIN's revenue in 2021?",
        "What metrics helped CCI determine if the combination would be anticompetitive?",
        "What was the bench in the Eastman Kodak Case?",
        "How many SCOTUS cases are in the set? Name them.",
        "What is the governing law in the NVCA IRA?",
        "If Pristine were to acquire an indian company that had turnover of 1Cr and no assets, would it have to notify the deal to the CCI?",
    ]
    
    # Find all PDFs (both .pdf and .PDF extensions)
    pdf_paths = glob.glob(os.path.join(PDF_DIR, "**/*.pdf"), recursive=True)
    pdf_paths += glob.glob(os.path.join(PDF_DIR, "**/*.PDF"), recursive=True)
    
    if not pdf_paths:
        print(f"⚠️ No PDFs found in {PDF_DIR}")
        print("Creating dummy PDFs for testing...")
        
        # Create test directory and dummy data
        os.makedirs(PDF_DIR, exist_ok=True)
        
        # Create dummy test chunks directly
        server = LucioServer(groq_api_key=GROQ_API_KEY, use_gpu=USE_GPU)
        
        # Add some test data
        server.chunks = [
            DocumentChunk(
                text=f"This is test document {i}. It contains information about machine learning, "
                     f"neural networks, and artificial intelligence. The main finding is that "
                     f"deep learning models can achieve high accuracy on various tasks.",
                doc_id=f"doc_{i:04d}",
                page_num=1,
                chunk_id="0",
                needs_ocr=False,
                metadata={"source_file": f"test_{i}.pdf"},
            )
            for i in range(200)
        ]
        
        print(f"✅ Created {len(server.chunks)} test chunks")
        
    else:
        print(f"📁 Found {len(pdf_paths)} PDFs in {PDF_DIR}")
        
        # Initialize server
        server = LucioServer(groq_api_key=GROQ_API_KEY, use_gpu=USE_GPU)
        
        # Stage 1: Ingest PDFs (uses Rust parallel processing)
        server.ingest_pdfs(pdf_paths)
    
    # Stage 2: Process and index (OCR, embed, index)
    server.process_and_index()
    
    # Stage 3: Answer questions (parallel search + synthesis)
    results = await server.answer_questions(QUESTIONS)
    
    # Print results
    print("\n" + "=" * 60)
    print("📋 RESULTS")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n❓ Q{i}: {result.question}")
        print(f"💡 A: {result.answer[:200]}..." if len(result.answer) > 200 else f"💡 A: {result.answer}")
        print(f"⏱️ Latency: {result.latency_ms:.0f}ms | Tokens: {result.tokens_used}")
    
    # Print stats
    stats = server.get_stats()
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE STATS")
    print("=" * 60)
    print(f"  Ingestion:    {stats['ingest_time']:.2f}s")
    print(f"  OCR:          {stats['ocr_time']:.2f}s")
    print(f"  Embedding:    {stats['embed_time']:.2f}s")
    print(f"  Indexing:     {stats['index_time']:.2f}s")
    print(f"  Search:       {stats['search_time']:.2f}s")
    print(f"  Synthesis:    {stats['synthesis_time']:.2f}s")
    print(f"  ─────────────────────")
    print(f"  TOTAL:        {stats['total_time']:.2f}s")
    
    total_elapsed = time.time() - total_start
    print(f"\n⏱️ Wall clock time: {total_elapsed:.2f}s")
    
    if total_elapsed < 30:
        print("🏆 SUCCESS! Completed under 30 seconds!")
    else:
        print(f"⚠️ Exceeded 30s target by {total_elapsed - 30:.2f}s")
    
    return results, stats


if __name__ == "__main__":
    asyncio.run(main())
