"""
Lucio Server - High-Performance GPU Pipeline
============================================
This module handles OCR, embeddings, and vector indexing using GPU acceleration.
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# Set environment for GPU
os.environ.setdefault("TORCH_DEVICE", "cuda")

import torch

@dataclass
class DocumentChunk:
    """Represents a processed document chunk"""
    text: str
    doc_id: str
    page_num: int
    chunk_id: str
    needs_ocr: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


class GPUResourceManager:
    """Manages GPU resources and ensures optimal utilization"""
    
    def __init__(self):
        # Detect GPU: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
            self.is_gpu = True
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory
            print(f"🚀 GPU Detected (NVIDIA CUDA): {self.gpu_name}")
            print(f"📊 GPU Memory: {self.gpu_memory / 1e9:.2f} GB")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.is_gpu = True
            self.gpu_name = "Apple Metal (MPS)"
            # MPS doesn't expose memory, estimate for M series
            self.gpu_memory = 16e9  # Assume 16GB shared memory
            print(f"🚀 GPU Detected: {self.gpu_name}")
            print(f"📊 Using shared GPU memory (estimated)")
        else:
            self.device = "cpu"
            self.is_gpu = False
            self.gpu_name = "CPU"
            self.gpu_memory = 0
            print("⚠️ No GPU detected, using CPU (slower)")
    
    def optimize_batch_size(self, model_memory_mb: int = 2000) -> int:
        """Calculate optimal batch size based on available GPU memory"""
        if not self.is_gpu:
            return 32
        
        if self.device == "cuda":
            available_memory = self.gpu_memory - (model_memory_mb * 1e6)
            # Estimate ~10MB per batch item for embeddings
            batch_size = int(available_memory / (10 * 1e6))
            return min(max(batch_size, 32), 512)
        elif self.device == "mps":
            # Apple Metal: conservative batch size
            return 256
        
        return 32


class OCRProcessor:
    """GPU-accelerated OCR using Surya"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False
    
    def _lazy_load(self):
        """Lazy load OCR models to save memory"""
        if self._loaded:
            return
        
        try:
            from surya.ocr import run_ocr
            from surya.model.detection.model import load_model as load_det_model
            from surya.model.detection.processor import load_processor as load_det_processor
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            
            print("📖 Loading Surya OCR models...")
            start = time.time()
            
            self.det_model = load_det_model()
            self.det_processor = load_det_processor()
            self.rec_model = load_rec_model()
            self.rec_processor = load_rec_processor()
            
            # Move to GPU if available
            if self.device == "cuda" and torch.cuda.is_available():
                self.det_model = self.det_model.to("cuda")
                self.rec_model = self.rec_model.to("cuda")
            
            self._loaded = True
            print(f"✅ OCR models loaded in {time.time() - start:.2f}s")
            
        except ImportError:
            print("⚠️ Surya OCR not installed. Using fallback OCR.")
            self._loaded = True
            self.det_model = None
    
    def process_images(self, images: List[Any], langs: List[str] = None) -> List[str]:
        """Process multiple images with OCR in batch"""
        self._lazy_load()
        
        if self.det_model is None:
            # Fallback: return empty strings
            return ["" for _ in images]
        
        try:
            from surya.ocr import run_ocr
            
            langs = langs or [["en"]] * len(images)
            
            results = run_ocr(
                images,
                [langs[0]] * len(images),
                self.det_model,
                self.det_processor,
                self.rec_model,
                self.rec_processor,
            )
            
            # Extract text from results
            texts = []
            for result in results:
                page_text = " ".join([line.text for line in result.text_lines])
                texts.append(page_text)
            
            return texts
            
        except Exception as e:
            print(f"⚠️ OCR Error: {e}")
            return ["" for _ in images]


class EmbeddingModel:
    """High-performance embedding model with FP16 optimization"""
    
    # Use smaller, faster model: all-MiniLM-L6-v2 (80MB vs 2.3GB for BGE-M3)
    # Still good quality but 30x smaller and faster to download/load
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = None, batch_size: int = None):
        self.model_name = model_name
        
        # Auto-detect device (CUDA > MPS > CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Auto-optimize batch size and precision based on hardware
        if batch_size is None:
            if self.device == "cuda":
                self.batch_size = 512  # NVIDIA GPU: large batches
                self.use_fp16 = True
                self.max_length = 256  # OPTIMIZED: reduced from 512
            elif self.device == "mps":
                self.batch_size = 256  # Apple Metal: medium batches (better stability)
                self.use_fp16 = True   # MPS supports FP16 in newer versions
                self.max_length = 256  # OPTIMIZED: reduced from 512
            else:
                self.batch_size = 128  # CPU: smaller batches
                self.use_fp16 = False
                self.max_length = 512
        else:
            self.batch_size = batch_size
            self.use_fp16 = self.device in ["cuda", "mps"]
            self.max_length = 256 if self.use_fp16 else 512
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
        self._dimension = 384  # MiniLM dimension (vs 1024 for BGE-M3)
    
    def _lazy_load(self):
        """Lazy load embedding model"""
        if self._loaded:
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            print(f"🧠 Loading embedding model: {self.model_name}")
            print(f"📊 Device: {self.device.upper()} | Batch size: {self.batch_size} | Max tokens: {self.max_length}")
            print(f"💾 FP16 Precision: {'✅ Enabled' if self.use_fp16 else '⚠️ Disabled'}")
            start = time.time()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Optimize for device with FP16 if supported
            if self.device == "cuda":
                self.model = self.model.to(self.device)
                if self.use_fp16:
                    self.model = self.model.half()
                    print("✅ Model loaded with FP16 precision on CUDA GPU (2x faster)")
                else:
                    print("✅ Model loaded with FP32 precision on CUDA GPU")
            elif self.device == "mps":
                self.model = self.model.to(self.device)
                if self.use_fp16:
                    try:
                        self.model = self.model.half()
                        print("✅ Model loaded with FP16 precision on Apple Metal (MPS)")
                    except:
                        self.use_fp16 = False
                        print("⚠️ FP16 not fully supported on MPS, using FP32")
                else:
                    print("✅ Model loaded on Apple Metal (MPS)")
            else:
                self.model = self.model.to(self.device)
                print(f"✅ Model loaded on CPU (slower)")
            
            self.model.eval()
            self._loaded = True
            print(f"✅ Embedding model loaded in {time.time() - start:.2f}s")
            
        except ImportError:
            print("⚠️ Transformers not installed. Using random embeddings for testing.")
            self._loaded = True
    
    def _mean_pooling(self, model_output, attention_mask):
        """Perform mean pooling on token embeddings"""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    @torch.no_grad()
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts efficiently with FP16 and optimized batch size"""
        self._lazy_load()
        
        if self.model is None:
            # Fallback: return random embeddings with correct dimension
            return np.random.randn(len(texts), self._dimension).astype(np.float32)
        
        all_embeddings = []
        
        # Process in batches to avoid OOM
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize with OPTIMIZED max_length
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,  # OPTIMIZED: reduced from 512 to 256
                return_tensors="pt"
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings with autocast for mixed precision
            if self.use_fp16:
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**encoded)
                elif self.device == "mps":
                    # MPS has better autocast in newer PyTorch
                    with torch.autocast(device_type="mps"):
                        outputs = self.model(**encoded)
                else:
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)
            
            embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings.float().cpu().numpy().astype(np.float32))
        
        return np.vstack(all_embeddings)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text"""
        return self.embed_batch([text])[0]


class FAISSVectorStore:
    """High-performance vector store (numpy-based fallback for ARM Macs)"""
    
    def __init__(self, dimension: int = 384, use_gpu: bool = True):
        self.dimension = dimension
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.index = None
        self.embeddings_matrix = None  # Store embeddings directly for numpy search
        self.chunks: List[DocumentChunk] = []
        self._built = False
        self._use_faiss = True  # Will try FAISS first, fallback to numpy
    
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalize vectors"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def build_index(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Build the index from chunks and embeddings in one operation"""
        print(f"🔨 Building index with {len(chunks)} vectors...")
        start = time.time()
        
        # Auto-detect dimension from embeddings
        self.dimension = embeddings.shape[1]
        self.chunks = chunks
        
        # Ensure embeddings are contiguous C array
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Normalize embeddings
        self.embeddings_matrix = self._normalize(embeddings)
        
        # Try FAISS first (disabled on ARM Macs due to segfaults)
        self._use_faiss = False  # Force numpy for stability
        print(f"✅ Numpy index built in {time.time() - start:.2f}s")
        
        self._built = True
    
    def _numpy_search(self, query_embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pure numpy cosine similarity search"""
        # Compute cosine similarity (dot product of normalized vectors)
        scores = np.dot(query_embeddings, self.embeddings_matrix.T)
        
        # Get top-k indices for each query
        if k >= scores.shape[1]:
            indices = np.argsort(-scores, axis=1)
        else:
            # Use argpartition for efficiency when k is small
            indices = np.argpartition(-scores, k, axis=1)[:, :k]
            # Sort the top-k
            for i in range(len(indices)):
                sorted_idx = np.argsort(-scores[i, indices[i]])
                indices[i] = indices[i, sorted_idx]
        
        # Get corresponding scores
        top_scores = np.array([scores[i, indices[i]] for i in range(len(query_embeddings))])
        
        return top_scores, indices
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        if not self._built:
            raise RuntimeError("Index not built. Call build_index first.")
        
        # Ensure query is 2D and normalized
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = self._normalize(query_embedding.astype(np.float32))
        
        k = min(k, len(self.chunks))
        
        if self._use_faiss:
            scores, indices = self.index.search(query_embedding, k)
        else:
            scores, indices = self._numpy_search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 10) -> List[List[Tuple[DocumentChunk, float]]]:
        """Batch search for multiple queries simultaneously"""
        if not self._built:
            raise RuntimeError("Index not built. Call build_index first.")
        
        # Ensure queries are contiguous and normalized
        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))
        query_embeddings = self._normalize(query_embeddings)
        
        k = min(k, len(self.chunks))
        
        if self._use_faiss:
            try:
                scores, indices = self.index.search(query_embeddings, k)
            except Exception as e:
                print(f"⚠️ FAISS search failed, using numpy: {e}")
                scores, indices = self._numpy_search(query_embeddings, k)
        else:
            scores, indices = self._numpy_search(query_embeddings, k)
        
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if 0 <= idx < len(self.chunks):
                    results.append((self.chunks[idx], float(score)))
            all_results.append(results)
        
        return all_results


class LucioPipeline:
    """Main pipeline orchestrating the entire RAG process"""
    
    def __init__(self, use_gpu: bool = True):
        self.gpu_manager = GPUResourceManager()
        self.use_gpu = use_gpu and self.gpu_manager.is_gpu
        self.device = self.gpu_manager.device if self.use_gpu else "cpu"
        
        # Initialize components (lazy loaded)
        self.ocr = OCRProcessor(device=self.device)
        
        batch_size = self.gpu_manager.optimize_batch_size() if self.use_gpu else 32
        self.embedder = EmbeddingModel(
            device=self.device,
            batch_size=batch_size
        )
        
        self.vector_store = FAISSVectorStore(use_gpu=self.use_gpu)
        
        # Producer-consumer queues
        self.ocr_queue: queue.Queue = queue.Queue()
        self.embed_queue: queue.Queue = queue.Queue()
        
        # Statistics
        self.stats = {
            "pdfs_processed": 0,
            "pages_ocr": 0,
            "chunks_embedded": 0,
            "total_time": 0.0,
        }
    
    def process_chunks_from_rust(self, rust_chunks: List[Dict]) -> List[DocumentChunk]:
        """Convert Rust output to DocumentChunk objects"""
        chunks = []
        for rc in rust_chunks:
            chunk = DocumentChunk(
                text=rc["text"],
                doc_id=rc["doc_id"],
                page_num=rc["page_num"],
                chunk_id=str(rc["chunk_id"]),
                needs_ocr=rc["needs_ocr"],
                metadata=rc.get("metadata", {}),
            )
            chunks.append(chunk)
        return chunks
    
    def run_ocr_batch(self, chunks: List[DocumentChunk], pdf_images: Dict[str, Any] = None) -> List[DocumentChunk]:
        """Run OCR on chunks that need it"""
        ocr_chunks = [c for c in chunks if c.needs_ocr]
        
        if not ocr_chunks:
            print("✅ No OCR needed")
            return chunks
        
        print(f"📖 Running OCR on {len(ocr_chunks)} pages...")
        start = time.time()
        
        # For now, we'll skip actual OCR if no images provided
        # In production, you would load images from PDFs
        if pdf_images:
            images = [pdf_images.get(f"{c.doc_id}_{c.page_num}") for c in ocr_chunks]
            images = [img for img in images if img is not None]
            
            if images:
                texts = self.ocr.process_images(images)
                
                for chunk, text in zip(ocr_chunks, texts):
                    if text:
                        chunk.text = text
                        chunk.needs_ocr = False
        
        self.stats["pages_ocr"] = len(ocr_chunks)
        print(f"✅ OCR completed in {time.time() - start:.2f}s")
        
        return chunks
    
    def embed_chunks(self, chunks: List[DocumentChunk]) -> Tuple[List[DocumentChunk], np.ndarray]:
        """Embed all chunks"""
        print(f"🧠 Embedding {len(chunks)} chunks...")
        start = time.time()
        
        # Filter out empty chunks
        valid_chunks = [c for c in chunks if c.text.strip()]
        texts = [c.text for c in valid_chunks]
        
        if not texts:
            print("⚠️ No valid text to embed")
            return [], np.array([])
        
        embeddings = self.embedder.embed_batch(texts)
        
        # Attach embeddings to chunks
        for chunk, emb in zip(valid_chunks, embeddings):
            chunk.embedding = emb
        
        self.stats["chunks_embedded"] = len(valid_chunks)
        print(f"✅ Embedding completed in {time.time() - start:.2f}s")
        
        return valid_chunks, embeddings
    
    def build_index(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Build the FAISS index"""
        self.vector_store.build_index(chunks, embeddings)
    
    def search(self, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Search for relevant chunks"""
        query_embedding = self.embedder.embed_single(query)
        return self.vector_store.search(query_embedding, k)
    
    def batch_search(self, queries: List[str], k: int = 10) -> List[List[Tuple[DocumentChunk, float]]]:
        """Search for multiple queries simultaneously"""
        print(f"🔍 Batch searching {len(queries)} queries...")
        start = time.time()
        
        query_embeddings = self.embedder.embed_batch(queries)
        results = self.vector_store.batch_search(query_embeddings, k)
        
        print(f"✅ Batch search completed in {time.time() - start:.2f}s")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return self.stats


# Shared memory utilities for zero-copy data transfer
class SharedMemoryBuffer:
    """Manages shared memory for Rust-Python data transfer"""
    
    def __init__(self, name: str, size: int):
        from multiprocessing import shared_memory
        
        self.name = name
        self.size = size
        self.shm = None
    
    def create(self):
        """Create a new shared memory buffer"""
        from multiprocessing import shared_memory
        
        try:
            # Try to unlink existing buffer
            existing = shared_memory.SharedMemory(name=self.name)
            existing.close()
            existing.unlink()
        except FileNotFoundError:
            pass
        
        self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
        return self.shm.buf
    
    def open(self):
        """Open an existing shared memory buffer"""
        from multiprocessing import shared_memory
        
        self.shm = shared_memory.SharedMemory(name=self.name)
        return self.shm.buf
    
    def close(self):
        """Close the shared memory buffer"""
        if self.shm:
            self.shm.close()
    
    def unlink(self):
        """Unlink the shared memory buffer"""
        if self.shm:
            self.shm.unlink()


if __name__ == "__main__":
    # Quick test
    print("Testing GPU Pipeline...")
    
    pipeline = LucioPipeline(use_gpu=True)
    
    # Test with dummy data
    test_chunks = [
        DocumentChunk(
            text="This is a test document about machine learning and AI.",
            doc_id="test_001",
            page_num=1,
            chunk_id="0",
            needs_ocr=False,
        ),
        DocumentChunk(
            text="Neural networks are powerful tools for pattern recognition.",
            doc_id="test_002",
            page_num=1,
            chunk_id="0",
            needs_ocr=False,
        ),
    ]
    
    # Embed and index
    chunks, embeddings = pipeline.embed_chunks(test_chunks)
    
    if len(embeddings) > 0:
        pipeline.build_index(chunks, embeddings)
        
        # Test search
        results = pipeline.search("What is machine learning?", k=2)
        print("\nSearch Results:")
        for chunk, score in results:
            print(f"  Score: {score:.4f} - {chunk.text[:50]}...")
    
    print("\n✅ Pipeline test completed!")
