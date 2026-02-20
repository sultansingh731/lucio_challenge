use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use lopdf::Document;
use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Represents a chunk of text extracted from a PDF document
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub text: String,
    pub doc_id: String,
    pub page_num: usize,
    pub chunk_id: usize,
    pub needs_ocr: bool,
    pub metadata: HashMap<String, String>,
}

/// Extract text from PDF content stream operators
fn extract_text_operators(content: &str) -> String {
    let mut result = String::new();
    let mut chars = content.chars().peekable();
    
    while let Some(c) = chars.next() {
        if c == '(' {
            // Found start of string literal
            let mut depth = 1;
            let mut text = String::new();
            
            while let Some(&next) = chars.peek() {
                chars.next();
                if next == '(' {
                    depth += 1;
                    text.push(next);
                } else if next == ')' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                    text.push(next);
                } else if next == '\\' {
                    // Handle escape sequences
                    if let Some(&escaped) = chars.peek() {
                        chars.next();
                        match escaped {
                            'n' => text.push('\n'),
                            'r' => text.push('\r'),
                            't' => text.push('\t'),
                            _ => text.push(escaped),
                        }
                    }
                } else {
                    text.push(next);
                }
            }
            
            result.push_str(&text);
            result.push(' ');
        }
    }
    
    result
}

/// Extract text content from a PDF document
fn extract_document_text(doc: &Document) -> Vec<(usize, String)> {
    let mut pages_text = Vec::new();
    let page_ids: Vec<_> = doc.get_pages().into_iter().collect();
    
    for (page_num, (_, object_id)) in page_ids.iter().enumerate() {
        let mut page_text = String::new();
        
        // Try to get page content
        if let Ok(content) = doc.get_page_content(*object_id) {
            if let Ok(text) = String::from_utf8(content) {
                page_text = extract_text_operators(&text);
            }
        }
        
        pages_text.push((page_num + 1, page_text));
    }
    
    pages_text
}

/// Process a single PDF file and extract all text chunks
fn process_single_pdf(path: &str, doc_id: &str) -> Vec<DocumentChunk> {
    let mut chunks = Vec::new();
    
    // Read PDF file
    let pdf_bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Failed to read PDF {}: {}", path, e);
            return chunks;
        }
    };
    
    // Parse PDF document
    let doc = match Document::load_mem(&pdf_bytes) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to parse PDF {}: {}", path, e);
            return chunks;
        }
    };
    
    // Get page count
    let page_count = doc.get_pages().len();
    
    // Extract metadata
    let mut metadata = HashMap::new();
    metadata.insert("source_file".to_string(), path.to_string());
    metadata.insert("page_count".to_string(), page_count.to_string());
    
    // Extract text from all pages
    let pages_text = extract_document_text(&doc);
    
    for (page_num, text) in pages_text {
        // Check if OCR is needed (less than 10 chars means likely scanned)
        let needs_ocr = text.len() < 10;
        
        // Create chunk for this page
        let chunk = DocumentChunk {
            text,
            doc_id: doc_id.to_string(),
            page_num,
            chunk_id: page_num - 1,
            needs_ocr,
            metadata: metadata.clone(),
        };
        
        chunks.push(chunk);
    }
    
    chunks
}

/// Split text into smaller chunks for embedding
fn split_into_chunks(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![text.to_string()];
    }
    
    let words: Vec<&str> = text.split_whitespace().collect();
    
    if words.len() <= chunk_size {
        return vec![text.to_string()];
    }
    
    let mut chunks = Vec::new();
    let mut start = 0;
    
    while start < words.len() {
        let end = std::cmp::min(start + chunk_size, words.len());
        let chunk: String = words[start..end].join(" ");
        chunks.push(chunk);
        
        start += chunk_size - overlap;
        if start + overlap >= words.len() {
            break;
        }
    }
    
    chunks
}

/// Python-exposed function: Process multiple PDFs in parallel
#[pyfunction]
fn process_pdfs_parallel(py: Python<'_>, paths: Vec<String>) -> PyResult<Vec<PyObject>> {
    let counter = AtomicUsize::new(0);
    let total = paths.len();
    
    // Process all PDFs in parallel using rayon
    let all_chunks: Vec<DocumentChunk> = paths
        .par_iter()
        .enumerate()
        .flat_map(|(idx, path)| {
            let doc_id = format!("doc_{:04}", idx);
            let chunks = process_single_pdf(path, &doc_id);
            
            let processed = counter.fetch_add(1, Ordering::SeqCst) + 1;
            if processed % 50 == 0 || processed == total {
                eprintln!("Processed {}/{} PDFs", processed, total);
            }
            
            chunks
        })
        .collect();
    
    // Convert to Python objects
    let mut result = Vec::new();
    
    for chunk in all_chunks {
        let dict = PyDict::new(py);
        dict.set_item("text", &chunk.text)?;
        dict.set_item("doc_id", &chunk.doc_id)?;
        dict.set_item("page_num", chunk.page_num)?;
        dict.set_item("chunk_id", chunk.chunk_id)?;
        dict.set_item("needs_ocr", chunk.needs_ocr)?;
        
        // Convert metadata to Python dict
        let meta_dict = PyDict::new(py);
        for (k, v) in &chunk.metadata {
            meta_dict.set_item(k, v)?;
        }
        dict.set_item("metadata", meta_dict)?;
        
        result.push(dict.into_pyobject(py)?.into_any().unbind());
    }
    
    Ok(result)
}

/// Python-exposed function: Process PDFs and return chunked text for embeddings
#[pyfunction]
fn process_and_chunk_pdfs(
    py: Python<'_>, 
    paths: Vec<String>,
    chunk_size: usize,
    overlap: usize
) -> PyResult<Vec<PyObject>> {
    let counter = AtomicUsize::new(0);
    let total = paths.len();
    
    // Process all PDFs in parallel
    let all_chunks: Vec<(DocumentChunk, Vec<String>)> = paths
        .par_iter()
        .enumerate()
        .flat_map(|(idx, path)| {
            let doc_id = format!("doc_{:04}", idx);
            let page_chunks = process_single_pdf(path, &doc_id);
            
            let processed = counter.fetch_add(1, Ordering::SeqCst) + 1;
            if processed % 50 == 0 || processed == total {
                eprintln!("Processed {}/{} PDFs", processed, total);
            }
            
            // Split each page into smaller chunks for embedding
            page_chunks
                .into_iter()
                .map(|chunk| {
                    let text_chunks = split_into_chunks(&chunk.text, chunk_size, overlap);
                    (chunk, text_chunks)
                })
                .collect::<Vec<_>>()
        })
        .collect();
    
    // Convert to Python objects
    let mut result = Vec::new();
    
    for (chunk, text_chunks) in all_chunks {
        for (sub_idx, text) in text_chunks.iter().enumerate() {
            let dict = PyDict::new(py);
            dict.set_item("text", text)?;
            dict.set_item("doc_id", &chunk.doc_id)?;
            dict.set_item("page_num", chunk.page_num)?;
            dict.set_item("chunk_id", format!("{}_{}", chunk.chunk_id, sub_idx))?;
            dict.set_item("needs_ocr", chunk.needs_ocr)?;
            
            // Add metadata
            let meta_dict = PyDict::new(py);
            for (k, v) in &chunk.metadata {
                meta_dict.set_item(k, v)?;
            }
            meta_dict.set_item("original_text_length", chunk.text.len())?;
            dict.set_item("metadata", meta_dict)?;
            
            result.push(dict.into_pyobject(py)?.into_any().unbind());
        }
    }
    
    Ok(result)
}

/// Get system information for optimization
#[pyfunction]
fn get_system_info() -> PyResult<HashMap<String, String>> {
    let mut info = HashMap::new();
    info.insert("num_cpus".to_string(), rayon::current_num_threads().to_string());
    info.insert("rayon_threads".to_string(), rayon::current_num_threads().to_string());
    Ok(info)
}

/// Python module definition
#[pymodule]
fn lucio_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_pdfs_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(process_and_chunk_pdfs, m)?)?;
    m.add_function(wrap_pyfunction!(get_system_info, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_text_extraction() {
        let content = "(Hello) Tj (World) Tj";
        let result = extract_text_operators(content);
        assert!(result.contains("Hello"));
        assert!(result.contains("World"));
    }
    
    #[test]
    fn test_chunking() {
        let text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10";
        let chunks = split_into_chunks(text, 5, 2);
        assert!(chunks.len() >= 2);
    }
}
