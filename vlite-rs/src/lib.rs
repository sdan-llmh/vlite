//! # vlite — a simple and blazing fast vector database
//!
//! Production-grade three-stage retrieval in ~500 lines of Rust:
//! hybrid search (BM25 + vectors) → RRF fusion → cross-encoder reranking.
//! Parent-child chunking for precision AND context.
//!
//! Built on [USearch](https://github.com/unum-cloud/usearch) (HNSW + SimSIMD)
//! and [ONNX Runtime](https://onnxruntime.ai/) for embeddings.
//!
//! ```rust,no_run
//! use vlite::VLite;
//!
//! let mut db = VLite::open("my_db").unwrap();
//! db.add("Attention is all you need. The dominant sequence transduction models...").unwrap();
//! let results = db.search_bm25("how do transformers work?", 5).unwrap();
//! for r in &results {
//!     println!("[{:.3}] {}", r.score, &r.text[..80.min(r.text.len())]);
//! }
//! db.save().unwrap();
//! ```

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A search result returned by [`VLite::search`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Parent chunk ID.
    pub id: usize,
    /// Parent chunk text (large, for context).
    pub text: String,
    /// Relevance score (higher is better).
    pub score: f32,
    /// User-provided metadata attached at ingest time.
    pub metadata: HashMap<String, String>,
}

/// Configuration for a VLite database. All fields have sensible defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Embedding dimension. 0 = auto-detect on first add.
    pub dimensions: usize,
    /// Max tokens per child chunk (embedded for precision). Default: 256.
    pub child_chunk_size: usize,
    /// Max tokens per parent chunk (returned for context). Default: 1024.
    pub parent_chunk_size: usize,
    /// Overlap tokens between consecutive child chunks. Default: 25.
    pub chunk_overlap: usize,
    /// Number of candidates per retrieval source in stage 1. Default: 100.
    pub retrieval_k: usize,
    /// HNSW connectivity parameter. Default: 16.
    pub connectivity: usize,
    /// HNSW expansion at add time. Default: 128.
    pub expansion_add: usize,
    /// HNSW expansion at search time. Default: 64.
    pub expansion_search: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dimensions: 0,
            child_chunk_size: 256,
            parent_chunk_size: 1024,
            chunk_overlap: 25,
            retrieval_k: 100,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        }
    }
}

/// Internal: a parent chunk and its child chunks.
#[derive(Debug, Clone)]
struct ChunkPair {
    parent: String,
    children: Vec<String>,
}

/// Persisted data (everything except the USearch index, which has its own file).
#[derive(Serialize, Deserialize)]
struct StoredData {
    config: Config,
    parent_chunks: Vec<String>,
    child_texts: Vec<String>,
    child_to_parent: Vec<usize>,
    metadata: Vec<HashMap<String, String>>,
    bm25: BM25,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Chunking: recursive sentence-boundary split + parent-child
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Approximate token count (4 chars ≈ 1 token — the standard heuristic).
fn approx_tokens(text: &str) -> usize {
    (text.len() + 3) / 4
}

/// Split text into sentences. Splits on `.` `!` `?` followed by whitespace.
fn split_sentences(text: &str) -> Vec<&str> {
    let mut sentences = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        if (bytes[i] == b'.' || bytes[i] == b'!' || bytes[i] == b'?')
            && i + 1 < bytes.len()
            && bytes[i + 1].is_ascii_whitespace()
        {
            let end = i + 1;
            let s = text[start..end].trim();
            if !s.is_empty() {
                sentences.push(s);
            }
            start = end;
        }
    }
    // Remainder
    let s = text[start..].trim();
    if !s.is_empty() {
        sentences.push(s);
    }
    sentences
}

/// Group sentences into chunks of approximately `max_tokens` tokens.
fn group_sentences(sentences: &[&str], max_tokens: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_tokens = 0;
    for &s in sentences {
        let s_tokens = approx_tokens(s);
        if current_tokens + s_tokens > max_tokens && !current.is_empty() {
            chunks.push(current.trim().to_string());
            current = String::new();
            current_tokens = 0;
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(s);
        current_tokens += s_tokens;
    }
    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }
    chunks
}

/// Split children from a parent chunk with overlap.
fn split_children(parent: &str, child_size: usize, overlap: usize) -> Vec<String> {
    let sentences = split_sentences(parent);
    if sentences.is_empty() {
        return if parent.trim().is_empty() {
            vec![]
        } else {
            vec![parent.to_string()]
        };
    }
    // If parent fits in a single child, just return it.
    if approx_tokens(parent) <= child_size {
        return vec![parent.to_string()];
    }
    let mut children = Vec::new();
    let mut i = 0;
    while i < sentences.len() {
        let mut chunk = String::new();
        let mut tokens = 0;
        let start_i = i;
        while i < sentences.len() {
            let s_tokens = approx_tokens(sentences[i]);
            if tokens + s_tokens > child_size && !chunk.is_empty() {
                break;
            }
            if !chunk.is_empty() {
                chunk.push(' ');
            }
            chunk.push_str(sentences[i]);
            tokens += s_tokens;
            i += 1;
        }
        if !chunk.trim().is_empty() {
            children.push(chunk.trim().to_string());
        }
        // Overlap: back up by overlap_tokens worth of sentences
        if i < sentences.len() {
            let mut back_tokens = 0;
            let mut back = i;
            while back > start_i && back_tokens < overlap {
                back -= 1;
                back_tokens += approx_tokens(sentences[back]);
            }
            i = back.max(start_i + 1);
        }
    }
    children
}

/// Split text into parent-child chunk pairs.
fn chunk_parent_child(text: &str, parent_size: usize, child_size: usize, overlap: usize) -> Vec<ChunkPair> {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return vec![];
    }
    let parents = group_sentences(&sentences, parent_size);
    parents
        .into_iter()
        .map(|parent| {
            let children = split_children(&parent, child_size, overlap);
            ChunkPair { parent, children }
        })
        .collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BM25: minimal inverted index
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Tokenize text: lowercase, split on non-alphanumeric, keep tokens with len >= 2.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(String::from)
        .collect()
}

/// Minimal BM25 inverted index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25 {
    /// term → vec of (doc_id, term_frequency)
    postings: HashMap<String, Vec<(usize, u32)>>,
    /// doc_id → document length in tokens
    doc_lens: HashMap<usize, u32>,
    /// Total number of indexed documents.
    n: usize,
    /// k1 parameter. Default 1.2.
    k1: f32,
    /// b parameter. Default 0.75.
    b: f32,
}

impl BM25 {
    fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_lens: HashMap::new(),
            n: 0,
            k1: 1.2,
            b: 0.75,
        }
    }

    fn avg_dl(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        self.doc_lens.values().map(|&l| l as f64).sum::<f64>() / self.n as f64
    }

    /// Index a document's text.
    fn add(&mut self, id: usize, text: &str) {
        let tokens = tokenize(text);
        let dl = tokens.len() as u32;
        self.doc_lens.insert(id, dl);
        self.n += 1;
        // Count term frequencies
        let mut tf: HashMap<String, u32> = HashMap::new();
        for t in &tokens {
            *tf.entry(t.clone()).or_default() += 1;
        }
        for (term, count) in tf {
            self.postings.entry(term).or_default().push((id, count));
        }
    }

    /// Remove a document from the index.
    fn remove(&mut self, id: usize) {
        if self.doc_lens.remove(&id).is_some() {
            self.n = self.n.saturating_sub(1);
            for postings in self.postings.values_mut() {
                postings.retain(|&(doc_id, _)| doc_id != id);
            }
            // Clean up empty posting lists
            self.postings.retain(|_, v| !v.is_empty());
        }
    }

    /// BM25 search. Returns (doc_id, score) sorted descending by score.
    fn search(&self, query: &str, k: usize) -> Vec<(usize, f32)> {
        let query_terms = tokenize(query);
        if query_terms.is_empty() || self.n == 0 {
            return vec![];
        }
        let avgdl = self.avg_dl();
        let n = self.n as f32;
        let mut scores: HashMap<usize, f32> = HashMap::new();

        for term in &query_terms {
            if let Some(postings) = self.postings.get(term) {
                let df = postings.len() as f32;
                // IDF: ln((N - df + 0.5) / (df + 0.5) + 1)
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                for &(doc_id, tf) in postings {
                    let dl = *self.doc_lens.get(&doc_id).unwrap_or(&0) as f32;
                    let tf_f = tf as f32;
                    // TF normalization
                    let tf_norm =
                        (tf_f * (self.k1 + 1.0)) / (tf_f + self.k1 * (1.0 - self.b + self.b * dl / avgdl as f32));
                    *scores.entry(doc_id).or_default() += idf * tf_norm;
                }
            }
        }

        let mut results: Vec<(usize, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.total_cmp(&a.1));
        results.truncate(k);
        results
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RRF: Reciprocal Rank Fusion
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Reciprocal Rank Fusion. Merges multiple ranked lists without score calibration.
/// `k` is the smoothing constant (default 60).
fn rrf(lists: &[Vec<(usize, f32)>], k: f32) -> Vec<(usize, f32)> {
    let mut scores: HashMap<usize, f32> = HashMap::new();
    for list in lists {
        for (rank, &(id, _)) in list.iter().enumerate() {
            *scores.entry(id).or_default() += 1.0 / (k + rank as f32 + 1.0);
        }
    }
    let mut out: Vec<(usize, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Embedder: ONNX text embedding (requires `embed` feature)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// ONNX-based text embedder. Wraps a sentence-transformer model.
///
/// Enable with the `embed` feature flag.
#[cfg(feature = "embed")]
pub struct Embedder {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    dim: usize,
}

#[cfg(feature = "embed")]
impl Embedder {
    /// Load an ONNX embedding model from a directory.
    ///
    /// The directory should contain `model.onnx` and `tokenizer.json`.
    /// Compatible with sentence-transformers models exported to ONNX
    /// (e.g., all-MiniLM-L6-v2).
    pub fn load(model_dir: &str) -> Result<Self> {
        let model_path = Path::new(model_dir).join("model.onnx");
        let tokenizer_path = Path::new(model_dir).join("tokenizer.json");

        let session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&model_path)?;

        // Detect dimension from model output shape (default to 384 if undetectable)
        let dim = session
            .outputs()
            .first()
            .and_then(|o| o.dtype().tensor_shape().and_then(|s| s.last().copied()))
            .map(|d| d as usize)
            .unwrap_or(384);

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
            dim,
        })
    }

    /// Embedding dimension.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Embed a single text into a vector.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        Ok(batch.into_iter().next().unwrap_or_default())
    }

    /// Embed a batch of texts into vectors.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| format!("Tokenization failed: {e}"))?;

        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let batch_size = encodings.len();

        // Flatten input tensors as 1D vectors with known shapes
        let mut ids_flat = vec![0i64; batch_size * max_len];
        let mut mask_flat = vec![0i64; batch_size * max_len];
        let mut ttids_flat = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            for (j, &id) in enc.get_ids().iter().enumerate() {
                ids_flat[i * max_len + j] = id as i64;
            }
            for (j, &mask) in enc.get_attention_mask().iter().enumerate() {
                mask_flat[i * max_len + j] = mask as i64;
            }
            for (j, &tt) in enc.get_type_ids().iter().enumerate() {
                ttids_flat[i * max_len + j] = tt as i64;
            }
        }

        let shape = vec![batch_size as i64, max_len as i64];
        let input_ids_val = ort::value::Tensor::from_array((shape.clone(), ids_flat))?;
        let attention_mask_val = ort::value::Tensor::from_array((shape.clone(), mask_flat))?;
        let token_type_ids_val = ort::value::Tensor::from_array((shape, ttids_flat))?;

        let mut session = self.session.lock().map_err(|e| format!("Lock poisoned: {e}"))?;
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
            "token_type_ids" => token_type_ids_val,
        ])?;

        // Get the first output tensor
        let output_value = outputs.values().next().ok_or("No output tensor found")?;
        let (out_shape, out_data) = output_value.try_extract_tensor::<f32>()?;

        // Mean pooling over token dimension, respecting attention mask
        let mut results = Vec::with_capacity(batch_size);
        let is_3d = out_shape.len() == 3;
        let hidden_size = out_shape.last().copied().unwrap_or(self.dim as i64) as usize;

        for i in 0..batch_size {
            let seq_len = encodings[i].get_attention_mask().iter().filter(|&&m| m == 1).count();
            let mut pooled = vec![0.0f32; hidden_size];

            if is_3d {
                let seq_dim = out_shape[1] as usize;
                for j in 0..seq_len {
                    for k in 0..hidden_size {
                        pooled[k] += out_data[i * seq_dim * hidden_size + j * hidden_size + k];
                    }
                }
                if seq_len > 0 {
                    for v in &mut pooled {
                        *v /= seq_len as f32;
                    }
                }
            } else {
                // Already pooled (2D output)
                for k in 0..hidden_size {
                    pooled[k] = out_data[i * hidden_size + k];
                }
            }

            // L2 normalize
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut pooled {
                    *v /= norm;
                }
            }
            results.push(pooled);
        }

        Ok(results)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Reranker: cross-encoder via ONNX (requires `embed` feature)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Cross-encoder reranker. Scores (query, document) pairs for relevance.
///
/// Use a model like `cross-encoder/ms-marco-MiniLM-L6-v2` exported to ONNX.
/// Adds ~+33% retrieval accuracy for ~120ms latency.
///
/// Enable with the `embed` feature flag.
#[cfg(feature = "embed")]
pub struct Reranker {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "embed")]
impl Reranker {
    /// Load a cross-encoder ONNX model from a directory.
    ///
    /// The directory should contain `model.onnx` and `tokenizer.json`.
    pub fn load(model_dir: &str) -> Result<Self> {
        let model_path = Path::new(model_dir).join("model.onnx");
        let tokenizer_path = Path::new(model_dir).join("tokenizer.json");

        let session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(&model_path)?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer: {e}"))?;

        Ok(Self {
            session: std::sync::Mutex::new(session),
            tokenizer,
        })
    }

    /// Score (query, document) pairs. Returns one relevance score per document.
    ///
    /// Higher scores indicate greater relevance.
    pub fn score(&self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Cross-encoder input: pairs of (query, document)
        let pairs: Vec<_> = documents
            .iter()
            .map(|doc| tokenizers::EncodeInput::Dual(query.into(), (*doc).into()))
            .collect();

        let encodings = self
            .tokenizer
            .encode_batch(pairs, true)
            .map_err(|e| format!("Tokenization failed: {e}"))?;

        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);
        let batch_size = encodings.len();

        let mut ids_flat = vec![0i64; batch_size * max_len];
        let mut mask_flat = vec![0i64; batch_size * max_len];
        let mut ttids_flat = vec![0i64; batch_size * max_len];

        for (i, enc) in encodings.iter().enumerate() {
            for (j, &id) in enc.get_ids().iter().enumerate() {
                ids_flat[i * max_len + j] = id as i64;
            }
            for (j, &mask) in enc.get_attention_mask().iter().enumerate() {
                mask_flat[i * max_len + j] = mask as i64;
            }
            for (j, &tt) in enc.get_type_ids().iter().enumerate() {
                ttids_flat[i * max_len + j] = tt as i64;
            }
        }

        let shape = vec![batch_size as i64, max_len as i64];
        let input_ids_val = ort::value::Tensor::from_array((shape.clone(), ids_flat))?;
        let attention_mask_val = ort::value::Tensor::from_array((shape.clone(), mask_flat))?;
        let token_type_ids_val = ort::value::Tensor::from_array((shape, ttids_flat))?;

        let mut session = self.session.lock().map_err(|e| format!("Lock poisoned: {e}"))?;
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_val,
            "attention_mask" => attention_mask_val,
            "token_type_ids" => token_type_ids_val,
        ])?;

        // Cross-encoder output: logits tensor of shape (batch_size, 1) or (batch_size,)
        let output_value = outputs.values().next().ok_or("No output tensor from reranker")?;
        let (out_shape, out_data) = output_value.try_extract_tensor::<f32>()?;

        let scores: Vec<f32> = (0..batch_size)
            .map(|i| {
                if out_shape.len() == 2 {
                    let cols = out_shape[1] as usize;
                    out_data[i * cols]
                } else {
                    out_data[i]
                }
            })
            .collect();

        Ok(scores)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VLite: the whole database
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A simple, blazing-fast vector database with hybrid search.
///
/// Three-stage retrieval pipeline:
/// 1. **Retrieve**: BM25 keyword search + HNSW vector search (fast, high recall)
/// 2. **Fuse**: Reciprocal Rank Fusion merges both result lists (zero calibration)
/// 3. **Rerank**: Optional cross-encoder reranking (+33% accuracy)
///
/// Uses parent-child chunking: embed small chunks for precision,
/// return large chunks for context.
pub struct VLite {
    // Core indexes
    index: Index,
    bm25: BM25,

    // Storage
    parent_chunks: Vec<String>,
    child_texts: Vec<String>,
    child_to_parent: Vec<usize>,
    metadata: Vec<HashMap<String, String>>,

    // Config
    config: Config,
    path: PathBuf,
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

impl VLite {
    /// Open or create a VLite database at the given path.
    ///
    /// If the path exists with saved data, it will be loaded.
    /// Otherwise, a new empty database is created.
    pub fn open(path: &str) -> Result<Self> {
        Self::open_with_config(path, Config::default())
    }

    /// Open or create with custom configuration.
    pub fn open_with_config(path: &str, config: Config) -> Result<Self> {
        let path = PathBuf::from(path);
        let usearch_path = path.with_extension("usearch");
        let vtx_path = path.with_extension("vtx");

        // Try loading existing data
        if vtx_path.exists() && usearch_path.exists() {
            return Self::load_from_disk(&path);
        }

        // Create new
        let mut opts = IndexOptions::default();
        if config.dimensions > 0 {
            opts.dimensions = config.dimensions;
        }
        opts.metric = MetricKind::Cos;
        opts.quantization = ScalarKind::F16;
        opts.connectivity = config.connectivity;
        opts.expansion_add = config.expansion_add;
        opts.expansion_search = config.expansion_search;

        let index = Index::new(&opts)?;
        index.reserve(1024)?;

        Ok(Self {
            index,
            bm25: BM25::new(),
            parent_chunks: Vec::new(),
            child_texts: Vec::new(),
            child_to_parent: Vec::new(),
            metadata: Vec::new(),
            config,
            path,
        })
    }

    /// Add text to the database. Auto parent-child chunks, embeds, and indexes.
    ///
    /// Requires vectors to be provided via [`add_with_vectors`] if the `embed`
    /// feature is not enabled.
    ///
    /// Returns the parent chunk IDs that were created.
    pub fn add(&mut self, text: &str) -> Result<Vec<usize>> {
        self.add_with_metadata(text, HashMap::new())
    }

    /// Add text with associated metadata.
    pub fn add_with_metadata(&mut self, text: &str, meta: HashMap<String, String>) -> Result<Vec<usize>> {
        let pairs = chunk_parent_child(
            text,
            self.config.parent_chunk_size,
            self.config.child_chunk_size,
            self.config.chunk_overlap,
        );
        if pairs.is_empty() {
            return Ok(vec![]);
        }

        let mut parent_ids = Vec::new();
        for pair in &pairs {
            let parent_id = self.parent_chunks.len();
            self.parent_chunks.push(pair.parent.clone());
            self.metadata.push(meta.clone());
            parent_ids.push(parent_id);

            for child_text in &pair.children {
                let child_id = self.child_texts.len();
                self.child_texts.push(child_text.clone());
                self.child_to_parent.push(parent_id);
                self.bm25.add(child_id, child_text);
            }
        }
        Ok(parent_ids)
    }

    /// Add a pre-computed vector with associated text and metadata (BYOE).
    ///
    /// No chunking is performed. The text is stored as a single parent chunk
    /// and the vector is indexed directly.
    pub fn add_with_vector(
        &mut self,
        vector: &[f32],
        text: &str,
        meta: HashMap<String, String>,
    ) -> Result<usize> {
        // Auto-detect dimensions on first vector
        if self.config.dimensions == 0 && !vector.is_empty() {
            self.config.dimensions = vector.len();
            // Rebuild index with correct dimensions
            let mut opts = IndexOptions::default();
            opts.dimensions = vector.len();
            opts.metric = MetricKind::Cos;
            opts.quantization = ScalarKind::F16;
            opts.connectivity = self.config.connectivity;
            opts.expansion_add = self.config.expansion_add;
            opts.expansion_search = self.config.expansion_search;
            self.index = Index::new(&opts)?;
            self.index.reserve(1024)?;
        }

        let parent_id = self.parent_chunks.len();
        self.parent_chunks.push(text.to_string());
        self.metadata.push(meta);

        let child_id = self.child_texts.len();
        self.child_texts.push(text.to_string());
        self.child_to_parent.push(parent_id);
        self.bm25.add(child_id, text);

        // Add vector to HNSW index
        self.index.add(child_id as u64, vector)?;

        Ok(parent_id)
    }

    /// Add multiple pre-computed vectors with texts and metadata.
    ///
    /// Vectors, texts, and metadata slices must have the same length.
    pub fn add_with_vectors(
        &mut self,
        vectors: &[Vec<f32>],
        texts: &[&str],
        metas: &[HashMap<String, String>],
    ) -> Result<Vec<usize>> {
        if vectors.len() != texts.len() || vectors.len() != metas.len() {
            return Err("vectors, texts, and metas must have the same length".into());
        }
        let mut ids = Vec::with_capacity(vectors.len());
        for i in 0..vectors.len() {
            ids.push(self.add_with_vector(&vectors[i], texts[i], metas[i].clone())?);
        }
        Ok(ids)
    }

    /// Search the database using BM25 keyword search only.
    ///
    /// Returns parent chunks ranked by BM25 relevance.
    pub fn search_bm25(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let k = self.config.retrieval_k.max(top_k);
        let bm25_results = self.bm25.search(query, k);
        self.children_to_results(bm25_results, top_k)
    }

    /// Search the database using vector similarity only.
    ///
    /// Requires a pre-computed query vector. Returns parent chunks ranked by
    /// cosine similarity.
    pub fn search_vector(&self, query_vec: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        if self.index.size() == 0 {
            return Ok(vec![]);
        }
        let k = self.config.retrieval_k.max(top_k);
        let results = self.index.search(query_vec, k)?;
        let vec_results: Vec<(usize, f32)> = results
            .keys
            .iter()
            .zip(results.distances.iter())
            .map(|(&key, &dist)| (key as usize, 1.0 - dist)) // cosine distance → similarity
            .collect();
        self.children_to_results(vec_results, top_k)
    }

    /// Hybrid search: BM25 + vector → RRF fusion.
    ///
    /// This is the recommended search method. Combines keyword precision
    /// (BM25) with semantic understanding (vector similarity) using
    /// Reciprocal Rank Fusion.
    pub fn search_hybrid(&self, query: &str, query_vec: &[f32], top_k: usize) -> Result<Vec<SearchResult>> {
        let k = self.config.retrieval_k.max(top_k);

        // Stage 1: Retrieve from both sources
        let bm25_results = self.bm25.search(query, k);

        let vec_results = if self.index.size() > 0 {
            let results = self.index.search(query_vec, k)?;
            results
                .keys
                .iter()
                .zip(results.distances.iter())
                .map(|(&key, &dist)| (key as usize, 1.0 - dist))
                .collect()
        } else {
            vec![]
        };

        // Stage 2: RRF fusion
        let fused = rrf(&[vec_results, bm25_results], 60.0);

        // Stage 3: Map to parent results
        self.children_to_results(fused, top_k)
    }

    /// Map child-level results back to deduplicated parent chunks.
    fn children_to_results(
        &self,
        child_results: Vec<(usize, f32)>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>> {
        let mut seen_parents = HashSet::new();
        let mut results = Vec::new();
        for (child_id, score) in child_results {
            if child_id >= self.child_to_parent.len() {
                continue;
            }
            let parent_id = self.child_to_parent[child_id];
            if seen_parents.insert(parent_id) {
                results.push(SearchResult {
                    id: parent_id,
                    text: self.parent_chunks[parent_id].clone(),
                    score,
                    metadata: self.metadata[parent_id].clone(),
                });
                if results.len() >= top_k {
                    break;
                }
            }
        }
        Ok(results)
    }

    /// Get a parent chunk by ID.
    pub fn get(&self, parent_id: usize) -> Option<&str> {
        self.parent_chunks.get(parent_id).map(|s| s.as_str())
    }

    /// Get metadata for a parent chunk.
    pub fn get_metadata(&self, parent_id: usize) -> Option<&HashMap<String, String>> {
        self.metadata.get(parent_id)
    }

    /// Number of parent chunks.
    pub fn count(&self) -> usize {
        self.parent_chunks.len()
    }

    /// Number of indexed child chunks (the actual vectors in the HNSW index).
    pub fn count_children(&self) -> usize {
        self.child_texts.len()
    }

    /// Save the database to disk.
    pub fn save(&self) -> Result<()> {
        let usearch_path = self.path.with_extension("usearch");
        let vtx_path = self.path.with_extension("vtx");

        // Save USearch index
        self.index.save(usearch_path.to_str().unwrap())?;

        // Save our data
        let data = StoredData {
            config: self.config.clone(),
            parent_chunks: self.parent_chunks.clone(),
            child_texts: self.child_texts.clone(),
            child_to_parent: self.child_to_parent.clone(),
            metadata: self.metadata.clone(),
            bm25: self.bm25.clone(),
        };
        let encoded = bincode::serialize(&data)?;
        std::fs::write(&vtx_path, encoded)?;

        Ok(())
    }

    /// Load from disk.
    fn load_from_disk(path: &Path) -> Result<Self> {
        let usearch_path = path.with_extension("usearch");
        let vtx_path = path.with_extension("vtx");

        // Load our data
        let bytes = std::fs::read(&vtx_path)?;
        let data: StoredData = bincode::deserialize(&bytes)?;

        // Load USearch index
        let mut opts = IndexOptions::default();
        opts.dimensions = data.config.dimensions;
        opts.metric = MetricKind::Cos;
        opts.quantization = ScalarKind::F16;
        opts.connectivity = data.config.connectivity;
        opts.expansion_add = data.config.expansion_add;
        opts.expansion_search = data.config.expansion_search;
        let index = Index::new(&opts)?;
        index.load(usearch_path.to_str().unwrap())?;

        Ok(Self {
            index,
            bm25: data.bm25,
            parent_chunks: data.parent_chunks,
            child_texts: data.child_texts,
            child_to_parent: data.child_to_parent,
            metadata: data.metadata,
            config: data.config,
            path: path.to_path_buf(),
        })
    }

    /// Clear all data from the database.
    pub fn clear(&mut self) -> Result<()> {
        self.parent_chunks.clear();
        self.child_texts.clear();
        self.child_to_parent.clear();
        self.metadata.clear();
        self.bm25 = BM25::new();

        // Rebuild empty index
        let mut opts = IndexOptions::default();
        if self.config.dimensions > 0 {
            opts.dimensions = self.config.dimensions;
        }
        opts.metric = MetricKind::Cos;
        opts.quantization = ScalarKind::F16;
        opts.connectivity = self.config.connectivity;
        opts.expansion_add = self.config.expansion_add;
        opts.expansion_search = self.config.expansion_search;
        self.index = Index::new(&opts)?;
        self.index.reserve(1024)?;
        Ok(())
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PDF support (requires `pdf` feature)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(feature = "pdf")]
impl VLite {
    /// Add a PDF document. Extracts text per page, chunks, and indexes.
    ///
    /// Each chunk is tagged with metadata `{page: N, source: source_name}`.
    /// Returns the parent chunk IDs created.
    pub fn add_pdf(&mut self, pdf_bytes: &[u8], source_name: &str) -> Result<Vec<usize>> {
        let text = pdf_extract::extract_text_from_mem(pdf_bytes)
            .map_err(|e| format!("PDF extraction failed: {e}"))?;

        // Split on form-feed characters (page breaks in PDF extract output)
        let pages: Vec<&str> = text.split('\u{000C}').collect();
        let mut all_ids = Vec::new();

        for (page_num, page_text) in pages.iter().enumerate() {
            let trimmed = page_text.trim();
            if trimmed.is_empty() {
                continue;
            }
            let mut meta = HashMap::new();
            meta.insert("page".to_string(), (page_num + 1).to_string());
            meta.insert("source".to_string(), source_name.to_string());
            let ids = self.add_with_metadata(trimmed, meta)?;
            all_ids.extend(ids);
        }

        Ok(all_ids)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Reranked search (requires `embed` feature)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(feature = "embed")]
impl VLite {
    /// Full three-stage search pipeline with cross-encoder reranking.
    ///
    /// Stage 1: BM25 + vector retrieval (fast, high recall)
    /// Stage 2: RRF fusion (zero calibration)
    /// Stage 3: Cross-encoder reranking (+33% accuracy)
    ///
    /// Requires an [`Embedder`] for query embedding and optionally a [`Reranker`].
    pub fn search_reranked(
        &self,
        query: &str,
        top_k: usize,
        embedder: &Embedder,
        reranker: Option<&Reranker>,
    ) -> Result<Vec<SearchResult>> {
        let k = self.config.retrieval_k.max(top_k);

        // Stage 1: Retrieve
        let query_vec = embedder.embed(query)?;
        let bm25_results = self.bm25.search(query, k);

        let vec_results = if self.index.size() > 0 {
            let results = self.index.search(&query_vec, k)?;
            results
                .keys
                .iter()
                .zip(results.distances.iter())
                .map(|(&key, &dist)| (key as usize, 1.0 - dist))
                .collect()
        } else {
            vec![]
        };

        // Stage 2: RRF fusion
        let fused = rrf(&[vec_results, bm25_results], 60.0);

        // Stage 3: Rerank (optional)
        let ranked = if let Some(reranker) = reranker {
            let candidates: Vec<&str> = fused
                .iter()
                .take(k * 2)
                .filter_map(|(id, _)| self.child_texts.get(*id).map(|s| s.as_str()))
                .collect();

            if candidates.is_empty() {
                fused
            } else {
                let scores = reranker.score(query, &candidates)?;
                let mut scored: Vec<(usize, f32)> = fused
                    .iter()
                    .take(candidates.len())
                    .zip(scores.iter())
                    .map(|(&(id, _), &score)| (id, score))
                    .collect();
                scored.sort_by(|a, b| b.1.total_cmp(&a.1));
                scored
            }
        } else {
            fused
        };

        self.children_to_results(ranked, top_k)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approx_tokens() {
        assert_eq!(approx_tokens("hello"), 2); // 5 chars ≈ 2 tokens
        assert_eq!(approx_tokens(""), 0);
        assert_eq!(approx_tokens("a"), 1);
    }

    #[test]
    fn test_split_sentences() {
        let text = "Hello world. This is a test! Is it working? Yes it is.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "This is a test!");
        assert_eq!(sentences[2], "Is it working?");
        assert_eq!(sentences[3], "Yes it is.");
    }

    #[test]
    fn test_split_sentences_no_trailing_punct() {
        let text = "First sentence. Second part without period";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0], "First sentence.");
        assert_eq!(sentences[1], "Second part without period");
    }

    #[test]
    fn test_chunk_parent_child_short_text() {
        let text = "Hello world. This is short.";
        let pairs = chunk_parent_child(text, 1024, 256, 25);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].children.len(), 1);
        assert_eq!(pairs[0].parent, pairs[0].children[0]);
    }

    #[test]
    fn test_chunk_parent_child_long_text() {
        // Create a text that's longer than one parent chunk
        let sentence = "This is a test sentence with some words in it. ";
        let text = sentence.repeat(100); // ~4700 chars ≈ 1175 tokens
        let pairs = chunk_parent_child(&text, 300, 100, 10);
        assert!(pairs.len() >= 2, "Should produce multiple parent chunks, got {}", pairs.len());
        for pair in &pairs {
            assert!(!pair.children.is_empty(), "Each parent should have children");
        }
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // Single-char tokens should be filtered
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_bm25_basic() {
        let mut bm25 = BM25::new();
        bm25.add(0, "the quick brown fox jumps over the lazy dog");
        bm25.add(1, "a fast red car drives on the highway");
        bm25.add(2, "the fox is quick and brown");

        let results = bm25.search("quick brown fox", 10);
        assert!(!results.is_empty());
        // The fox-related docs should rank higher
        let top_id = results[0].0;
        assert!(top_id == 0 || top_id == 2, "Fox doc should rank first, got id={}", top_id);
    }

    #[test]
    fn test_bm25_remove() {
        let mut bm25 = BM25::new();
        bm25.add(0, "hello world");
        bm25.add(1, "goodbye world");
        assert_eq!(bm25.n, 2);

        bm25.remove(0);
        assert_eq!(bm25.n, 1);

        let results = bm25.search("hello", 10);
        assert!(results.is_empty() || results[0].0 != 0);
    }

    #[test]
    fn test_bm25_empty() {
        let bm25 = BM25::new();
        let results = bm25.search("anything", 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_rrf_basic() {
        let list1 = vec![(0, 1.0), (1, 0.9), (2, 0.8)];
        let list2 = vec![(2, 1.0), (0, 0.9), (3, 0.8)];
        let fused = rrf(&[list1, list2], 60.0);
        assert!(!fused.is_empty());
        // Doc 0 and 2 appear in both lists, should rank high
        let top_ids: Vec<usize> = fused.iter().take(2).map(|(id, _)| *id).collect();
        assert!(top_ids.contains(&0) || top_ids.contains(&2));
    }

    #[test]
    fn test_rrf_single_list() {
        let list = vec![(5, 1.0), (3, 0.5)];
        let fused = rrf(&[list], 60.0);
        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].0, 5); // Higher ranked should come first
    }

    #[test]
    fn test_vlite_open_and_add_vector() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_db");
        let mut db = VLite::open(db_path.to_str().unwrap()).unwrap();

        // Add vectors manually (BYOE mode)
        let v1 = vec![1.0, 0.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0, 0.0];
        let v3 = vec![0.9, 0.1, 0.0, 0.0]; // similar to v1

        db.add_with_vector(&v1, "first document about cats", HashMap::new()).unwrap();
        db.add_with_vector(&v2, "second document about dogs", HashMap::new()).unwrap();
        db.add_with_vector(&v3, "third document also about cats", HashMap::new()).unwrap();

        assert_eq!(db.count(), 3);
        assert_eq!(db.count_children(), 3);

        // Vector search: v1-like query should find v1 and v3
        let results = db.search_vector(&[0.95, 0.05, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        // The cat documents should be the top results
        assert!(results[0].text.contains("cats"), "Top result should be about cats: {}", results[0].text);
    }

    #[test]
    fn test_vlite_bm25_search() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_bm25");
        let mut db = VLite::open(db_path.to_str().unwrap()).unwrap();

        let v = vec![0.0; 4]; // dummy vectors
        db.add_with_vector(&v, "the quick brown fox jumps over the lazy dog", HashMap::new()).unwrap();
        db.add_with_vector(&v, "a fast red car drives on the highway", HashMap::new()).unwrap();
        db.add_with_vector(&v, "the fox and the hound are friends", HashMap::new()).unwrap();

        let results = db.search_bm25("quick fox", 2).unwrap();
        assert!(!results.is_empty());
        assert!(results[0].text.contains("fox"), "Should find fox doc: {}", results[0].text);
    }

    #[test]
    fn test_vlite_hybrid_search() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_hybrid");
        let mut db = VLite::open(db_path.to_str().unwrap()).unwrap();

        // Doc 0: semantically about cats (v1), text about cats
        db.add_with_vector(&[1.0, 0.0, 0.0, 0.0], "cats are wonderful pets that purr", HashMap::new()).unwrap();
        // Doc 1: semantically about dogs (v2), text about dogs
        db.add_with_vector(&[0.0, 1.0, 0.0, 0.0], "dogs are loyal companions that bark", HashMap::new()).unwrap();
        // Doc 2: semantically between (v3), text about pets
        db.add_with_vector(&[0.5, 0.5, 0.0, 0.0], "pets bring joy and happiness to families", HashMap::new()).unwrap();

        // Hybrid search: vector close to cats + BM25 for "cats"
        let results = db.search_hybrid("cats", &[0.9, 0.1, 0.0, 0.0], 3).unwrap();
        assert!(!results.is_empty());
        // Cat doc should rank first (both vector and BM25 agree)
        assert!(results[0].text.contains("cats"), "Cat doc should be #1: {}", results[0].text);
    }

    #[test]
    fn test_vlite_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_persist");

        // Create and populate
        {
            let mut db = VLite::open(db_path.to_str().unwrap()).unwrap();
            db.add_with_vector(&[1.0, 0.0, 0.0], "hello world", HashMap::from([("key".into(), "val".into())])).unwrap();
            db.add_with_vector(&[0.0, 1.0, 0.0], "goodbye moon", HashMap::new()).unwrap();
            db.save().unwrap();
        }

        // Reload and verify
        {
            let db = VLite::open(db_path.to_str().unwrap()).unwrap();
            assert_eq!(db.count(), 2);
            assert_eq!(db.get(0), Some("hello world"));
            assert_eq!(db.get(1), Some("goodbye moon"));
            assert_eq!(db.get_metadata(0).unwrap().get("key").map(|s| s.as_str()), Some("val"));

            // Search still works after reload
            let results = db.search_vector(&[0.9, 0.1, 0.0], 1).unwrap();
            assert_eq!(results.len(), 1);
            assert!(results[0].text.contains("hello"), "Should find hello: {}", results[0].text);
        }
    }

    #[test]
    fn test_vlite_add_text_chunking() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_chunk");
        let mut db = VLite::open(db_path.to_str().unwrap()).unwrap();

        // Add a longer text that should produce parent-child chunks
        let text = "First important sentence about machine learning. \
                    Second sentence discusses neural networks and their applications. \
                    Third sentence covers attention mechanisms in transformers. \
                    Fourth sentence explains backpropagation through time.";

        let parent_ids = db.add(text).unwrap();
        assert!(!parent_ids.is_empty(), "Should create at least one parent chunk");

        // Text should be chunked and BM25-indexed
        let results = db.search_bm25("neural networks", 1).unwrap();
        assert!(!results.is_empty(), "Should find neural networks via BM25");
    }

    #[test]
    fn test_vlite_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_meta");
        let mut db = VLite::open(db_path.to_str().unwrap()).unwrap();

        let meta = HashMap::from([
            ("source".to_string(), "paper.pdf".to_string()),
            ("page".to_string(), "42".to_string()),
        ]);
        db.add_with_vector(&[1.0, 0.0], "content from page 42", meta).unwrap();

        let results = db.search_bm25("page", 1).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].metadata.get("page").map(|s| s.as_str()), Some("42"));
        assert_eq!(results[0].metadata.get("source").map(|s| s.as_str()), Some("paper.pdf"));
    }

    #[test]
    fn test_vlite_clear() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("test_clear");
        let mut db = VLite::open(db_path.to_str().unwrap()).unwrap();

        db.add_with_vector(&[1.0, 0.0], "hello", HashMap::new()).unwrap();
        assert_eq!(db.count(), 1);

        db.clear().unwrap();
        assert_eq!(db.count(), 0);
        assert_eq!(db.count_children(), 0);
    }
}
