//! # vlite — a simple and blazing fast vector database
//!
//! Production-grade retrieval in ~700 lines of Rust.
//! Hybrid search (vector + BM25 + RRF) by default. Parent-child chunking built in.
//!
//! ```no_run
//! use vlite::VLite;
//!
//! let mut db = VLite::new().unwrap();
//! db.add("the mitochondria is the powerhouse of the cell", None).unwrap();
//! let results = db.search("biology energy", 3).unwrap();
//! ```

pub mod chunk;
pub mod embed;
pub mod error;
pub mod extract;
pub mod storage;

pub use error::{Result, VLiteError};
pub use extract::{extract_html, extract_pdf};

use embed::{Embedder, VisionEmbedder};
use simsimd::SpatialSimilarity;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Metadata type alias.
pub type Metadata = HashMap<String, serde_json::Value>;

/// A search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Index in the database.
    pub index: usize,
    /// Parent text — larger context around the matching chunk (return this to LLMs).
    pub text: String,
    /// The chunk that actually matched (what was embedded + searched).
    pub chunk_text: String,
    /// Relevance score (higher = more relevant). Scale depends on search method.
    pub score: f32,
    /// User-provided metadata.
    pub metadata: Metadata,
}

/// A simple and blazing fast vector database.
///
/// Stores text (auto-chunked), embeds via ONNX, searches via hybrid vector + BM25 + RRF.
/// Optionally supports images via CLIP (use [`VLite::with_clip`]).
pub struct VLite {
    // Stored data
    texts: Vec<String>,      // chunk texts (what was embedded)
    parents: Vec<String>,    // parent texts (what gets returned)
    vectors: Vec<f32>,       // flat embedding buffer [dim × count]
    metadata: Vec<Metadata>,

    // BM25 state (updated incrementally)
    doc_lens: Vec<usize>,             // word count per chunk
    avg_doc_len: f32,                 // corpus average word count
    df: HashMap<String, usize>,       // document frequency per term

    // Config
    dim: usize,
    max_tokens: usize,
    has_vision: bool,

    // Models
    embedder: Embedder,
    vision: Option<VisionEmbedder>,

    // Persistence
    path: Option<PathBuf>,
}

// ============================================================================
// Construction
// ============================================================================

impl VLite {
    /// Create a new text-only database.
    ///
    /// Uses `all-MiniLM-L6-v2` (384-dim) — best quality for text search.
    /// Model auto-downloads from HuggingFace Hub on first use (~80MB).
    pub fn new() -> Result<Self> {
        let embedder = Embedder::new()?;
        let dim = embedder.dim;
        let max_tokens = embedder.max_tokens;
        Ok(Self {
            texts: Vec::new(),
            parents: Vec::new(),
            vectors: Vec::new(),
            metadata: Vec::new(),
            doc_lens: Vec::new(),
            avg_doc_len: 0.0,
            df: HashMap::new(),
            dim,
            max_tokens,
            has_vision: false,
            embedder,
            vision: None,
            path: None,
        })
    }

    /// Create a multimodal database (text + images).
    ///
    /// Uses CLIP ViT-B/32 (512-dim) — enables cross-modal text↔image search.
    /// Both text and vision models auto-download on first use.
    pub fn with_clip() -> Result<Self> {
        let embedder = Embedder::clip_text()?;
        let vision = VisionEmbedder::clip_vision()?;
        let dim = embedder.dim;
        let max_tokens = embedder.max_tokens;
        Ok(Self {
            texts: Vec::new(),
            parents: Vec::new(),
            vectors: Vec::new(),
            metadata: Vec::new(),
            doc_lens: Vec::new(),
            avg_doc_len: 0.0,
            df: HashMap::new(),
            dim,
            max_tokens,
            has_vision: true,
            embedder,
            vision: Some(vision),
            path: None,
        })
    }

    /// Open an existing `.vlite` file.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let state = storage::load(path)?;
        let embedder = if state.flags & storage::FLAG_HAS_VISION != 0 {
            Embedder::clip_text()?
        } else {
            Embedder::new()?
        };
        let vision = if state.flags & storage::FLAG_HAS_VISION != 0 {
            Some(VisionEmbedder::clip_vision()?)
        } else {
            None
        };
        Ok(Self {
            texts: state.texts,
            parents: state.parents,
            vectors: state.vectors,
            metadata: state.metadata,
            doc_lens: state.doc_lens,
            avg_doc_len: state.avg_doc_len,
            df: state.df,
            dim: state.dim,
            max_tokens: embedder.max_tokens,
            has_vision: state.flags & storage::FLAG_HAS_VISION != 0,
            embedder,
            vision,
            path: Some(path.to_path_buf()),
        })
    }
}

// ============================================================================
// Add
// ============================================================================

impl VLite {
    /// Add text to the database.
    ///
    /// If the text exceeds the model's context window, it is automatically chunked
    /// at natural boundaries (paragraphs → sentences → words) with parent-child
    /// grouping and contextual headers.
    ///
    /// Returns the indices of all stored items (1 for short text, N for chunked).
    pub fn add(&mut self, text: &str, metadata: Option<Metadata>) -> Result<Vec<usize>> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(vec![]);
        }

        let token_count = self.embedder.token_count(trimmed);
        let meta = metadata.unwrap_or_default();

        if token_count <= self.max_tokens {
            // Short text — embed directly, parent = self
            let vec = self.embedder.embed(trimmed)?;
            let idx = self.store(trimmed, trimmed, &vec, meta)?;
            Ok(vec![idx])
        } else {
            // Long text — auto-chunk with parent-child + contextual headers
            let title = meta
                .get("title")
                .and_then(|v| v.as_str())
                .or_else(|| meta.get("source").and_then(|v| v.as_str()))
                .unwrap_or("");

            // child: ~256 tokens ≈ 1024 chars, parent: ~1024 tokens ≈ 4096 chars
            let child_chars = self.max_tokens * 4;
            let parent_chars = self.max_tokens * 16;
            let overlap_chars = self.max_tokens / 2; // ~50 tokens overlap

            let chunks = chunk::chunk(trimmed, title, child_chars, parent_chars, overlap_chars);

            let total = chunks.len();
            let mut indices = Vec::with_capacity(total);

            // Batch embed all children
            let child_texts: Vec<&str> = chunks.iter().map(|c| c.child_text.as_str()).collect();
            let vecs = self.embedder.embed_batch(&child_texts)?;

            for (i, (c, vec)) in chunks.iter().zip(vecs.iter()).enumerate() {
                let mut item_meta = meta.clone();
                item_meta.insert("_chunk".into(), serde_json::json!(i));
                item_meta.insert("_total_chunks".into(), serde_json::json!(total));

                let idx = self.store(&c.child_text, &c.parent_text, vec, item_meta)?;
                indices.push(idx);
            }

            Ok(indices)
        }
    }

    /// Add an image (JPEG/PNG/WebP bytes). Requires multimodal mode (`VLite::with_clip()`).
    ///
    /// Embeds via CLIP vision encoder into the same vector space as text.
    /// Cross-modal search is automatic.
    pub fn add_image(&mut self, image_bytes: &[u8], metadata: Option<Metadata>) -> Result<usize> {
        let vision = self.vision.as_ref().ok_or(VLiteError::NoVisionModel)?;
        let vec = vision.embed_image(image_bytes)?;
        let description = format!("[image:{}bytes]", image_bytes.len());
        self.store(&description, &description, &vec, metadata.unwrap_or_default())
    }

    /// Internal: store an item and update BM25 state.
    fn store(
        &mut self,
        text: &str,
        parent: &str,
        vector: &[f32],
        metadata: Metadata,
    ) -> Result<usize> {
        let idx = self.texts.len();
        self.texts.push(text.to_string());
        self.parents.push(parent.to_string());
        self.vectors.extend_from_slice(vector);
        self.metadata.push(metadata);

        // Update BM25 state
        let words: Vec<&str> = text.split_whitespace().collect();
        let doc_len = words.len();
        self.doc_lens.push(doc_len);

        // Update average doc length
        let n = self.texts.len() as f32;
        self.avg_doc_len = ((self.avg_doc_len * (n - 1.0)) + doc_len as f32) / n;

        // Update document frequency (count each unique term once per doc)
        let mut seen = std::collections::HashSet::new();
        for word in &words {
            let lower = word.to_lowercase();
            // Strip basic punctuation
            let term: String = lower.chars().filter(|c| c.is_alphanumeric()).collect();
            if !term.is_empty() && seen.insert(term.clone()) {
                *self.df.entry(term).or_insert(0) += 1;
            }
        }

        Ok(idx)
    }
}

// ============================================================================
// Search — ALWAYS hybrid (vector + BM25 + RRF)
// ============================================================================

impl VLite {
    /// Search the database. Returns the top-k most relevant results.
    ///
    /// **Always hybrid**: combines vector similarity (SimSIMD cosine) with BM25 keyword
    /// matching via Reciprocal Rank Fusion (RRF). This catches both semantic meaning
    /// AND exact keyword matches (error codes, identifiers, rare terms).
    ///
    /// Returns parent texts (larger context) — not just the matching chunk.
    pub fn search(&self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        let n = self.len();
        if n == 0 {
            return Err(VLiteError::EmptyDatabase);
        }
        let k = top_k.min(n);

        // Embed query
        let query_vec = self.embedder.embed(query)?;

        // Lowercase query terms for BM25
        let terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.chars().filter(|c| c.is_alphanumeric()).collect::<String>())
            .filter(|s| !s.is_empty())
            .collect();

        // Stage 1: Vector ranking (SimSIMD cosine distance → similarity)
        let mut vec_rank: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let v = &self.vectors[start..end];
                let dist = f32::cosine(&query_vec, v).unwrap_or(1.0) as f32;
                (i, 1.0f32 - dist) // distance → similarity
            })
            .collect();
        vec_rank.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Stage 2: BM25 ranking (inline, no inverted index)
        let mut bm25_rank: Vec<(usize, f32)> = (0..n)
            .map(|i| (i, self.bm25_score(&terms, i)))
            .collect();
        bm25_rank.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Stage 3: RRF fusion — rank-based, no score normalization needed
        let mut rrf_scores = vec![0.0f32; n];
        for (rank, &(i, _)) in vec_rank.iter().enumerate() {
            rrf_scores[i] += 1.0 / (60.0 + rank as f32);
        }
        for (rank, &(i, _)) in bm25_rank.iter().enumerate() {
            rrf_scores[i] += 1.0 / (60.0 + rank as f32);
        }

        let mut fused: Vec<(usize, f32)> = rrf_scores.into_iter().enumerate().collect();
        fused.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        fused.truncate(k);

        // Build results — return PARENT texts for rich context
        Ok(fused
            .into_iter()
            .map(|(i, score)| SearchResult {
                index: i,
                text: self.parents[i].clone(),
                chunk_text: self.texts[i].clone(),
                score,
                metadata: self.metadata[i].clone(),
            })
            .collect())
    }

    /// BM25 score for a single document against query terms.
    /// Same algorithm as Elasticsearch — whole-word matching, not substring.
    fn bm25_score(&self, query_terms: &[String], doc_idx: usize) -> f32 {
        let doc_len = self.doc_lens[doc_idx] as f32;
        // Tokenize document into words (same normalization as df index)
        let doc_words: Vec<String> = self.texts[doc_idx]
            .split_whitespace()
            .map(|w| w.to_lowercase().chars().filter(|c| c.is_alphanumeric()).collect::<String>())
            .filter(|s| !s.is_empty())
            .collect();
        let n = self.len() as f32;
        let k1: f32 = 1.5;
        let b: f32 = 0.75;
        let avg_dl = if self.avg_doc_len > 0.0 {
            self.avg_doc_len
        } else {
            1.0
        };

        query_terms
            .iter()
            .map(|term| {
                // Whole-word match: count occurrences of term in document words
                let tf = doc_words.iter().filter(|w| *w == term).count() as f32;
                let df = self.df.get(term).copied().unwrap_or(0) as f32;
                let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
                idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * doc_len / avg_dl))
            })
            .sum()
    }
}

// ============================================================================
// Delete, Save, Utility
// ============================================================================

impl VLite {
    /// Delete an item by index.
    pub fn delete(&mut self, index: usize) -> Result<()> {
        let n = self.len();
        if index >= n {
            return Err(VLiteError::IndexOutOfBounds(index));
        }

        // Update df: decrement for terms in this document
        let words: Vec<String> = self.texts[index]
            .split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
            })
            .filter(|s| !s.is_empty())
            .collect();
        let mut seen = std::collections::HashSet::new();
        for term in &words {
            if seen.insert(term.clone()) {
                if let Some(count) = self.df.get_mut(term) {
                    *count = count.saturating_sub(1);
                    if *count == 0 {
                        self.df.remove(term);
                    }
                }
            }
        }

        // Remove from all parallel vecs
        self.texts.remove(index);
        self.parents.remove(index);
        self.metadata.remove(index);
        self.doc_lens.remove(index);

        // Remove vector slice
        let start = index * self.dim;
        let end = start + self.dim;
        self.vectors.drain(start..end);

        // Recompute avg_doc_len
        if self.texts.is_empty() {
            self.avg_doc_len = 0.0;
        } else {
            self.avg_doc_len =
                self.doc_lens.iter().sum::<usize>() as f32 / self.texts.len() as f32;
        }

        Ok(())
    }

    /// Save the database to a `.vlite` file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let flags = if self.has_vision {
            storage::FLAG_HAS_VISION
        } else {
            0
        };
        let state = storage::SavedState {
            dim: self.dim,
            flags,
            avg_doc_len: self.avg_doc_len,
            vectors: self.vectors.clone(),
            texts: self.texts.clone(),
            parents: self.parents.clone(),
            metadata: self.metadata.clone(),
            doc_lens: self.doc_lens.clone(),
            df: self.df.clone(),
        };
        storage::save(&state, path.as_ref())
    }

    /// Number of items in the database.
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Get an item's text by index.
    pub fn get_text(&self, index: usize) -> Option<&str> {
        self.texts.get(index).map(|s| s.as_str())
    }

    /// Get an item's parent text by index.
    pub fn get_parent(&self, index: usize) -> Option<&str> {
        self.parents.get(index).map(|s| s.as_str())
    }

    /// Get an item's metadata by index.
    pub fn get_metadata(&self, index: usize) -> Option<&Metadata> {
        self.metadata.get(index)
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }
}
