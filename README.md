# vlite

**a simple and blazing fast vector database**

> *The three things that actually move the needle in retrieval are boring.* — vlite v2 design philosophy

vlite v2 is a production-grade hybrid search engine in ~500 lines of Rust. Built on [USearch](https://github.com/unum-cloud/usearch) (HNSW + [SimSIMD](https://github.com/ashvardanian/SimSIMD)) for vector search, with built-in BM25 for keyword search and RRF fusion for hybrid retrieval.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  INGEST                                                     │
│                                                             │
│  "long document about attention mechanisms..."              │
│    ↓                                                        │
│  Split into parent chunks (1024 tokens — returned to user)  │
│    ↓                                                        │
│  Split parents into child chunks (256 tokens — embedded)    │
│    ↓                                                        │
│  Index children ──→ USearch HNSW + BM25 inverted index      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  SEARCH  (three-stage production pipeline)                  │
│                                                             │
│  Stage 1: RETRIEVE — HNSW vector search + BM25 keyword     │
│  Stage 2: FUSE    — Reciprocal Rank Fusion (zero calibrate) │
│  Stage 3: RERANK  — Cross-encoder reranking (+33% accuracy) │
│    ↓                                                        │
│  Return parent chunks (big context, deduplicated)           │
└─────────────────────────────────────────────────────────────┘
```

## Features

- 🔥 **HNSW vector search** via USearch with SimSIMD acceleration (10x faster than FAISS)
- 📝 **BM25 keyword search** built-in — find exact terms, IDs, error codes
- 🔀 **Hybrid search** with RRF fusion — vector + BM25 in one query
- 🪆 **Parent-child chunking** — embed small for precision, return big for context
- 💾 **Single-file persistence** — save/load your database
- 🎯 **BYOE** (Bring Your Own Embeddings) — use any embedding model
- ⚡ **~500 lines** of Rust — the whole thing fits in your head

## Quick Start (Rust)

```rust
use vlite::VLite;
use std::collections::HashMap;

let mut db = VLite::open("my_db").unwrap();

// Add documents with pre-computed vectors (BYOE)
let v1 = vec![1.0, 0.0, 0.0, 0.0];
let v2 = vec![0.0, 1.0, 0.0, 0.0];
db.add_with_vector(&v1, "cats are wonderful pets", HashMap::new()).unwrap();
db.add_with_vector(&v2, "dogs are loyal companions", HashMap::new()).unwrap();

// Vector similarity search
let results = db.search_vector(&[0.9, 0.1, 0.0, 0.0], 5).unwrap();

// BM25 keyword search
let results = db.search_bm25("wonderful pets", 5).unwrap();

// Hybrid search (vector + BM25 fused with RRF)
let results = db.search_hybrid("cats", &[0.9, 0.1, 0.0, 0.0], 5).unwrap();

// Persistence
db.save().unwrap();
```

## How It Works

### Parent-Child Chunking

When you add text, vlite splits it into a two-level hierarchy:

- **Parent chunks** (1024 tokens): Returned to the user — large enough for context
- **Child chunks** (256 tokens): Embedded and indexed — small enough for precise matching

This is the standard production RAG pattern: embed small for precision, return big for context.

### Three-Stage Search Pipeline

1. **Retrieve**: BM25 returns top-100 keyword matches. HNSW returns top-100 vector matches. Fast, high recall.
2. **Fuse**: Reciprocal Rank Fusion merges both lists. Zero calibration needed — it works on ranks, not scores.
3. **Rerank** (optional): Cross-encoder model scores the top candidates for +33% accuracy.

### Storage

Two files per database:
- `my_db.usearch` — USearch HNSW index (vectors + graph)
- `my_db.vtx` — vlite data (texts, metadata, BM25 index)

## Configuration

```rust
use vlite::Config;

let config = Config {
    dimensions: 384,         // embedding dimension (0 = auto-detect)
    child_chunk_size: 256,   // tokens per child chunk
    parent_chunk_size: 1024, // tokens per parent chunk
    chunk_overlap: 25,       // overlap between children
    retrieval_k: 100,        // candidates per retrieval source
    connectivity: 16,        // HNSW graph connectivity
    expansion_add: 128,      // HNSW build quality
    expansion_search: 64,    // HNSW search quality
    ..Config::default()
};

let db = VLite::open_with_config("my_db", config).unwrap();
```

## Design Principles

Inspired by [micrograd](https://github.com/karpathy/micrograd), [USearch](https://github.com/unum-cloud/usearch), and [SimSIMD](https://github.com/ashvardanian/SimSIMD):

1. **Use primitives, don't reimplement them.** USearch does HNSW + SimSIMD + quantization. We build the magic layer on top.
2. **The three things that matter.** Hybrid search (BM25 + vectors), reranking (cross-encoder), parent-child chunks. Everything else is marginal.
3. **One file.** The entire database is `lib.rs`. ~500 lines. Fits in your head.
4. **BYOE first.** Works with any embedding model. Built-in ONNX embedding is optional (`embed` feature).

## vlite v1 (Python)

The original Python vlite is still available in the `vlite/` directory. See the [v1 docs](docs.md) for the Python API.

## License

AGPL-3.0 License
