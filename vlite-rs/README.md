# vlite-rs

a simple and blazing fast vector database. ~700 lines of Rust.

## Features

- 🔥 **Hybrid search by default** — vector (SimSIMD cosine) + BM25 + Reciprocal Rank Fusion
- 🧠 **Embedding baked in** — ONNX Runtime inference, auto-downloads models from HuggingFace
- 📄 **Auto-chunking** — long text splits at natural boundaries with parent-child retrieval
- 🏷️ **Contextual headers** — prepends document context to chunks (Anthropic's 49% improvement)
- 🖼️ **Multimodal** — text + images via CLIP (optional)
- 📦 **Single file** — your `.vlite` file IS your database
- ⚡ **Fast** — SimSIMD SIMD-accelerated cosine, inline BM25, O(n) partial sort

## Quick Start

```rust
use vlite::VLite;

let mut db = VLite::new()?;
db.add("the mitochondria is the powerhouse of the cell", None)?;
db.add("photosynthesis converts sunlight into energy", None)?;
let results = db.search("biology energy", 3)?;

db.save("bio.vlite")?;
```

## How Search Works

Every `search()` call runs three stages:

1. **Vector search** — embed query → cosine similarity against all vectors via SimSIMD (200x faster than numpy)
2. **BM25 search** — keyword matching with TF-IDF scoring (catches exact matches like error codes)
3. **RRF fusion** — Reciprocal Rank Fusion merges both rankings without score normalization

This is the same architecture used by Elasticsearch, Vespa, and production RAG systems.

## How Chunking Works

When you `add()` text longer than the model's context window:

1. **Recursive split** — paragraphs → lines → sentences → words
2. **Contextual headers** — prepend "From {title}: " to each chunk
3. **Parent-child** — embed small chunks (precise), return large parents (contextual)
4. **Overlap** — adjacent chunks share boundary text to prevent information loss

## Multimodal

```rust
let mut db = VLite::with_clip()?;  // CLIP model for text + images
db.add("a cat sitting on a mat", None)?;
db.add_image(&jpeg_bytes, None)?;
db.search("photo of a cat", 5)?;  // finds both text AND images
```

## PDF Ingestion

```rust
let text = vlite::extract_pdf("paper.pdf")?;
db.add(&text, Some(json!({"source": "paper.pdf"})))?;  // auto-chunks
```

## License

AGPL-3.0
