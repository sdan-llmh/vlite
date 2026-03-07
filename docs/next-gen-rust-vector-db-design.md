# Next-Generation VLite Design

VLite is evolving from a small Python vector store into a **Rust-core, local-first vector database** designed for the 2026 retrieval stack:

- **simple enough to feel like SQLite**
- **fast enough to scale beyond brute force**
- **multimodal enough to handle modern RAG**
- **extensible enough to grow into disk-native ANN**

This document describes the target architecture and why the design is intentionally opinionated.

## Design principles

### 1. Local-first and embeddable

The default deployment target is **in-process**, not “bring up a cluster”. A user should be able to:

```python
from vlite import open

db = open("knowledge.vlite")
db.add("hello world")
db.search("hello")
```

An optional server layer can exist later, but the product should feel embedded first.

### 2. Rust core, thin Python layer

The critical path moves into Rust:

- document storage
- chunk planning
- filtering
- retrieval
- persistence

Python remains important for:

- notebooks
- existing integrations
- user ergonomics
- glue around external embedding models

### 3. Document graph over flat chunks

Modern retrieval quality comes less from blind 512-token windows and more from preserving structure.

VLite should store:

- **Documents**
- **Segments** (sections, paragraphs, tables, figures, pages, regions)
- **Embedding views** (dense vector, lexical view, multivector view)
- **Parent-child relationships**

This lets retrieval return relevant leaf segments while still expanding to useful parent context.

### 4. Adaptive chunking

“Auto chunking” should not mean fixed-size slicing. The default strategy should be modality-aware:

- plain text / markdown → header-aware structural chunks
- HTML / webpages → DOM-aware structural chunks
- PDFs → page and region aware chunks
- visually rich documents → optional page-as-image retrieval path

Advanced strategies should remain available:

- **contextual retrieval** for chunk enrichment
- **late chunking** for long-context embedding pipelines
- **vision-guided chunking** for difficult PDFs and forms

### 5. Hybrid retrieval by default

Dense-only retrieval is no longer enough. The default planner should support:

- dense retrieval
- lexical signals
- metadata filtering
- optional reranking

Late-interaction / multivector retrieval should be a second stage for harder queries, not a mandatory cost for every search.

## Target architecture

## Public API

The public API should stay intentionally small:

- `open(path, config=None)`
- `add(input, metadata=None)`
- `search(query, top_k=5, where=None)`
- `get(ids=None, where=None)`
- `delete(ids)`
- `info()`
- `compact()`

Optional strategy knobs may be exposed without overwhelming the default path:

- `chunking="auto" | "structural" | "late" | "vision"`
- `retrieval="hybrid" | "dense" | "late_interaction"`
- `store_raw=True/False`

## Internal layers

### Ingest

Responsible for:

- detecting modality
- extracting normalized content
- preserving provenance
- producing segment candidates

### Chunking

Responsible for:

- structural segmentation
- hierarchy retention
- optional contextual enrichment
- modality-aware boundary decisions

### Storage

Responsible for:

- manifest and metadata persistence
- document / segment tables
- embedding view persistence
- raw artifact references

### Retrieval

Responsible for:

- search planning
- index backend selection
- filtering
- parent expansion
- reranking

## Index backends

The first backend should be deliberately boring:

- **exact / SIMD-friendly dense search**

Why start here:

- easy to reason about
- deterministic
- low maintenance
- good enough for small and medium collections
- compatible with the “simplicity first” product position

Future backends should be pluggable:

- HNSW-like backend
- disk-native ANN backend inspired by DiskANN / FreshDiskANN
- multivector late-interaction backend

## Why this direction

This architecture combines the best ideas from the current Rust vector landscape without copying any one system blindly:

- **CoreNN** contributes disk-native thinking and a strong single-node philosophy
- **Qdrant** contributes named vector and filtering ideas
- **LanceDB** contributes embedded multimodal ergonomics
- **Anthropic / Jina / ColPali / vision-guided chunking** contribute chunking and retrieval quality lessons

The result should feel:

- simpler than a production cluster-first vector DB
- more retrieval-aware than a generic table store with embeddings
- more multimodal than a text-only ANN library

## First implementation boundary

The first credible milestone is not “build all of DiskANN in a weekend”.

It is:

1. Rust workspace
2. canonical document/segment schema
3. exact-search backend
4. structural chunker
5. Python bindings
6. tests proving deterministic behavior

That gives VLite a sound foundation without over-claiming.
