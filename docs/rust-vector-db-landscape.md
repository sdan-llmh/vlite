# Rust Vector Database Landscape

This note summarizes the current Rust-heavy landscape and explains which ideas VLite should adopt.

## Evaluation criteria

We care about systems that are:

- written in Rust or Rust-core
- simple to embed or operate
- credible for modern multimodal retrieval
- architecturally relevant for an extensible local-first design

## CoreNN

**Best lesson:** scale economics and single-node clarity.

Wilson Lin’s CoreNN is compelling because it takes the “vector DB” problem seriously as a systems problem:

- disk-native design
- single-node scale
- live updates
- strong recall focus
- embedded and single-binary ergonomics

### What to adopt

- local-first mindset
- disk-native scale path
- support for real upserts / deletes
- avoid unnecessary distributed complexity

### What not to copy blindly

- CoreNN is primarily an indexing and systems story, not a multimodal ingestion story
- it does not define the ideal “auto chunking” UX for rich documents

## Qdrant

**Best lesson:** robust vector primitives.

Qdrant is one of the strongest Rust-native production systems today because it has good operational primitives:

- payload filtering
- named vectors
- multivectors
- quantization
- practical ANN tuning

### What to adopt

- multiple embedding views per item
- metadata-aware retrieval
- explicit backend and index configuration

### What not to copy blindly

- named vectors alone do not solve multimodal retrieval
- vector DB primitives are necessary, but not the whole product

## LanceDB

**Best lesson:** multimodal developer experience.

LanceDB has the strongest story around raw multimodal artifacts:

- embedded feel
- local workflows
- columnar data model
- multimodal storage
- late interaction support

### What to adopt

- store more than just vectors
- make local usage delightful
- support multimodal artifacts as first-class data

### What not to copy blindly

- VLite should stay more opinionated and smaller in public surface area
- a lakehouse-style abstraction can become heavier than needed for a simple retrieval engine

## SurrealDB

**Best lesson:** unification.

SurrealDB is relevant because it shows how vectors, documents, graphs, and search can coexist in one system.

### What to adopt

- rich metadata and relationships matter
- graph-like links are useful for context expansion

### What not to copy blindly

- VLite does not need to become a broad multi-model database
- retrieval-first focus should remain intact

## Meilisearch

**Best lesson:** product ergonomics and hybrid search.

Meilisearch is a search engine, not a canonical vector database, but its hybrid-search direction is instructive.

### What to adopt

- hybrid search should feel normal, not exotic
- API clarity and DX matter as much as raw internals

### What not to copy blindly

- VLite should remain centered on retrieval pipelines rather than general site search

## Research influence beyond products

### DiskANN / FreshDiskANN

Teach the scale path:

- SSD-first architectures
- high recall on one machine
- live update strategies

### Contextual Retrieval

Teaches that chunk quality matters as much as vector math:

- isolated chunks lose meaning
- enriched chunks retrieve better

### Late Chunking

Teaches that chunking after full-document encoding can preserve context while staying efficient.

### ColPali / ColQwen

Teach that visually rich documents should sometimes be indexed as pages or images, not flattened into OCR text only.

### Vision-Guided Chunking

Teaches that rich PDFs, tables, and figures need multimodal chunking, not just token windows.

## VLite position

VLite should not try to be:

- a clone of CoreNN
- a clone of Qdrant
- a clone of LanceDB

It should instead be:

- **smaller than Qdrant**
- **more retrieval-opinionated than LanceDB**
- **more multimodal than CoreNN**
- **simpler to embed than a server-first system**

## Strategic choice

The right move is:

1. Rust core
2. embedded/local-first default
3. document graph data model
4. exact/simple backend first
5. pluggable scale path later

That choice preserves simplicity while leaving room for serious systems work later.
