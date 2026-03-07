# vlite Documentation

This repository is in the middle of a transition from the legacy Python implementation to a new **Rust-core VLite** architecture.

## What changed

The old documentation described VLite as a finished Python-first vector store. The new direction is different:

- Rust on the systems path
- embedded/local-first by default
- document graph storage instead of anonymous flat chunks
- adaptive chunking for modern RAG
- multimodal retrieval foundations

## Design documents

Start here:

- [`docs/next-gen-rust-vector-db-design.md`](docs/next-gen-rust-vector-db-design.md)
- [`docs/rust-vector-db-landscape.md`](docs/rust-vector-db-landscape.md)
- [`docs/chunking-retrieval-strategy.md`](docs/chunking-retrieval-strategy.md)

## Legacy implementation

The current `vlite/` Python package still exists in this repository and remains the compatibility baseline while the Rust-core implementation is built out.

That legacy implementation provides:

- a local file-based collection model
- text and document ingestion helpers
- retrieval over embedded chunks
- a FastAPI wrapper

It is useful as a reference and migration target, but it is **not** the long-term architecture.

## Target API shape

The future VLite API is intentionally small:

```python
from vlite import open

db = open("knowledge.vlite")
db.add("hello world", metadata={"source": "notes"})
results = db.search("hello")
```

Expected core operations:

- `open(path, config=None)`
- `add(input, metadata=None)`
- `search(query, top_k=5, where=None)`
- `get(ids=None, where=None)`
- `delete(ids)`
- `info()`

## Chunking direction

The new default is **structural auto chunking**, not fixed-size slices.

Examples:

- markdown → headings + paragraphs
- webpages → semantic blocks + DOM structure
- PDFs → page/region aware segmentation
- rich documents → optional page-as-image or multimodal retrieval path

## Retrieval direction

The first Rust retrieval backend will remain intentionally simple:

- exact dense search
- metadata filtering
- parent-child context expansion

That simple baseline will leave room for:

- hybrid lexical retrieval
- reranking
- multivector late interaction
- disk-native ANN backends later

## Migration policy

The migration is being staged carefully:

1. design docs
2. Rust workspace scaffold
3. exact-search Rust MVP
4. Python bindings
5. multimodal/document retrieval foundations

This keeps the repo honest and avoids making claims the code does not yet support.