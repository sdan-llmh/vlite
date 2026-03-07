# vlite

VLite is transitioning into a **Rust-core, local-first vector database** for the 2026 retrieval stack.

The goal is simple:

- **SQLite-like ergonomics**
- **Rust speed on the hot path**
- **multimodal ingestion**
- **modern auto chunking**
- **an extensible path from exact search to disk-native ANN**

The current repository still contains the original Python implementation. This branch starts the migration toward a next-generation architecture.

## Why VLite is changing

The existing package proved out a useful developer experience:

- no server required
- easy local persistence
- simple Python API

But modern retrieval systems need more than a flat embedding store:

- better document structure preservation
- multimodal ingestion
- stronger metadata and provenance
- higher-quality chunking defaults
- a Rust systems core for long-term performance and extensibility

## Design direction

The new direction combines ideas from the current state of the art:

- **CoreNN / DiskANN / FreshDiskANN** for single-node scale and disk-native thinking
- **Qdrant** for named vectors, filtering, and practical vector primitives
- **LanceDB** for embedded multimodal ergonomics
- **Contextual Retrieval, Late Chunking, ColPali, and vision-guided chunking** for higher-quality modern RAG

The product stance is intentionally opinionated:

- embedded first
- small public API
- structural chunking by default
- hybrid retrieval by default
- parent-child document context preserved

## Planned architecture

VLite is being redesigned around:

- a **Rust core** for storage, chunk planning, filtering, and retrieval
- a **thin Python layer** for compatibility and notebook ergonomics
- a **document graph** model instead of anonymous flat chunks
- **adaptive chunking** across text, PDFs, and richer multimodal inputs
- a **simple exact-search backend first**, with room for future ANN backends

See the design docs for details:

- [`docs/next-gen-rust-vector-db-design.md`](docs/next-gen-rust-vector-db-design.md)
- [`docs/rust-vector-db-landscape.md`](docs/rust-vector-db-landscape.md)
- [`docs/chunking-retrieval-strategy.md`](docs/chunking-retrieval-strategy.md)

## Current status

This repository currently contains:

- the legacy Python implementation under `vlite/`
- new design documentation for the Rust-core migration
- an in-progress Rust workspace scaffold on this branch

During the migration, the old Python implementation remains useful as a compatibility layer and reference point, but it is **not** the final architecture.

## Legacy Python usage

The existing Python API still looks like:

```python
from vlite import VLite

db = VLite()
db.add("hello world", metadata={"artist": "adele"})
results = db.retrieve("hello")
print(results)
```

That API will be preserved or closely mirrored where practical, but the implementation beneath it is moving toward Rust.

## Principles

VLite should eventually feel like:

- one local file or directory
- one object
- one obvious `add(...)`
- one obvious `search(...)`
- strong defaults
- escape hatches for advanced retrieval strategies

## License

AGPL-3.0 License