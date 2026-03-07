# Chunking and Retrieval Strategy

This document defines what “auto chunking” should mean in VLite.

## The problem with naive chunking

Traditional RAG systems often:

- split every document into fixed windows
- embed each chunk independently
- retrieve isolated fragments

That is simple, but it breaks down on modern workloads:

- headings lose their body
- pronouns lose their referents
- tables are shredded
- figures become invisible
- page layout information disappears

The result is often worse retrieval even if the vector index itself is excellent.

## VLite default: structural auto chunking

The default should be a **structural chunker**.

Its job is to preserve meaning while staying cheap.

### Plain text / markdown

Preferred boundaries:

1. title / heading changes
2. paragraph boundaries
3. sentence boundaries
4. token budget fallback

The chunker should preserve:

- heading ancestry
- local section path
- source offsets

### HTML and webpages

Preferred boundaries:

1. semantic block elements
2. headings and subtrees
3. paragraph groups
4. fallback token budget

The chunker should avoid collapsing the whole DOM into plain text too early.

### PDFs and office documents

Preferred boundaries:

1. document
2. page
3. detected region
4. logical content group

Important region types:

- paragraph blocks
- tables
- figures
- captions
- footnotes
- headers / footers

For rich PDFs, the system should preserve both:

- extracted text view
- visual provenance

## Advanced chunking modes

## Contextual chunking

Inspired by contextual retrieval:

- each chunk is enriched with short context derived from the parent document
- useful for ambiguous chunks such as “it”, “the company”, “the following table”

Tradeoff:

- higher preprocessing cost
- often better retrieval quality

## Late chunking

Inspired by long-context embedding pipelines:

- encode a longer parent window first
- apply chunk boundaries after contextualized token representations exist

Tradeoff:

- requires long-context embedders
- cheaper than LLM-based contextual chunking
- often better than naive per-chunk embeddings

## Vision-guided chunking

For visually rich documents:

- use page images and layout-aware segmentation
- keep tables, charts, and page-local structure intact
- allow page-as-image retrieval path

This should be selective, not the default for every plain-text corpus.

## Retrieval planner

The retrieval planner should combine multiple signals.

### Stage 1: candidate generation

Default candidate generation should support:

- dense vector similarity
- lexical matching
- metadata filters

### Stage 2: optional reranking

For difficult or multimodal queries:

- contextual preference
- parent-child coherence scoring
- late interaction / multivector reranking

### Stage 3: context assembly

Never return naked fragments when a richer parent exists.

The planner should:

- retrieve the best leaf segments
- expand to parent section or page when appropriate
- preserve provenance in the result

## Result shape

Search results should eventually include:

- `document_id`
- `segment_id`
- `text`
- `segment_kind`
- `path` or heading ancestry
- `metadata`
- `score`
- optional `parent_context`
- optional provenance like page number or byte offsets

## Operating rule

VLite should optimize first for:

1. retrieval quality
2. simplicity
3. speed

Not the other way around.

Speed matters, but “fast wrong chunks” is not a good product.
