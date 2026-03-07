use crate::chunking::Chunker;
use crate::document::{Document, EmbeddingViewKind, Modality};
use crate::ingest::{IngestedDocument, IngestedSegment};
use crate::metadata::Metadata;

pub fn ingest_text_document(
    document_id: String,
    text: String,
    metadata: Metadata,
    source_uri: Option<String>,
    chunker: &impl Chunker,
) -> IngestedDocument {
    let title = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| line.trim_start_matches('#').trim().to_string())
        .filter(|line| !line.is_empty());

    let document = Document {
        id: document_id,
        title,
        raw_text: text.clone(),
        metadata: metadata.clone(),
        modality: Modality::Text,
        source_uri,
    };

    let segments = chunker
        .chunk_document(&document)
        .into_iter()
        .enumerate()
        .map(|(index, draft)| IngestedSegment {
            key: format!("seg:{index:04}"),
            parent_key: Some("root".into()),
            kind: draft.kind,
            path: draft.path,
            text: draft.text,
            metadata: metadata.clone(),
            modality: Modality::Text,
            embedding_views: vec![EmbeddingViewKind::Dense, EmbeddingViewKind::Lexical],
            page_number: None,
            region_kind: Some("paragraph".into()),
            searchable: true,
        })
        .collect();

    IngestedDocument { document, segments }
}
