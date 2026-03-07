use crate::document::{Document, EmbeddingViewKind, Modality, SegmentKind};
use crate::ingest::{IngestedDocument, IngestedSegment};
use crate::metadata::Metadata;

pub fn ingest_image_reference(
    document_id: String,
    source_uri: String,
    metadata: Metadata,
    caption: Option<String>,
) -> IngestedDocument {
    let raw_text = caption.clone().unwrap_or_default();
    let document = Document {
        id: document_id,
        title: Some(source_uri.clone()),
        raw_text: raw_text.clone(),
        metadata: metadata.clone(),
        modality: Modality::Image,
        source_uri: Some(source_uri),
    };

    let mut segments = Vec::new();
    if !raw_text.trim().is_empty() {
        segments.push(IngestedSegment {
            key: "caption:0000".into(),
            parent_key: Some("root".into()),
            kind: SegmentKind::Region,
            path: vec!["Image".into(), "Caption".into()],
            text: raw_text,
            metadata,
            modality: Modality::Image,
            embedding_views: vec![
                EmbeddingViewKind::Dense,
                EmbeddingViewKind::Lexical,
                EmbeddingViewKind::PageImage,
            ],
            page_number: None,
            region_kind: Some("caption".into()),
            searchable: true,
        });
    }

    IngestedDocument { document, segments }
}
