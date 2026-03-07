use crate::chunking::Chunker;
use crate::document::{Document, EmbeddingViewKind, Modality, SegmentKind};
use crate::ingest::{IngestedDocument, IngestedSegment};
use crate::metadata::Metadata;

pub fn ingest_pdf_pages(
    document_id: String,
    pages: Vec<String>,
    metadata: Metadata,
    source_uri: Option<String>,
    chunker: &impl Chunker,
) -> IngestedDocument {
    let raw_text = pages.join("\n\n");
    let title = pages
        .iter()
        .flat_map(|page| page.lines())
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToString::to_string);

    let document = Document {
        id: document_id,
        title,
        raw_text,
        metadata: metadata.clone(),
        modality: Modality::Pdf,
        source_uri,
    };

    let mut segments = Vec::new();

    for (page_index, page_text) in pages.into_iter().enumerate() {
        let page_number = page_index + 1;
        let page_key = format!("page:{page_number:04}");
        let page_label = format!("Page {page_number}");

        segments.push(IngestedSegment {
            key: page_key.clone(),
            parent_key: Some("root".into()),
            kind: SegmentKind::Page,
            path: vec![page_label.clone()],
            text: page_text.clone(),
            metadata: metadata.clone(),
            modality: Modality::Pdf,
            embedding_views: vec![EmbeddingViewKind::Lexical],
            page_number: Some(page_number),
            region_kind: Some("page".into()),
            searchable: false,
        });

        let page_document = Document {
            id: format!("{}:{page_key}", document.id),
            title: Some(page_label.clone()),
            raw_text: page_text,
            metadata: metadata.clone(),
            modality: Modality::Pdf,
            source_uri: document.source_uri.clone(),
        };

        for (segment_index, draft) in chunker.chunk_document(&page_document).into_iter().enumerate() {
            let mut path = vec![page_label.clone()];
            path.extend(draft.path);

            segments.push(IngestedSegment {
                key: format!("{page_key}:seg:{segment_index:04}"),
                parent_key: Some(page_key.clone()),
                kind: draft.kind,
                path,
                text: draft.text,
                metadata: metadata.clone(),
                modality: Modality::Pdf,
                embedding_views: vec![EmbeddingViewKind::Dense, EmbeddingViewKind::Lexical],
                page_number: Some(page_number),
                region_kind: Some("paragraph".into()),
                searchable: true,
            });
        }
    }

    IngestedDocument { document, segments }
}
