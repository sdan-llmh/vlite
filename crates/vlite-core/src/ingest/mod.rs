pub mod image;
pub mod pdf;
pub mod text;

use crate::document::{Document, EmbeddingViewKind, Modality, SegmentKind};
use crate::metadata::Metadata;

#[derive(Debug, Clone)]
pub struct IngestedSegment {
    pub key: String,
    pub parent_key: Option<String>,
    pub kind: SegmentKind,
    pub path: Vec<String>,
    pub text: String,
    pub metadata: Metadata,
    pub modality: Modality,
    pub embedding_views: Vec<EmbeddingViewKind>,
    pub page_number: Option<usize>,
    pub region_kind: Option<String>,
    pub searchable: bool,
}

#[derive(Debug, Clone)]
pub struct IngestedDocument {
    pub document: Document,
    pub segments: Vec<IngestedSegment>,
}
