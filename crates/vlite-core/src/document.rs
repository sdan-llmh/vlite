use crate::metadata::Metadata;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Document {
    pub id: String,
    pub title: Option<String>,
    pub raw_text: String,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SegmentKind {
    Document,
    Section,
    Paragraph,
    Page,
    Region,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub id: String,
    pub document_id: String,
    pub parent_id: Option<String>,
    pub kind: SegmentKind,
    pub path: Vec<String>,
    pub text: String,
    pub metadata: Metadata,
    pub embedding: Vec<f32>,
    pub searchable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AddResult {
    pub document_id: String,
    pub segment_ids: Vec<String>,
    pub chunk_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SearchHit {
    pub document_id: String,
    pub segment_id: String,
    pub segment_kind: SegmentKind,
    pub path: Vec<String>,
    pub text: String,
    pub metadata: Metadata,
    pub score: f32,
    pub parent_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CollectionInfo {
    pub path: String,
    pub document_count: usize,
    pub segment_count: usize,
    pub chunking_strategy: String,
    pub retrieval_strategy: String,
}
