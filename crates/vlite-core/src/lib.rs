pub mod chunking;
pub mod config;
pub mod document;
pub mod index;
pub mod metadata;
pub mod retrieval;
pub mod storage;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Context;
use chunking::{Chunker, StructuralChunker};
use config::VLiteConfig;
use document::{AddResult, CollectionInfo, Document, SearchHit, Segment, SegmentKind};
use index::{ExactIndex, IndexBackend};
use metadata::Metadata;
use retrieval::{EmbedderAdapter, HashedEmbedder, Retriever, SearchRequest};
use storage::{CollectionSnapshot, JsonStore, Store};
use uuid::Uuid;

pub struct ExactVLite {
    config: VLiteConfig,
    chunker: StructuralChunker,
    embedder: HashedEmbedder,
    index: ExactIndex,
    store: JsonStore,
    documents: BTreeMap<String, Document>,
    segments: BTreeMap<String, Segment>,
}

impl ExactVLite {
    pub fn open(path: impl Into<PathBuf>) -> anyhow::Result<Self> {
        Self::open_with_config(path, VLiteConfig::default())
    }

    pub fn open_with_config(path: impl Into<PathBuf>, config: VLiteConfig) -> anyhow::Result<Self> {
        let store = JsonStore::open(path)?;
        let snapshot = store.load()?;
        let effective_config = snapshot.config.unwrap_or(config);

        let documents = snapshot
            .documents
            .into_iter()
            .map(|document| (document.id.clone(), document))
            .collect::<BTreeMap<_, _>>();
        let segments = snapshot
            .segments
            .into_iter()
            .map(|segment| (segment.id.clone(), segment))
            .collect::<BTreeMap<_, _>>();

        Ok(Self {
            embedder: HashedEmbedder::new(effective_config.embedding_dimensions),
            config: effective_config,
            chunker: StructuralChunker,
            index: ExactIndex,
            store,
            documents,
            segments,
        })
    }

    pub fn add_text(
        &mut self,
        text: impl Into<String>,
        metadata: Metadata,
        document_id: Option<String>,
    ) -> anyhow::Result<AddResult> {
        let raw_text = text.into();
        let document_id = document_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let title = raw_text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .map(|line| line.trim_start_matches('#').trim().to_string())
            .filter(|line| !line.is_empty());

        let document = Document {
            id: document_id.clone(),
            title,
            raw_text: raw_text.clone(),
            metadata: metadata.clone(),
        };

        let root_segment_id = format!("{document_id}:root");
        let root_segment = Segment {
            id: root_segment_id.clone(),
            document_id: document_id.clone(),
            parent_id: None,
            kind: SegmentKind::Document,
            path: Vec::new(),
            text: raw_text.clone(),
            metadata: metadata.clone(),
            embedding: self.embedder.embed(&raw_text),
            searchable: false,
        };

        let drafts = self.chunker.chunk_document(&document);
        let segment_ids = drafts
            .iter()
            .enumerate()
            .map(|(index, draft)| {
                let segment_id = format!("{document_id}:seg:{index:04}");
                let segment = Segment {
                    id: segment_id.clone(),
                    document_id: document_id.clone(),
                    parent_id: Some(root_segment_id.clone()),
                    kind: draft.kind.clone(),
                    path: draft.path.clone(),
                    text: draft.text.clone(),
                    metadata: metadata.clone(),
                    embedding: self.embedder.embed(&draft.text),
                    searchable: true,
                };
                self.segments.insert(segment_id.clone(), segment);
                segment_id
            })
            .collect::<Vec<_>>();

        self.documents.insert(document_id.clone(), document);
        self.segments.insert(root_segment_id, root_segment);
        self.persist()?;

        Ok(AddResult {
            document_id,
            segment_ids: segment_ids.clone(),
            chunk_count: segment_ids.len(),
        })
    }

    pub fn get_documents(
        &self,
        ids: Option<&[String]>,
        where_filter: Option<&Metadata>,
    ) -> Vec<Document> {
        self.documents
            .values()
            .filter(|document| {
                ids.map_or(true, |ids| ids.iter().any(|id| id == &document.id))
                    && metadata::matches_filter(&document.metadata, where_filter)
            })
            .cloned()
            .collect()
    }

    pub fn delete_documents(&mut self, ids: &[String]) -> anyhow::Result<usize> {
        let existing = ids
            .iter()
            .filter(|id| self.documents.contains_key((*id).as_str()))
            .cloned()
            .collect::<Vec<_>>();

        for id in &existing {
            self.documents.remove(id);
            self.segments.retain(|_, segment| &segment.document_id != id);
        }

        self.persist()?;
        Ok(existing.len())
    }

    pub fn info(&self) -> CollectionInfo {
        CollectionInfo {
            path: self.store.root().display().to_string(),
            document_count: self.documents.len(),
            segment_count: self.segments.values().filter(|segment| segment.searchable).count(),
            chunking_strategy: format!("{:?}", self.config.chunking_strategy),
            retrieval_strategy: format!("{:?}", self.config.retrieval_strategy),
        }
    }

    pub fn compact(&self) -> anyhow::Result<()> {
        self.persist()
    }

    fn persist(&self) -> anyhow::Result<()> {
        let snapshot = CollectionSnapshot {
            config: Some(self.config.clone()),
            documents: self.documents.values().cloned().collect(),
            segments: self.segments.values().cloned().collect(),
        };
        self.store
            .save(&snapshot)
            .context("failed to persist VLite collection")
    }
}

impl Retriever for ExactVLite {
    fn search(&self, request: SearchRequest) -> anyhow::Result<Vec<SearchHit>> {
        let query_embedding = self.embedder.embed(&request.query);
        let searchable_segments = self.segments.values().cloned().collect::<Vec<_>>();
        let hits = self.index.search(
            &query_embedding,
            &searchable_segments,
            request.top_k,
            request.where_filter.as_ref(),
        );

        Ok(hits
            .into_iter()
            .filter_map(|(idx, score)| searchable_segments.get(idx).cloned().map(|segment| (segment, score)))
            .map(|(segment, score)| SearchHit {
                document_id: segment.document_id.clone(),
                segment_id: segment.id.clone(),
                segment_kind: segment.kind.clone(),
                path: segment.path.clone(),
                text: segment.text.clone(),
                metadata: segment.metadata.clone(),
                score,
                parent_text: segment
                    .parent_id
                    .as_ref()
                    .and_then(|parent_id| self.segments.get(parent_id))
                    .map(|parent| parent.text.clone()),
            })
            .collect())
    }
}
