pub mod chunking;
pub mod config;
pub mod document;
pub mod ingest;
pub mod index;
pub mod metadata;
pub mod retrieval;
pub mod storage;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::Context;
use chunking::StructuralChunker;
use config::VLiteConfig;
use document::{AddResult, CollectionInfo, Document, EmbeddingViewKind, SearchHit, Segment, SegmentKind};
use ingest::image::ingest_image_reference;
use ingest::pdf::ingest_pdf_pages;
use ingest::text::ingest_text_document;
use ingest::{IngestedDocument, IngestedSegment};
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
        let document_id = document_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let ingested = ingest_text_document(
            document_id,
            text.into(),
            metadata,
            None,
            &self.chunker,
        );
        self.add_ingested_document(ingested, true)
    }

    pub fn add_texts(
        &mut self,
        texts: Vec<String>,
        metadata: Metadata,
        document_ids: Option<Vec<String>>,
    ) -> anyhow::Result<Vec<AddResult>> {
        let document_ids = document_ids.unwrap_or_default();
        let mut results = Vec::with_capacity(texts.len());

        for (index, text) in texts.into_iter().enumerate() {
            let document_id = document_ids
                .get(index)
                .cloned()
                .unwrap_or_else(|| Uuid::new_v4().to_string());
            let ingested =
                ingest_text_document(document_id, text, metadata.clone(), None, &self.chunker);
            let result = self.add_ingested_document(ingested, false)?;
            results.push(result);
        }

        self.persist()?;
        Ok(results)
    }

    pub fn add_pdf(
        &mut self,
        pages: Vec<String>,
        metadata: Metadata,
        document_id: Option<String>,
        source_uri: Option<String>,
    ) -> anyhow::Result<AddResult> {
        let document_id = document_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let ingested = ingest_pdf_pages(document_id, pages, metadata, source_uri, &self.chunker);
        self.add_ingested_document(ingested, true)
    }

    pub fn add_image(
        &mut self,
        source_uri: String,
        metadata: Metadata,
        document_id: Option<String>,
        caption: Option<String>,
    ) -> anyhow::Result<AddResult> {
        let document_id = document_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let ingested = ingest_image_reference(document_id, source_uri, metadata, caption);
        self.add_ingested_document(ingested, true)
    }

    fn add_ingested_document(
        &mut self,
        ingested: IngestedDocument,
        persist_now: bool,
    ) -> anyhow::Result<AddResult> {
        let document_id = ingested.document.id.clone();
        let root_segment_id = format!("{document_id}:root");
        let root_segment = Segment {
            id: root_segment_id.clone(),
            document_id: document_id.clone(),
            parent_id: None,
            kind: SegmentKind::Document,
            path: Vec::new(),
            text: ingested.document.raw_text.clone(),
            metadata: ingested.document.metadata.clone(),
            modality: ingested.document.modality.clone(),
            embedding_views: vec![EmbeddingViewKind::Dense, EmbeddingViewKind::Lexical],
            page_number: None,
            region_kind: Some("document".into()),
            embedding: self.embedder.embed(&ingested.document.raw_text),
            searchable: false,
        };

        let segment_ids = ingested
            .segments
            .into_iter()
            .map(|segment| self.materialize_segment(&document_id, &root_segment_id, segment))
            .collect::<Vec<_>>();

        self.documents
            .insert(document_id.clone(), ingested.document);
        self.segments.insert(root_segment_id, root_segment);
        if persist_now {
            self.persist()?;
        }

        Ok(AddResult {
            document_id,
            segment_ids: segment_ids.clone(),
            chunk_count: segment_ids.len(),
        })
    }

    fn materialize_segment(
        &mut self,
        document_id: &str,
        root_segment_id: &str,
        segment: IngestedSegment,
    ) -> String {
        let segment_id = format!("{document_id}:{}", segment.key);
        let parent_id = segment.parent_key.as_ref().map(|parent_key| {
            if parent_key == "root" {
                root_segment_id.to_string()
            } else {
                format!("{document_id}:{parent_key}")
            }
        });

        let record = Segment {
            id: segment_id.clone(),
            document_id: document_id.to_string(),
            parent_id,
            kind: segment.kind,
            path: segment.path,
            text: segment.text.clone(),
            metadata: segment.metadata,
            modality: segment.modality,
            embedding_views: segment.embedding_views,
            page_number: segment.page_number,
            region_kind: segment.region_kind,
            embedding: self.embedder.embed(&segment.text),
            searchable: segment.searchable,
        };
        self.segments.insert(segment_id.clone(), record);
        segment_id
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
                modality: segment.modality.clone(),
                embedding_views: segment.embedding_views.clone(),
                page_number: segment.page_number,
                region_kind: segment.region_kind.clone(),
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
