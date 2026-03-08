use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use vlite_core::document::{AddResult, CollectionInfo, Document, SearchHit};
use vlite_core::metadata::Metadata;
use vlite_core::retrieval::{Retriever, SearchRequest};
use vlite_core::ExactVLite;

fn to_py_error(error: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn into_metadata(metadata: Option<HashMap<String, String>>) -> Metadata {
    metadata.unwrap_or_default().into_iter().collect()
}

#[pyclass]
#[derive(Clone)]
struct PyAddResult {
    #[pyo3(get)]
    document_id: String,
    #[pyo3(get)]
    segment_ids: Vec<String>,
    #[pyo3(get)]
    chunk_count: usize,
}

impl From<AddResult> for PyAddResult {
    fn from(value: AddResult) -> Self {
        Self {
            document_id: value.document_id,
            segment_ids: value.segment_ids,
            chunk_count: value.chunk_count,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct PySearchHit {
    #[pyo3(get)]
    document_id: String,
    #[pyo3(get)]
    segment_id: String,
    #[pyo3(get)]
    segment_kind: String,
    #[pyo3(get)]
    path: Vec<String>,
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    metadata: HashMap<String, String>,
    #[pyo3(get)]
    modality: String,
    #[pyo3(get)]
    embedding_views: Vec<String>,
    #[pyo3(get)]
    page_number: Option<usize>,
    #[pyo3(get)]
    region_kind: Option<String>,
    #[pyo3(get)]
    score: f32,
    #[pyo3(get)]
    parent_text: Option<String>,
}

impl From<SearchHit> for PySearchHit {
    fn from(value: SearchHit) -> Self {
        Self {
            document_id: value.document_id,
            segment_id: value.segment_id,
            segment_kind: format!("{:?}", value.segment_kind),
            path: value.path,
            text: value.text,
            metadata: value.metadata.into_iter().collect(),
            modality: format!("{:?}", value.modality),
            embedding_views: value
                .embedding_views
                .into_iter()
                .map(|view| format!("{view:?}"))
                .collect(),
            page_number: value.page_number,
            region_kind: value.region_kind,
            score: value.score,
            parent_text: value.parent_text,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct PyDocument {
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    title: Option<String>,
    #[pyo3(get)]
    raw_text: String,
    #[pyo3(get)]
    metadata: HashMap<String, String>,
    #[pyo3(get)]
    modality: String,
    #[pyo3(get)]
    source_uri: Option<String>,
}

impl From<Document> for PyDocument {
    fn from(value: Document) -> Self {
        Self {
            id: value.id,
            title: value.title,
            raw_text: value.raw_text,
            metadata: value.metadata.into_iter().collect(),
            modality: format!("{:?}", value.modality),
            source_uri: value.source_uri,
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct PyCollectionInfo {
    #[pyo3(get)]
    path: String,
    #[pyo3(get)]
    document_count: usize,
    #[pyo3(get)]
    segment_count: usize,
    #[pyo3(get)]
    chunking_strategy: String,
    #[pyo3(get)]
    retrieval_strategy: String,
}

impl From<CollectionInfo> for PyCollectionInfo {
    fn from(value: CollectionInfo) -> Self {
        Self {
            path: value.path,
            document_count: value.document_count,
            segment_count: value.segment_count,
            chunking_strategy: value.chunking_strategy,
            retrieval_strategy: value.retrieval_strategy,
        }
    }
}

#[pyclass]
struct RustVLite {
    inner: Mutex<ExactVLite>,
}

#[pymethods]
impl RustVLite {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        let inner = ExactVLite::open(path).map_err(to_py_error)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[pyo3(signature = (text, metadata=None, document_id=None))]
    fn add(
        &self,
        text: String,
        metadata: Option<HashMap<String, String>>,
        document_id: Option<String>,
    ) -> PyResult<PyAddResult> {
        let result = self
            .inner
            .lock()
            .expect("mutex poisoned")
            .add_text(text, into_metadata(metadata), document_id)
            .map_err(to_py_error)?;
        Ok(result.into())
    }

    #[pyo3(signature = (texts, metadata=None, document_ids=None))]
    fn add_many(
        &self,
        texts: Vec<String>,
        metadata: Option<HashMap<String, String>>,
        document_ids: Option<Vec<String>>,
    ) -> PyResult<Vec<PyAddResult>> {
        let results = self
            .inner
            .lock()
            .expect("mutex poisoned")
            .add_texts(texts, into_metadata(metadata), document_ids)
            .map_err(to_py_error)?;
        Ok(results.into_iter().map(Into::into).collect())
    }

    #[pyo3(signature = (pages, metadata=None, document_id=None, source_uri=None))]
    fn add_pdf_pages(
        &self,
        pages: Vec<String>,
        metadata: Option<HashMap<String, String>>,
        document_id: Option<String>,
        source_uri: Option<String>,
    ) -> PyResult<PyAddResult> {
        let result = self
            .inner
            .lock()
            .expect("mutex poisoned")
            .add_pdf(pages, into_metadata(metadata), document_id, source_uri)
            .map_err(to_py_error)?;
        Ok(result.into())
    }

    #[pyo3(signature = (source_uri, metadata=None, document_id=None, caption=None))]
    fn add_image(
        &self,
        source_uri: String,
        metadata: Option<HashMap<String, String>>,
        document_id: Option<String>,
        caption: Option<String>,
    ) -> PyResult<PyAddResult> {
        let result = self
            .inner
            .lock()
            .expect("mutex poisoned")
            .add_image(source_uri, into_metadata(metadata), document_id, caption)
            .map_err(to_py_error)?;
        Ok(result.into())
    }

    #[pyo3(signature = (query, top_k=5, where_=None))]
    fn search(
        &self,
        query: String,
        top_k: usize,
        where_: Option<HashMap<String, String>>,
    ) -> PyResult<Vec<PySearchHit>> {
        let request = SearchRequest {
            query,
            top_k,
            where_filter: Some(into_metadata(where_)).filter(|value| !value.is_empty()),
        };
        let results = self
            .inner
            .lock()
            .expect("mutex poisoned")
            .search(request)
            .map_err(to_py_error)?;
        Ok(results.into_iter().map(Into::into).collect())
    }

    #[pyo3(signature = (ids=None, where_=None))]
    fn get(
        &self,
        ids: Option<Vec<String>>,
        where_: Option<HashMap<String, String>>,
    ) -> Vec<PyDocument> {
        let filter = into_metadata(where_);
        self.inner
            .lock()
            .expect("mutex poisoned")
            .get_documents(ids.as_deref(), Some(&filter).filter(|value| !value.is_empty()))
            .into_iter()
            .map(Into::into)
            .collect()
    }

    fn delete(&self, ids: Vec<String>) -> PyResult<usize> {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .delete_documents(&ids)
            .map_err(to_py_error)
    }

    fn info(&self) -> PyCollectionInfo {
        self.inner.lock().expect("mutex poisoned").info().into()
    }

    fn compact(&self) -> PyResult<()> {
        self.inner
            .lock()
            .expect("mutex poisoned")
            .compact()
            .map_err(to_py_error)
    }
}

#[pyfunction]
fn open_local(path: String) -> PyResult<RustVLite> {
    RustVLite::new(path)
}

#[pymodule]
fn vlite_py(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RustVLite>()?;
    module.add_class::<PyAddResult>()?;
    module.add_class::<PySearchHit>()?;
    module.add_class::<PyDocument>()?;
    module.add_class::<PyCollectionInfo>()?;
    module.add_function(wrap_pyfunction!(open_local, module)?)?;
    Ok(())
}
