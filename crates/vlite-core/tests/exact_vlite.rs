use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use vlite_core::metadata::Metadata;
use vlite_core::retrieval::{Retriever, SearchRequest};
use vlite_core::ExactVLite;

fn metadata(pairs: &[(&str, &str)]) -> Metadata {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

fn temp_path(name: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("vlite-core-{name}-{nanos}"));
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn structural_chunks_preserve_heading_path_and_parent_context() {
    let path = temp_path("structure");
    let mut db = ExactVLite::open(&path).unwrap();

    db.add_text(
        "# Intro\nRust makes systems programming pleasant.\n\n# Retrieval\nHybrid retrieval combines lexical and dense search.",
        metadata(&[("source", "notes")]),
        Some("doc-1".into()),
    )
    .unwrap();

    let hits = db
        .search(SearchRequest {
            query: "hybrid retrieval".into(),
            top_k: 1,
            where_filter: None,
        })
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document_id, "doc-1");
    assert_eq!(hits[0].path, vec!["Retrieval".to_string()]);
    assert!(hits[0].text.contains("Hybrid retrieval"));
    assert!(hits[0]
        .parent_text
        .as_ref()
        .is_some_and(|text| text.contains("Rust makes systems programming pleasant")));
}

#[test]
fn metadata_filters_limit_search_results() {
    let path = temp_path("filters");
    let mut db = ExactVLite::open(&path).unwrap();

    db.add_text(
        "Rust retrieval for alpha documents",
        metadata(&[("tenant", "alpha")]),
        Some("alpha".into()),
    )
    .unwrap();
    db.add_text(
        "Rust retrieval for beta documents",
        metadata(&[("tenant", "beta")]),
        Some("beta".into()),
    )
    .unwrap();

    let hits = db
        .search(SearchRequest {
            query: "beta retrieval".into(),
            top_k: 5,
            where_filter: Some(metadata(&[("tenant", "beta")])),
        })
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document_id, "beta");
    assert_eq!(hits[0].metadata.get("tenant"), Some(&"beta".to_string()));
}

#[test]
fn collections_round_trip_through_json_store() {
    let path = temp_path("roundtrip");

    {
        let mut db = ExactVLite::open(&path).unwrap();
        db.add_text(
            "# Storage\nPersistence should survive reopen.",
            metadata(&[("kind", "test")]),
            Some("persisted".into()),
        )
        .unwrap();
    }

    let db = ExactVLite::open(&path).unwrap();
    let info = db.info();
    assert_eq!(info.document_count, 1);
    assert_eq!(info.segment_count, 1);

    let hits = db
        .search(SearchRequest {
            query: "survive reopen".into(),
            top_k: 1,
            where_filter: Some(metadata(&[("kind", "test")])),
        })
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].document_id, "persisted");
}
