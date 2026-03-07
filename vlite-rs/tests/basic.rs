//! Basic integration tests for vlite-rs.
//! These test the chunking, storage, and extraction modules without requiring
//! ONNX model downloads (which would make CI slow).

use std::collections::HashMap;

// ============================================================================
// Chunking tests
// ============================================================================

#[test]
fn test_chunk_short_text() {
    let chunks = vlite::chunk::chunk("hello world", "doc", 1024, 4096, 200);
    assert_eq!(chunks.len(), 1);
    assert!(chunks[0].child_text.contains("hello world"));
    assert!(chunks[0].child_text.starts_with("From doc: "));
    assert_eq!(chunks[0].parent_text, "hello world");
}

#[test]
fn test_chunk_empty() {
    let chunks = vlite::chunk::chunk("", "doc", 1024, 4096, 200);
    assert!(chunks.is_empty());
}

#[test]
fn test_chunk_whitespace_only() {
    let chunks = vlite::chunk::chunk("   \n\n  \t  ", "doc", 1024, 4096, 200);
    assert!(chunks.is_empty());
}

#[test]
fn test_chunk_no_title() {
    let chunks = vlite::chunk::chunk("hello world", "", 1024, 4096, 200);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].child_text, "hello world");
}

#[test]
fn test_chunk_long_text_splits() {
    // Create a text that's definitely longer than 50 chars
    let text = (0..20)
        .map(|i| format!("Paragraph {i} about topic number {i} with extra content."))
        .collect::<Vec<_>>()
        .join("\n\n");

    let chunks = vlite::chunk::chunk(&text, "test", 60, 300, 0);
    assert!(chunks.len() > 1, "long text should be split into multiple chunks");

    // All chunks should have content
    for c in &chunks {
        assert!(!c.child_text.is_empty());
        assert!(!c.parent_text.is_empty());
    }
}

#[test]
fn test_chunk_contextual_header() {
    let text = "First part.\n\nSecond part.\n\nThird part.";
    let chunks = vlite::chunk::chunk(text, "My Document", 15, 100, 0);
    for c in &chunks {
        assert!(
            c.child_text.starts_with("From My Document: "),
            "chunk should have contextual header: {:?}",
            c.child_text
        );
    }
}

#[test]
fn test_chunk_parent_child_relationship() {
    let text = (0..30)
        .map(|i| format!("Sentence {i} about topic {i}."))
        .collect::<Vec<_>>()
        .join(" ");

    let chunks = vlite::chunk::chunk(&text, "", 50, 200, 0);

    // Parents should be larger than or equal to children
    for c in &chunks {
        assert!(
            c.parent_text.len() >= c.child_text.len().saturating_sub(20),
            "parent should be >= child in size"
        );
    }
}

// ============================================================================
// Storage tests
// ============================================================================

#[test]
fn test_storage_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.vlite");

    let state = vlite::storage::SavedState {
        dim: 4,
        flags: 0,
        avg_doc_len: 3.5,
        vectors: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        texts: vec!["hello world".into(), "foo bar baz".into()],
        parents: vec!["hello world parent".into(), "foo bar baz parent".into()],
        metadata: vec![HashMap::new(), {
            let mut m = HashMap::new();
            m.insert("key".into(), serde_json::json!("value"));
            m
        }],
        doc_lens: vec![2, 3],
        df: {
            let mut m = HashMap::new();
            m.insert("hello".into(), 1);
            m.insert("foo".into(), 1);
            m.insert("bar".into(), 1);
            m
        },
    };

    vlite::storage::save(&state, &path).unwrap();
    let loaded = vlite::storage::load(&path).unwrap();

    assert_eq!(loaded.dim, 4);
    assert_eq!(loaded.vectors, state.vectors);
    assert_eq!(loaded.texts, state.texts);
    assert_eq!(loaded.parents, state.parents);
    assert_eq!(loaded.doc_lens, state.doc_lens);
    assert_eq!(loaded.df.len(), 3);
    assert_eq!(loaded.avg_doc_len, 3.5);
    assert_eq!(loaded.flags, 0);

    // Metadata check
    assert!(loaded.metadata[0].is_empty());
    assert_eq!(loaded.metadata[1].get("key").unwrap(), "value");
}

#[test]
fn test_storage_empty_db() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.vlite");

    let state = vlite::storage::SavedState {
        dim: 384,
        flags: 0,
        avg_doc_len: 0.0,
        vectors: vec![],
        texts: vec![],
        parents: vec![],
        metadata: vec![],
        doc_lens: vec![],
        df: HashMap::new(),
    };

    vlite::storage::save(&state, &path).unwrap();
    let loaded = vlite::storage::load(&path).unwrap();

    assert_eq!(loaded.dim, 384);
    assert!(loaded.texts.is_empty());
    assert!(loaded.vectors.is_empty());
}

#[test]
fn test_storage_bad_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.vlite");
    std::fs::write(&path, b"NOPE1234").unwrap();

    let result = vlite::storage::load(&path);
    assert!(result.is_err());
}

// ============================================================================
// Extract tests
// ============================================================================

#[test]
fn test_extract_html_basic() {
    let html = "<html><body><h1>Title</h1><p>Hello &amp; world</p></body></html>";
    let text = vlite::extract_html(html);
    assert!(text.contains("Title"));
    assert!(text.contains("Hello & world"));
    assert!(!text.contains("<"));
}

#[test]
fn test_extract_html_script_removal() {
    let html = "<p>Before</p><script>alert('xss');</script><p>After</p>";
    let text = vlite::extract_html(html);
    assert!(text.contains("Before"));
    assert!(text.contains("After"));
    assert!(!text.contains("alert"));
}

#[test]
fn test_extract_html_style_removal() {
    let html = "<p>Text</p><style>body { color: red; }</style><p>More</p>";
    let text = vlite::extract_html(html);
    assert!(text.contains("Text"));
    assert!(text.contains("More"));
    assert!(!text.contains("color"));
}

#[test]
fn test_extract_html_entities() {
    let html = "&lt;code&gt; &amp; &quot;quoted&quot;";
    let text = vlite::extract_html(html);
    assert!(text.contains("<code>"));
    assert!(text.contains("& \"quoted\""));
}
