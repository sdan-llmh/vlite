def test_add_search_get_and_delete_round_trip(rust_db):
    added = rust_db.add(
        "# Intro\nRust databases can stay simple.\n\n# Search\nHybrid retrieval feels practical.",
        {"tenant": "alpha", "kind": "note"},
        "doc-alpha",
    )

    assert added.document_id == "doc-alpha"
    assert added.chunk_count == 2

    docs = rust_db.get(["doc-alpha"], {"tenant": "alpha"})
    assert len(docs) == 1
    assert docs[0].modality == "Text"

    hits = rust_db.search("hybrid retrieval", 2, {"tenant": "alpha"})
    assert len(hits) == 2
    assert hits[0].document_id == "doc-alpha"
    assert hits[0].path == ["Search"]
    assert hits[0].modality == "Text"
    assert "Dense" in hits[0].embedding_views
    assert hits[0].parent_text is not None

    info = rust_db.info()
    assert info.document_count == 1
    assert info.segment_count == 2

    deleted = rust_db.delete(["doc-alpha"])
    assert deleted == 1
    assert rust_db.get(["doc-alpha"], None) == []


def test_add_many_batches_persist_once(rust_db):
    results = rust_db.add_many(
        [
            "# One\nFirst doc",
            "# Two\nSecond doc",
            "# Three\nThird doc",
        ],
        {"batch": "yes"},
        ["doc-1", "doc-2", "doc-3"],
    )

    assert [result.document_id for result in results] == ["doc-1", "doc-2", "doc-3"]

    docs = rust_db.get(None, {"batch": "yes"})
    assert len(docs) == 3
    assert {doc.id for doc in docs} == {"doc-1", "doc-2", "doc-3"}


def test_pdf_ingestion_keeps_page_provenance(rust_db):
    added = rust_db.add_pdf_pages(
        [
            "# Page One\nRust retrieval on the first page.",
            "# Page Two\nTables and figures belong to page two.",
        ],
        {"source": "pdf"},
        "pdf-doc",
        "memory://guide.pdf",
    )

    assert added.document_id == "pdf-doc"
    assert added.chunk_count >= 2

    docs = rust_db.get(["pdf-doc"], None)
    assert docs[0].modality == "Pdf"
    assert docs[0].source_uri == "memory://guide.pdf"

    hits = rust_db.search("page two figures", 3, {"source": "pdf"})
    assert hits
    assert hits[0].modality == "Pdf"
    assert hits[0].page_number == 2
    assert hits[0].path[0] == "Page 2"
    assert hits[0].region_kind == "paragraph"


def test_image_ingestion_exposes_multimodal_schema(rust_db):
    added = rust_db.add_image(
        "memory://image.png",
        {"asset": "diagram"},
        "image-doc",
        "System diagram with retrieval blocks",
    )

    assert added.document_id == "image-doc"
    assert added.chunk_count == 1

    hits = rust_db.search("diagram retrieval blocks", 1, {"asset": "diagram"})
    assert len(hits) == 1
    assert hits[0].modality == "Image"
    assert "PageImage" in hits[0].embedding_views