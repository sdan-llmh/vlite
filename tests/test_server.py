from pathlib import Path

from fastapi.testclient import TestClient

import vlite.server as server_module


def test_rust_api_add_search_info(rust_module, tmp_path):
    lib_path = Path(server_module.__file__).resolve().parents[1] / "target" / "debug" / "libvlite_py.so"
    app = server_module.create_rust_app(
        collection_path=str(tmp_path / "api-db"),
        search_paths=[str(lib_path)],
    )
    client = TestClient(app)

    add_response = client.post(
        "/add",
        json={"text": "# Intro\nSimple text.\n\n# Search\nRust search is local.", "metadata": {"tenant": "demo"}},
    )
    assert add_response.status_code == 200
    assert add_response.json()["chunk_count"] == 2

    search_response = client.post(
        "/search",
        json={"query": "rust search", "top_k": 2, "where": {"tenant": "demo"}},
    )
    assert search_response.status_code == 200
    payload = search_response.json()
    assert len(payload) == 2
    assert payload[0]["path"] == ["Search"]
    assert payload[0]["modality"] == "Text"

    info_response = client.get("/info")
    assert info_response.status_code == 200
    assert info_response.json()["document_count"] == 1


def test_rust_api_pdf_route_uses_page_aware_ingestion(monkeypatch, rust_module, tmp_path):
    lib_path = Path(server_module.__file__).resolve().parents[1] / "target" / "debug" / "libvlite_py.so"

    def fake_add_pdf_to_rust_vlite(db, file_path, metadata=None, document_id=None, use_ocr=False, langs=None):
        return db.add_pdf_pages(
            ["# Page One\nA first page.", "# Page Two\nA second page about retrieval."],
            metadata or {},
            document_id,
            file_path,
        )

    monkeypatch.setattr(server_module, "add_pdf_to_rust_vlite", fake_add_pdf_to_rust_vlite)
    app = server_module.create_rust_app(
        collection_path=str(tmp_path / "pdf-api-db"),
        search_paths=[str(lib_path)],
    )
    client = TestClient(app)

    response = client.post(
        "/add_pdf",
        files={"file": ("example.pdf", b"%PDF-1.4 fake pdf bytes", "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["chunk_count"] >= 2

    search_response = client.post("/search", json={"query": "second page retrieval", "top_k": 2})
    assert search_response.status_code == 200
    results = search_response.json()
    assert any(result["page_number"] == 2 for result in results)
    assert any(result["path"][0] == "Page 2" for result in results)