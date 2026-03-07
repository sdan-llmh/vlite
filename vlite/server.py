import os
from importlib.metadata import PackageNotFoundError, version
from typing import List, Optional, Union

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from vlite.utils import (
    add_pdf_to_rust_vlite,
    create_rust_vlite,
    process_file,
    process_pdf,
    process_webpage,
)


_legacy_vlite = None

try:
    APP_VERSION = version("vlite")
except PackageNotFoundError:  # pragma: no cover - local editable source tree
    APP_VERSION = "0.0.0-dev"


class TextData(BaseModel):
    text: str
    metadata: Optional[dict] = None


class RetrieveRequest(BaseModel):
    text: Optional[str] = None
    top_k: int = 5
    metadata: Optional[dict] = None


class UpdateRequest(BaseModel):
    text: Optional[str] = None
    metadata: Optional[dict] = None
    vector: Optional[List[float]] = None


class RustSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    where: Optional[dict] = None


def get_legacy_vlite():
    global _legacy_vlite
    if _legacy_vlite is None:
        from vlite.main import VLite

        _legacy_vlite = VLite()
    return _legacy_vlite


def create_legacy_app() -> FastAPI:
    app = FastAPI(
        title="VLite API",
        description="Legacy VLite API for text embedding and retrieval.",
        version=APP_VERSION,
    )

    @app.post("/add", response_model=List[tuple], summary="Add text to the legacy collection")
    async def add_text(data: Union[TextData, List[TextData]]):
        vlite = get_legacy_vlite()
        if isinstance(data, TextData):
            data = [data]
        texts = [item.text for item in data]
        metadatas = [item.metadata for item in data]
        return vlite.add(texts, metadata=metadatas)

    @app.post("/add_file", response_model=List[tuple], summary="Add text from a file to the legacy collection")
    async def add_file(file: UploadFile = File(...)):
        vlite = get_legacy_vlite()
        file_path = await save_upload_file(file)
        chunks = process_file(file_path)
        return vlite.add(chunks)

    @app.post("/add_pdf", response_model=List[tuple], summary="Add text from a PDF file to the legacy collection")
    async def add_pdf(file: UploadFile = File(...), use_ocr: bool = False):
        vlite = get_legacy_vlite()
        file_path = await save_upload_file(file)
        chunks = process_pdf(file_path, use_ocr=use_ocr)
        return vlite.add(chunks)

    @app.post("/add_webpage", response_model=List[tuple], summary="Add text from a webpage to the legacy collection")
    async def add_webpage(url: str):
        vlite = get_legacy_vlite()
        chunks = process_webpage(url)
        return vlite.add(chunks)

    @app.post("/retrieve", response_model=List[tuple], summary="Retrieve similar texts from the legacy collection")
    async def retrieve_text(request: RetrieveRequest):
        vlite = get_legacy_vlite()
        if request.text is None and request.metadata is None:
            raise HTTPException(status_code=400, detail="Either 'text' or 'metadata' must be provided")
        return vlite.retrieve(text=request.text, top_k=request.top_k, metadata=request.metadata)

    @app.delete("/delete", response_model=int, summary="Delete items from the legacy collection")
    async def delete_texts(ids: Union[str, List[str]]):
        vlite = get_legacy_vlite()
        return vlite.delete(ids)

    @app.put("/update/{item_id}", response_model=bool, summary="Update an item in the legacy collection")
    async def update_text(item_id: str, request: UpdateRequest):
        vlite = get_legacy_vlite()
        return vlite.update(item_id, text=request.text, metadata=request.metadata, vector=request.vector)

    @app.get("/get", response_model=List[tuple], summary="Get items from the legacy collection")
    async def get_texts(ids: Optional[List[str]] = None, where: Optional[dict] = None):
        vlite = get_legacy_vlite()
        return vlite.get(ids=ids, where=where)

    @app.get("/count", response_model=int, summary="Get the count of items in the legacy collection")
    async def count_items():
        vlite = get_legacy_vlite()
        return vlite.count()

    @app.post("/save", response_model=None, summary="Save the legacy collection")
    async def save_collection():
        get_legacy_vlite().save()

    @app.post("/clear", response_model=None, summary="Clear the legacy collection")
    async def clear_collection():
        get_legacy_vlite().clear()

    @app.get("/info", response_model=dict, summary="Get legacy collection information")
    async def get_info():
        vlite = get_legacy_vlite()
        return {
            "count": vlite.count(),
            "collection": vlite.collection,
            "model": str(vlite.model),
        }

    @app.get("/dump", response_model=dict, summary="Dump the legacy collection data")
    async def dump_data():
        return get_legacy_vlite().dump()

    return app


def create_rust_app(collection_path: str = "contexts/rust-api", search_paths: Optional[List[str]] = None) -> FastAPI:
    db = create_rust_vlite(collection_path, search_paths=search_paths)
    app = FastAPI(
        title="VLite Rust API",
        description="Rust-core VLite API for structured local retrieval.",
        version=APP_VERSION,
    )

    @app.post("/add")
    async def add_text(data: TextData):
        result = db.add(data.text, data.metadata or {}, None)
        return {
            "document_id": result.document_id,
            "segment_ids": result.segment_ids,
            "chunk_count": result.chunk_count,
        }

    @app.post("/add_pdf")
    async def add_pdf(file: UploadFile = File(...), use_ocr: bool = False):
        file_path = await save_upload_file(file)
        result = add_pdf_to_rust_vlite(db, file_path, metadata={"filename": file.filename}, use_ocr=use_ocr)
        return {
            "document_id": result.document_id,
            "segment_ids": result.segment_ids,
            "chunk_count": result.chunk_count,
        }

    @app.post("/search")
    async def search(request: RustSearchRequest):
        hits = db.search(request.query, request.top_k, request.where or None)
        return [
            {
                "document_id": hit.document_id,
                "segment_id": hit.segment_id,
                "segment_kind": hit.segment_kind,
                "path": hit.path,
                "text": hit.text,
                "metadata": hit.metadata,
                "modality": hit.modality,
                "embedding_views": hit.embedding_views,
                "page_number": hit.page_number,
                "region_kind": hit.region_kind,
                "score": hit.score,
                "parent_text": hit.parent_text,
            }
            for hit in hits
        ]

    @app.get("/get")
    async def get(ids: Optional[List[str]] = None):
        docs = db.get(ids, None)
        return [
            {
                "id": doc.id,
                "title": doc.title,
                "raw_text": doc.raw_text,
                "metadata": doc.metadata,
                "modality": doc.modality,
                "source_uri": doc.source_uri,
            }
            for doc in docs
        ]

    @app.delete("/delete")
    async def delete(ids: List[str]):
        return {"deleted": db.delete(ids)}

    @app.get("/info")
    async def info():
        value = db.info()
        return {
            "path": value.path,
            "document_count": value.document_count,
            "segment_count": value.segment_count,
            "chunking_strategy": value.chunking_strategy,
            "retrieval_strategy": value.retrieval_strategy,
        }

    @app.post("/compact")
    async def compact():
        db.compact()
        return {"status": "ok"}

    return app


app = create_legacy_app()


async def save_upload_file(upload_file: UploadFile) -> str:
    """
    Save the uploaded file to disk and return its file path.
    """
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{upload_file.filename}"
    with open(file_path, "wb") as file:
        file.write(await upload_file.read())
    return file_path