import importlib.util
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_LIB = REPO_ROOT / "target" / "debug" / "libvlite_py.so"


def load_rust_module():
    if not RUST_LIB.exists():
        subprocess.run(["cargo", "build", "-p", "vlite-py"], cwd=REPO_ROOT, check=True)

    spec = importlib.util.spec_from_file_location("vlite_py", RUST_LIB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_documents(count: int):
    return [
        f"# Doc {index}\nRust document {index} discusses retrieval quality and chunk provenance."
        for index in range(count)
    ]


def run_rust_core_benchmark(document_count: int = 1000, query_count: int = 25, top_k: int = 5):
    """
    Run a small local benchmark against the Rust-core VLite binding.

    This benchmark is intentionally simple and fully local. It is meant to provide
    repeatable relative measurements for this repository, not headline marketing claims.
    """
    module = load_rust_module()
    documents = generate_documents(document_count)
    queries = [f"retrieval quality {index % 10}" for index in range(query_count)]

    with tempfile.TemporaryDirectory() as tmpdir:
        db = module.open_local(tmpdir)

        ingest_start = time.perf_counter()
        for index, document in enumerate(documents):
            db.add(document, {"bucket": str(index % 10)}, f"doc-{index}")
        ingest_seconds = time.perf_counter() - ingest_start

        search_times = []
        for query in queries:
            query_start = time.perf_counter()
            hits = db.search(query, top_k, None)
            query_seconds = time.perf_counter() - query_start
            assert hits
            search_times.append(query_seconds)

        p95_index = max(0, int(len(search_times) * 0.95) - 1)
        return {
            "document_count": document_count,
            "query_count": query_count,
            "top_k": top_k,
            "ingest_seconds": ingest_seconds,
            "avg_search_seconds": statistics.mean(search_times),
            "p95_search_seconds": sorted(search_times)[p95_index],
        }


if __name__ == "__main__":
    print(run_rust_core_benchmark())
