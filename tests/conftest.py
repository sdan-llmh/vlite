import importlib.util
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUST_LIB = REPO_ROOT / "target" / "debug" / "libvlite_py.so"


def load_rust_module():
    subprocess.run(
        ["cargo", "build", "-p", "vlite-py"],
        cwd=REPO_ROOT,
        check=True,
    )

    spec = importlib.util.spec_from_file_location("vlite_py", RUST_LIB)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def rust_module():
    return load_rust_module()


@pytest.fixture()
def rust_db(rust_module, tmp_path):
    return rust_module.open_local(str(tmp_path / "collection"))
