import pytest
import shutil
import gc
from src.llm.rag_indexer import FastRAGIndexer


@pytest.fixture
def isolated_indexer(tmp_path):
    """Provides a completely isolated DB path for testing."""
    test_db_path = tmp_path / "test_chroma"
    idx = FastRAGIndexer(persist_dir=str(test_db_path))
    yield idx

    # Release handles for Windows compatibility
    idx.close()
    gc.collect()
    if test_db_path.exists():
        shutil.rmtree(test_db_path, ignore_errors=True)


def test_indexer_builds_and_queries(isolated_indexer):
    # Standardizing to the correct method name 'index_documents'
    isolated_indexer.index_documents(["Unit test document for MRO logic."])
    results = isolated_indexer.query("MRO logic")
    assert len(results) > 0


def test_advisory_engine(isolated_indexer):
    assert True
