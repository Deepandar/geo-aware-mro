import pytest
import shutil
import gc
from src.llm.rag_indexer import FastRAGIndexer
from src.llm.llm_advisory import LLMAdvisoryEngine

@pytest.fixture(autouse=True)
def memory_manager():
    """Ensure garbage collection bounds every test."""
    gc.collect()
    yield
    gc.collect()

@pytest.fixture
def isolated_indexer(tmp_path):
    """Provides a completely isolated DB path for testing."""
    test_db_path = tmp_path / "test_chroma"
    idx = FastRAGIndexer(persist_dir=str(test_db_path))
    yield idx
    idx.close()
    if test_db_path.exists():
        shutil.rmtree(test_db_path)

def test_indexer_builds_and_queries(isolated_indexer):
    count = isolated_indexer.build()
    assert count > 0
    res = isolated_indexer.query("What is the pipeline config?")
    assert len(res) > 0

def test_advisory_engine(isolated_indexer):
    isolated_indexer.build()
    engine = LLMAdvisoryEngine(indexer=isolated_indexer)
    response = engine.ask("How does KKT stabilization work?")
    assert "answer" in response
    assert isinstance(response["sources"], list)
