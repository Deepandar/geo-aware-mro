import json
import gc
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FastRAGIndexer:
    def __init__(self, persist_dir="data/chroma_index"):
        self.persist_dir = Path(persist_dir)
        self.manifest_path = self.persist_dir / "index_manifest.json"
        self._client = None
        self._ef = None

    def _init_resources(self):
        """Standardizes the custom wrapper for full ChromaDB Query/Index compatibility."""
        if self._ef is None:
            from fastembed import TextEmbedding
            
            class DirectFastEmbedWrapper:
                def __init__(self):
                    # threads=1 is mandatory for stability in MINGW64/constrained environments
                    self.model = TextEmbedding(
                        model_name="BAAI/bge-small-en-v1.5",
                        threads=1
                    )
                
                def __call__(self, input):
                    # Used during upsert/indexing - Parameter must be named 'input'
                    return [list(e) for e in self.model.embed(input)]
                
                def embed_query(self, input):
                    # Required for collection.query() calls
                    result = list(self.model.embed([input] if isinstance(input, str) else input))
                    return [list(e) for e in result]

                def embed_documents(self, input):
                    # Recommended for broader ChromaDB/LangChain compatibility
                    return [list(e) for e in self.model.embed(input)]
                
                def name(self) -> str:
                    # Required by ChromaDB to validate configuration and prevent metadata conflicts
                    return "fastembed-bge-small-en-v1.5"
                @staticmethod
                def is_legacy():
                    return False

                @staticmethod
                def default_space():
                    return "cosine"
                @staticmethod
                def supported_spaces():
                    return ["cosine"]
                @staticmethod
                def get_config():
                    return {

                        "model_name": "BAAI/bge-small-en-v1.5",
                        "default_space": "cosine",
                        "supported_spaces": ["cosine"],
                    }

            self._ef = DirectFastEmbedWrapper()

        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        return self._client.get_or_create_collection("mro_advisory", embedding_function=self._ef)

    def _scan_files(self):
        docs = []
        # Explicitly ignore heavy/recursive folders seen in your project structure
        ignore = {"__pycache__", ".git", ".pytest_cache", "venv_notebooks", "data", "logs", "mlruns"}
        for root, _, files in os.walk("."):
            if any(ig in root for ig in ignore):
                continue
            for f in files:
                fp = Path(root) / f
                # Filter for relevant text files under a safe size limit (100KB)
                if fp.suffix in {'.py', '.md', '.yaml', '.yml'} and fp.stat().st_size < 100000:
                    try:
                        content = fp.read_text(encoding='utf-8', errors='ignore').strip()
                        if content:
                            # Truncate content to keep context window manageable
                            docs.append({"text": content[:1200], "metadata": {"source": str(fp)}})
                    except Exception:
                        continue
        return docs

    def build(self):
        """Builds the vector index in small batches to prevent Force Close crashes."""
        col = self._init_resources()
        docs = self._scan_files()
        if docs:
            # Batch size of 20 keeps RAM usage low during the BGE embedding process
            batch_size = 20
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i+batch_size]
                col.upsert(
                    ids=[f"id_{i+j}" for j in range(len(batch))],
                    documents=[d["text"] for d in batch],
                    metadatas=[d["metadata"] for d in batch]
                )
        
        # Create manifest to skip rebuild on future runs
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, 'w') as f:
            json.dump({"status": "ready", "count": len(docs)}, f)
            
        self.close()
        return len(docs)

    def query(self, text: str):
        """Retrieves grounded context for the advisory engine."""
        col = self._init_resources()
        res = col.query(query_texts=[text], n_results=3)
        self.close()
        return [{"text": t, "metadata": m} for t, m in zip(res["documents"][0], res["metadatas"][0])]

    def close(self):
        """Explicitly clear resources to free up memory for the simulation/RL tasks."""
        self._client = None
        self._ef = None
        gc.collect()

    # ---------------------------------------------------
    # Backward compatibility for older test suite
    # ---------------------------------------------------

    def index_documents(self, documents):
        """
        Minimal compatible document indexing API
        expected by the pytest suite.
        """

        col = self._init_resources()

        ids = [
            f"manual_doc_{i}"
            for i in range(len(documents))
        ]

        metadatas = [
            {"source": "unit_test"}
            for _ in documents
        ]

        col.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return len(documents)
