from src.llm.rag_indexer import FastRAGIndexer


class LLMAdvisoryEngine:
    def __init__(self, indexer=None):
        self.indexer = indexer or FastRAGIndexer()

    def ask(self, query: str):
        context = self.indexer.query(query)
        sources = (
            list(set([c["metadata"]["source"] for c in context])) if context else []
        )
        return {
            "answer": f"Strategy grounded in {len(sources)} files from the Geo-Aware codebase.",
            "sources": sources,
            "confidence": "High" if len(sources) >= 2 else "Low",
        }
