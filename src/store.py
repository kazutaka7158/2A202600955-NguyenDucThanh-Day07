from __future__ import annotations

import math
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            # Cách ly hoàn toàn dữ liệu giữa các vòng chạy test case
            client = chromadb.EphemeralClient()
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
                
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Builds a normalized stored record for one document."""
        embedding = self._embedding_fn(doc.content)
        raw_id = getattr(doc, "id", None) or f"chunk_{self._next_index}"
        return {
            # Giải pháp cốt lõi: Kết hợp số index để tạo ID độc nhất cho ChromaDB không nuốt mất chunk
            "id": f"{raw_id}_idx_{self._next_index}",
            "original_id": raw_id, # Giữ lại ID gốc để phục vụ hàm delete_document
            "content": doc.content,
            "embedding": embedding,
            "metadata": getattr(doc, "metadata", {}) or {},
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Runs an in-memory similarity search over a provided list of records using cosine similarity."""
        if not records or top_k <= 0:
            return []

        query_vector = self._embedding_fn(query)
        q_mag = math.sqrt(_dot(query_vector, query_vector))

        scored_records = []
        for record in records:
            vec = record["embedding"]
            dot_prod = _dot(query_vector, vec)
            v_mag = math.sqrt(_dot(vec, vec))

            similarity = dot_prod / (q_mag * v_mag) if (q_mag > 0 and v_mag > 0) else 0.0
            scored_records.append((similarity, record))

        scored_records.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, rec in scored_records[:top_k]:
            results.append({
                "id": rec["original_id"],  # Trả về ID nguyên bản cho bộ test
                "content": rec["content"],
                "metadata": rec["metadata"],
                "score": score
            })
        return results

    def add_documents(self, docs: list[Document]) -> None:
        """Embed each document's content and store it."""
        if not docs:
            return

        if self._use_chroma and self._collection is not None:
            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for doc in docs:
                record = self._make_record(doc)
                self._next_index += 1
                
                ids.append(record["id"])
                documents.append(record["content"])
                embeddings.append(record["embedding"])
                
                meta = record["metadata"]
                if not meta:
                    meta = {"_is_dummy": True}
                metadatas.append(meta)

            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._next_index += 1
                self._store.append(record)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Find the top_k most similar documents to query."""
        if self._use_chroma and self._collection is not None:
            query_embeddings = [self._embedding_fn(query)]
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k
            )
            
            parsed_results = []
            if results and results.get("ids") and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    raw_distance = results["distances"][0][i] if results.get("distances") else 0.0
                    score = 1.0 / (1.0 + raw_distance)

                    # Tách chuỗi để lấy lại ID gốc ban đầu trả về cho test case
                    full_id = results["ids"][0][i]
                    orig_id = full_id.split("_idx_")[0] if "_idx_" in full_id else full_id

                    parsed_results.append({
                        "id": orig_id,
                        "content": results["documents"][0][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "score": score
                    })
                parsed_results.sort(key=lambda x: x["score"], reverse=True)
                return parsed_results
            return []
        else:
            return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """Search with optional metadata pre-filtering."""
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        if self._use_chroma and self._collection is not None:
            query_embeddings = [self._embedding_fn(query)]
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=metadata_filter
            )
            
            parsed_results = []
            if results and results.get("ids") and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    raw_distance = results["distances"][0][i] if results.get("distances") else 0.0
                    score = 1.0 / (1.0 + raw_distance)

                    full_id = results["ids"][0][i]
                    orig_id = full_id.split("_idx_")[0] if "_idx_" in full_id else full_id

                    parsed_results.append({
                        "id": orig_id,
                        "content": results["documents"][0][i] if results.get("documents") else "",
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "score": score
                    })
                parsed_results.sort(key=lambda x: x["score"], reverse=True)
                return parsed_results
            return []
        else:
            filtered_records = []
            for record in self._store:
                match = True
                for key, val in metadata_filter.items():
                    if record["metadata"].get(key) != val:
                        match = False
                        break
                if match:
                    filtered_records.append(record)
                    
            return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """Remove all chunks belonging to a document where doc_id equals target or metadata['doc_id'] == doc_id."""
        if self._use_chroma and self._collection is not None:
            existing = self._collection.get()
            target_ids = []
            if existing and existing.get("ids"):
                for i in range(len(existing["ids"])):
                    full_id = existing["ids"][i]
                    orig_id = full_id.split("_idx_")[0] if "_idx_" in full_id else full_id
                    meta = existing["metadatas"][i] if existing.get("metadatas") else {}
                    
                    if orig_id == doc_id or full_id == doc_id or meta.get("doc_id") == doc_id:
                        target_ids.append(full_id)
            
            if target_ids:
                self._collection.delete(ids=target_ids)
                return True
            return False
        else:
            initial_size = len(self._store)
            self._store = [
                r for r in self._store 
                if r["original_id"] != doc_id and r["metadata"].get("doc_id") != doc_id
            ]
            return len(self._store) < initial_size