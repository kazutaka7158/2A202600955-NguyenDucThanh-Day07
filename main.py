from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    GEMINI_EMBEDDING_MODEL,
    LocalEmbedder,
    GeminiEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# Import chính xác từ file chứa code chunking của bạn
from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    ChunkingStrategyComparator,
    compute_similarity,
)

SAMPLE_FILES = [
    "data/caremark-oct2013.txt",
    "data/FDAMDD_v3b_1216_15Feb2008_nostructures.txt",
    "data/healthalliance-2013.txt",
    "data/humana_2014_wi.txt",
    "data/uhc-cns-drugs-pdf-list.txt"
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue
        
        try:    
            content = path.read_text(encoding="utf-8")
        except Exception:
            try:
                content = path.read_bytes().decode("cp1252")
            except Exception:
                print(f"Failed to read file: {path}")
                continue

        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:200].replace("\n", " ")
    return f"[DEMO LLM] Generated answer based on context preview: {preview}..."


def apply_chunking_strategy(strategy_key: str, docs: list[Document]) -> list[Document]:
    """
    Áp dụng các class chunker dựa trên các key định danh chuẩn của bộ Test.
    """
    chunked_documents: list[Document] = []
    
    # Đồng bộ hóa key theo đúng tiêu chuẩn test suite
    if strategy_key == "fixed_size":
        chunker = FixedSizeChunker(chunk_size=600, overlap=100)
    elif strategy_key == "by_sentences":
        chunker = SentenceChunker(max_sentences_per_chunk=3)
    elif strategy_key == "recursive":
        chunker = RecursiveChunker(chunk_size=600)
    else:
        return docs

    for doc in docs:
        text_chunks = chunker.chunk(doc.content)
        
        for i, text_chunk in enumerate(text_chunks):
            chunked_documents.append(
                Document(
                    id=f"{doc.id}_{strategy_key}_{i}",
                    content=text_chunk,
                    metadata={
                        **doc.metadata, 
                        "chunk_index": i, 
                        "strategy": strategy_key
                    }
                )
            )
            
    return chunked_documents


def get_smart_query(loaded_docs: list[Document]) -> str:
    """Tự động sinh query phù hợp dựa trên danh sách file được nạp thực tế."""
    doc_ids = {doc.id.lower() for doc in loaded_docs}
    
    if len(doc_ids) == 1:
        doc_id = list(doc_ids)[0]
        if "caremark" in doc_id:
            return "What is the effective date of the Caremark document and what are the main terms?"
        if "fda" in doc_id:
            return "What are the FDA drug safety guidelines or regulations mentioned?"
        if "healthalliance" in doc_id:
            return "Summarize the medical policy or coverage limits for HealthAlliance."
        if "humana" in doc_id:
            return "What information is provided regarding Humana health insurance plans or guidelines?"
        if "uhc" in doc_id:
            return "List the CNS drugs included in the UnitedHealthcare (UHC) list."

    return "What are the common medical coverage, drug policies, or effective guidelines across these insurance documents?"


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES

    print("=== Manual File Test ===")
    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        return 1

    print(f"\nLoaded {len(docs)} documents.")
    query = question or get_smart_query(docs)

    # Cấu hình Embedding Backend
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    
    if provider == "gemini":
        try:
            embedder = GeminiEmbedder(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"))
        except Exception as e:
            print(f"Failed to load Gemini Embedder ({e}). Falling back to Mock.")
            embedder = _mock_embed
    elif provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"Embedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")
    print(f"Query hiện tại: '{query}'\n")

    # ĐỔI THÀNH KEY CHUẨN: Đồng bộ 100% dữ liệu đầu vào với Test Suite
    strategies = ["fixed_size", "by_sentences", "recursive"]
    
    print("=" * 80)
    print(" SO SÁNH HIỆU NĂNG VÀ CHỈ SỐ RETRIEVAL SCORE CỦA CÁC CHIẾN LƯỢC CHUNKING")
    print("=" * 80)

    summary_report = {}

    for idx, strategy in enumerate(strategies, start=1):
        print(f"\n⚡ [{idx}/{len(strategies)}] Chiến lược: {strategy}")
        
        # 1. Đo lường thời gian Chunking và Thêm tài liệu
        start_time = time.time()
        chunked_docs = apply_chunking_strategy(strategy, docs)
        
        store = EmbeddingStore(collection_name=f"store_comparison_{idx}", embedding_fn=embedder)
        store.add_documents(chunked_docs)
        indexing_time = time.time() - start_time
        
        print(f"   -> Đã xử lý {len(chunked_docs)} chunks trong {indexing_time:.3f} giây.")
        
        # 2. Thực hiện Tìm kiếm (Retrieval)
        search_results = store.search(query, top_k=2)
        
        print(f"   -> Phân tích kết quả Retrieval:")
        if not search_results:
            print("      (Không tìm thấy dữ liệu tương đồng phù hợp)")
            continue
            
        query_vector = embedder(query)
        
        calculated_scores = []
        for s_idx, result in enumerate(search_results, start=1):
            raw_score = result.get('score')
            raw_score_str = f"{raw_score:.4f}" if isinstance(raw_score, (int, float)) else "N/A"
            
            # ĐỔI TỪ result['text'] THÀNH result['content'] để khớp với bản vá EmbeddingStore
            chunk_content = result['content']
            chunk_vector = embedder(chunk_content)
            cosine_sim = compute_similarity(query_vector, chunk_vector)
            calculated_scores.append(cosine_sim)
            
            content_preview = chunk_content[:120].replace('\n', ' ').strip()
            print(f"      [{s_idx}] Store Score/Distance: {raw_score_str} | Cosine Similarity Tính Toán: {cosine_sim:.4f}")
            print(f"          Nguồn file: {result['metadata'].get('source')}")
            print(f"          Nội dung: \"{content_preview}...\"")
            
        # 3. Phân tích chỉ số chất lượng Retrieval (Retrieval Quality Metrics)
        if len(calculated_scores) >= 2:
            score_gap = calculated_scores[0] - calculated_scores[1]
            top1_score = calculated_scores[0]
        else:
            score_gap = 0.0
            top1_score = calculated_scores[0] if calculated_scores else 0.0
            
        print(f"   📊 Chỉ số Chất lượng:")
        print(f"      - Độ tương đồng Top 1 (Càng gần 1 càng tốt): {top1_score:.4f}")
        print(f"      - Khoảng cách an toàn (Top 1 vs Top 2 Gap): {score_gap:.4f}")

        summary_report[strategy] = {
            "total_chunks": len(chunked_docs),
            "top1_cosine": top1_score,
            "score_gap": score_gap,
            "time": indexing_time
        }
            
        # 4. Kiểm tra phản hồi từ Agent
        agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
        print(f"   -> Phản hồi từ Agent:")
        print(f"      {agent.answer(query, top_k=2)}")
        print("-" * 75)

    # --- BẢNG ĐÁNH GIÁ TỔNG HỢP CUỐI CÙNG ---
    print("\n" + "=" * 80)
    print(" BẢNG TỔNG HỢP VÀ ĐÁNH GIÁ CHỈ SỐ RETRIEVAL GIỮA CÁC CHIẾN LƯỢC")
    print("=" * 80)
    print(f"{'Chiến lược Chunking':<22} | {'Tổng số Chunk':<13} | {'Top 1 Cosine':<12} | {'Top1-Top2 Gap':<13} | {'Thời gian (s)':<12}")
    print("-" * 80)
    for strat, metrics in summary_report.items():
        print(f"{strat:<22} | {metrics['total_chunks']:<13} | {metrics['top1_cosine']:<12.4f} | {metrics['score_gap']:<13.4f} | {metrics['time']:<12.3f}")
    print("=" * 80)

    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    main()