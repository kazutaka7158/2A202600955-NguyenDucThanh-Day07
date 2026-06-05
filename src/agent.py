from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Retrieves relevant context from the vector store, compiles a grounded 
        prompt, and gets a response from the LLM.
        """
        if not question or not question.strip():
            return "Please provide a valid question."

        # 1. Retrieve top-k relevant chunks from the store
        matched_records = self.store.search(query=question, top_k=top_k)
        
        # 2. SỬA TẠI ĐÂY: Trích xuất text từ key "content" thay vì "text" để đồng bộ với bộ Test
        context_blocks = []
        for record in matched_records:
            if "content" in record:
                context_blocks.append(record["content"])
            elif "text" in record:  # Cơ chế fallback an toàn phòng trường hợp store dùng cấu trúc cũ
                context_blocks.append(record["text"])
        
        if not context_blocks:
            # Fallback if no relevant documents are found in the store
            context_str = "No relevant context found in the knowledge base."
        else:
            # Join chunks with clear visual separation
            context_str = "\n---\n".join(context_blocks)

        # 3. Build a structured prompt grounding the LLM in the retrieved context
        prompt = (
            "You are a helpful assistant answering questions using the provided knowledge base.\n"
            "Answer the question using only the context provided below. If you do not know the answer "
            "or if it isn't in the context, state that you don't know.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # 4. Call the LLM to generate the final answer
        try:
            return self.llm_fn(prompt)
        except Exception as e:
            return f"An error occurred while generating the answer: {str(e)}"