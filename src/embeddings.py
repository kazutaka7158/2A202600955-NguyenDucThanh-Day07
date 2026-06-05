from __future__ import annotations

import hashlib
import math

LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"


class MockEmbedder:
    """Deterministic embedding backend used by tests and default classroom runs."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._backend_name = "mock embeddings fallback"

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode()).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class LocalEmbedder:
    """Sentence Transformers-backed local embedder."""

    def __init__(self, model_name: str = LOCAL_EMBEDDING_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._backend_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        if hasattr(embedding, "tolist"):
            return embedding.tolist()
        return [float(value) for value in embedding]


# class OpenAIEmbedder:
#     """OpenAI embeddings API-backed embedder."""

#     def __init__(self, model_name: str = OPENAI_EMBEDDING_MODEL) -> None:
#         from openai import OpenAI

#         self.model_name = model_name
#         self._backend_name = model_name
#         self.client = OpenAI()

#     def __call__(self, text: str) -> list[float]:
#         response = self.client.embeddings.create(model=self.model_name, input=text)
#         return [float(value) for value in response.data[0].embedding]


_mock_embed = MockEmbedder()

import os
from typing import Optional


class GeminiEmbedder:
    """Google GenAI SDK-backed Gemini embedding model."""

    def __init__(
        self,
        model_name: str = GEMINI_EMBEDDING_MODEL,
        dimensions: Optional[int] = None,
    ) -> None:
        """Khởi tạo Client kết nối Google AI Studio.

        Args:
            model_name: Tên model embedding (Ví dụ: 'gemini-embedding-001'
              hoặc 'gemini-embedding-2').
            dimensions: Số chiều vector đầu ra mong muốn (Chỉ áp dụng với
              gemini-embedding-001).
                        Hỗ trợ từ 128 đến 3072 (Mặc định nếu không truyền là
                        3072).
        """
        from google import genai

        self.model_name = model_name
        self._backend_name = "gemini"
        self.dimensions = dimensions

        # Tự động nhận biến môi trường GEMINI_API_KEY
        self.client = genai.Client()

    def __call__(self, text: str) -> list[float]:
        from google.genai import types

        # Thiết lập cấu hình bổ sung nếu dùng model v1 và có chỉnh dimension
        config = None
        if self.dimensions and "gemini-embedding-001" in self.model_name:
            config = types.EmbedContentConfig(output_dimensionality=self.dimensions)

        # Gọi API chính thức từ bộ genai SDK mới
        response = self.client.models.embed_content(
            model=self.model_name, contents=text, config=config
        )

        # Trả về danh sách vector float
        return response.embeddings[0].values