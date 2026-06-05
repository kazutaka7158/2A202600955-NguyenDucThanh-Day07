from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        # Match boundaries: '. ', '! ', '? ', or '.\n' using a positive lookbehind
        # This keeps the punctuation attached to the preceding sentence
        sentence_end = re.compile(r'(?<=\. |! |\? |.\n)')
        
        # Split and clean up empty tokens caused by consecutive spaces/boundaries
        raw_sentences = sentence_end.split(text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        
        chunks: list[str] = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunk_slice = sentences[i : i + self.max_sentences_per_chunk]
            # Join with a standard space and strip trailing/leading whitespace
            chunks.append(" ".join(chunk_slice).strip())
            
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]
        
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text.strip()] if current_text.strip() else []

        if not remaining_separators:
            return [
                current_text[i : i + self.chunk_size].strip()
                for i in range(0, len(current_text), self.chunk_size)
            ]

        separator = remaining_separators[0]
        next_separators = remaining_separators[1:]

        if separator == "":
            splits = list(current_text)
        else:
            splits = current_text.split(separator)

        final_chunks = []
        good_buffer = ""

        for split in splits:
            potential_chunk = (good_buffer + separator + split) if good_buffer else split

            if len(potential_chunk) <= self.chunk_size:
                good_buffer = potential_chunk
            else:
                if good_buffer:
                    final_chunks.append(good_buffer.strip())
                    good_buffer = ""

                if len(split) > self.chunk_size:
                    deeper_chunks = self._split(split, next_separators)
                    final_chunks.extend(deeper_chunks)
                else:
                    good_buffer = split

        if good_buffer:
            final_chunks.append(good_buffer.strip())

        return [c for c in final_chunks if c]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    dot_product = _dot(vec_a, vec_b)
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
        
    return dot_product / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        """
        Executes and gathers metrics on all three chunking mechanisms.
        """
        # Khởi tạo chunkers với chunk_size cấu hình tự động theo yêu cầu của test suite
        fixed_chunker = FixedSizeChunker(chunk_size=chunk_size, overlap=min(50, chunk_size // 4))
        sentence_chunker = SentenceChunker(max_sentences_per_chunk=3)
        recursive_chunker = RecursiveChunker(chunk_size=chunk_size)

        # Tiến hành bóc tách chuỗi
        fixed_chunks = fixed_chunker.chunk(text)
        sentence_chunks = sentence_chunker.chunk(text)
        recursive_chunks = recursive_chunker.chunk(text)

        # Hàm bổ trợ tính toán thông số metric trực tiếp
        def get_metrics(chunks: list[str]) -> tuple[int, float]:
            if not chunks:
                return 0, 0.0
            lengths = [len(c) for c in chunks]
            return len(chunks), sum(lengths) / len(chunks)

        fixed_count, fixed_avg = get_metrics(fixed_chunks)
        sentence_count, sentence_avg = get_metrics(sentence_chunks)
        recursive_count, recursive_avg = get_metrics(recursive_chunks)

        # SỬA LỖI ĐỊNG DẠNG: Đưa 'count' và 'avg_length' lên lớp phẳng của dict
        # Đổi key chiến lược câu từ 'sentence' thành 'by_sentences' theo test suite dòng 209
        return {
            "fixed_size": {
                "chunks": fixed_chunks,
                "count": fixed_count,
                "avg_length": fixed_avg
            },
            "by_sentences": {
                "chunks": sentence_chunks,
                "count": sentence_count,
                "avg_length": sentence_avg
            },
            "recursive": {
                "chunks": recursive_chunks,
                "count": recursive_count,
                "avg_length": recursive_avg
            }
        }