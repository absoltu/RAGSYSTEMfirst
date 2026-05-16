from typing import List
from config import TOKENIZER_MODEL
from transformers import AutoTokenizer
import re


# =========================================================
# HYBRID CHUNKER IMPORT
# =========================================================

try:

    from docling_core.transforms.chunker.hybrid_chunker import (
        HybridChunker
    )

    HYBRID_AVAILABLE = True

except:

    HYBRID_AVAILABLE = False


# =========================================================
# CHARACTER CHUNKER
# =========================================================

class CharacterChunker:

    def __init__(
        self,
        chunk_size: int = 1000,
        overlap: int = 100
    ):

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(
        self,
        text: str
    ) -> List[str]:

        chunks = []

        start = 0

        while start < len(text):

            end = start + self.chunk_size

            chunk = text[start:end]

            chunks.append(chunk)

            start += (
                self.chunk_size
                - self.overlap
            )

        return chunks


# =========================================================
# TOKEN CHUNKER
# =========================================================

class TokenChunker:

    def __init__(
        self,
        model_name: str,
        chunk_size: int = 256,
        overlap: int = 32
    ):

        self.chunk_size = chunk_size

        self.overlap = overlap

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

    def token_count(
        self,
        text: str
    ) -> int:

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

    def split_sentences(
        self,
        text: str
    ):

        sentences = re.split(
            r'(?<=[.!?])\s+',
            text
        )

        return [
            s.strip()
            for s in sentences
            if s.strip()
        ]

    def split_large_paragraph(
        self,
        paragraph: str
    ):
        """Split large paragraph with token-based overlap."""
        sentences = self.split_sentences(paragraph)
        chunks = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.token_count(sentence)

            # Handle giant sentences
            if sentence_tokens > self.chunk_size:
                # Save current chunk if exists
                if current:
                    chunks.append(" ".join(current))
                    current = []
                    current_tokens = 0

                # Split giant sentence by words with overlap
                words = sentence.split()
                temp = []
                word_groups = []

                for word in words:
                    temp.append(word)
                    temp_text = " ".join(temp)

                    if self.token_count(temp_text) >= self.chunk_size:
                        word_groups.append(temp[:-1])  # Remove last word
                        temp = [temp[-1]]

                if temp:
                    word_groups.append(temp)

                # Create chunks with overlap
                for i, group in enumerate(word_groups):
                    if i > 0:
                        # Add overlap from previous group
                        overlap_text = " ".join(word_groups[i - 1][-self.overlap:])
                        group_text = " ".join(group)
                        chunks.append(f"{overlap_text} {group_text}" if overlap_text else group_text)
                    else:
                        chunks.append(" ".join(group))

                continue

            # Add sentence to current chunk
            if current_tokens + sentence_tokens > self.chunk_size:
                if current:
                    chunks.append(" ".join(current))
                current = [sentence]
                current_tokens = sentence_tokens
            else:
                current.append(sentence)
                current_tokens += sentence_tokens

        if current:
            chunks.append(" ".join(current))

        return chunks

    def split_blocks(
        self,
        text: str
    ):

        blocks = re.split(
            r"\n\s*\n",
            text
        )

        return [
            b.strip()
            for b in blocks
            if b.strip()
        ]

    def chunk(
        self,
        text: str
    ) -> List[str]:
        """Chunk text with token-based overlap."""
        blocks = self.split_blocks(text)
        chunks = []
        current_block_group = []  # Group of blocks for current chunk
        current_tokens = 0
        last_chunk_text = None  # Store last chunk for overlap

        for block in blocks:
            block_tokens = self.token_count(block)

            # Handle giant blocks that exceed chunk_size
            if block_tokens > self.chunk_size:
                # Flush current group
                if current_block_group:
                    chunk_text = "\n\n".join(current_block_group)
                    chunks.append(chunk_text)
                    last_chunk_text = chunk_text
                    current_block_group = []
                    current_tokens = 0

                # Split giant block
                large_chunks = self.split_large_paragraph(block)
                chunks.extend(large_chunks)
                last_chunk_text = large_chunks[-1] if large_chunks else None
                continue

            # Check if adding this block would exceed chunk_size
            if current_tokens + block_tokens > self.chunk_size:
                # Save current group as a chunk
                if current_block_group:
                    chunk_text = "\n\n".join(current_block_group)
                    chunks.append(chunk_text)
                    last_chunk_text = chunk_text

                # Start new chunk with overlap
                current_block_group = [block]
                current_tokens = block_tokens
            else:
                # Add block to current group
                current_block_group.append(block)
                current_tokens += block_tokens

        # Flush remaining blocks
        if current_block_group:
            chunk_text = "\n\n".join(current_block_group)
            chunks.append(chunk_text)

        return chunks

# =========================================================
# DOCLING HYBRID CHUNKER
# =========================================================

class DoclingHybridChunker:

    def __init__(self):

        if not HYBRID_AVAILABLE:

            raise ImportError(
                "HybridChunker not available"
            )

        self.chunker = HybridChunker()

    def chunk(self, document):

        chunks = []

        for chunk in self.chunker.chunk(document):

            chunks.append(chunk.text)

        return chunks


# =========================================================
# CHUNKER FACTORY
# =========================================================

def get_chunker(
    chunker_type: str,
    model_name: str = None,
    chunk_size: int = 512,
    overlap: int = 50
):

    # =====================================
    # CHARACTER CHUNKER
    # =====================================

    if chunker_type == "char":

        return CharacterChunker(
            chunk_size=chunk_size,
            overlap=overlap
        )

    # =====================================
    # TOKEN CHUNKER
    # =====================================

    elif chunker_type == "token":

        if model_name is None:

            raise ValueError(
                "model_name required "
                "for token chunker"
            )

        return TokenChunker(
            model_name=model_name,
            chunk_size=chunk_size,
            overlap=overlap
        )

    # =====================================
    # HYBRID CHUNKER
    # =====================================

    elif chunker_type == "hybrid":

        return DoclingHybridChunker()

    # =====================================
    # UNKNOWN
    # =====================================

    else:

        raise ValueError(
            f"Unknown chunker: {chunker_type}"
        )