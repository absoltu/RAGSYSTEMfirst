from pathlib import Path

from converters.document_converter import (
    DocumentParser
)

from chunkers.chunker import (
    get_chunker
)

from embeddings.embedding_client import (
    EmbeddingClient
)

from vectordb.chroma_client import (
    ChromaVectorDB
)

from utils.markdown_cleaner import (
    MarkdownCleaner
)

from utils.pdf_splitter import (
    split_pdf
)

from utils.chunk_sectioning import (
    ChunkSectionMapper
)

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    EMBEDDING_MODEL,
    TOKENIZER_MODEL,
    PDF_BATCH_PAGES
)

import gc
import os


# =========================================================
# CONFIG
# =========================================================

FILE_PATH = (
    "documents/digital_processing_of_signals.pdf"
)

OUTPUT_FORMAT = "markdown"

CHUNKER_TYPE = "token"

CHUNK_SIZE = DEFAULT_CHUNK_SIZE

OVERLAP = DEFAULT_OVERLAP


filename = Path(FILE_PATH).stem

COLLECTION_NAME = (
    f"{filename}"
    f"_chtype_{CHUNKER_TYPE}"
    f"_chsize_{CHUNK_SIZE}"
    f"_overlap_{OVERLAP}"
    f"_outtype_{OUTPUT_FORMAT}"
)

print(
    f"Collection name: "
    f"{COLLECTION_NAME}"
)


# =========================================================
# INIT
# =========================================================

parser = DocumentParser()

chunker = get_chunker(
    chunker_type=CHUNKER_TYPE,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP,
    model_name=TOKENIZER_MODEL
)

embedding_client = EmbeddingClient()

vectordb = ChromaVectorDB()

section_mapper = ChunkSectionMapper(
    add_section_prefix=True
)


# =========================================================
# CREATE COLLECTION
# =========================================================

collection_metadata = {
    "embedding_model": EMBEDDING_MODEL,
    "chunker": CHUNKER_TYPE,
    "chunk_size": CHUNK_SIZE,
    "overlap": OVERLAP,
    "output_format": OUTPUT_FORMAT
}

collection = vectordb.create_collection(
    collection_name=COLLECTION_NAME,
    metadata=collection_metadata
)


# =========================================================
# SPLIT PDF
# =========================================================

split_files = split_pdf(
    FILE_PATH,
    output_dir="temp_pdf_parts",
    pages_per_chunk=PDF_BATCH_PAGES
)

print(
    f"PDF split into "
    f"{len(split_files)} parts"
)


# =========================================================
# PROCESS PDF PARTS
# =========================================================

all_chunk_count = 0

for pdf_part in split_files:

    print(f"\nProcessing: {pdf_part}")
    chunk_metadata = None  # Initialize for each iteration
    # =====================================
    # DOCUMENT CONVERSION
    # =====================================

    if CHUNKER_TYPE == "hybrid":

        document = parser.parse_document(
            pdf_part
        )

        chunks = chunker.chunk(document)
        chunk_metadata = None

    else:

        converted = parser.convert(
            pdf_part,
            output_format=OUTPUT_FORMAT
        )

        converted = MarkdownCleaner.clean(
            converted
        )

        chunks = chunker.chunk(converted)
        
        # =====================================
        # EXTRACT SECTION INFORMATION
        # =====================================
        
        enhanced_chunks, chunk_metadata = (
            section_mapper.enrich_chunks(
                chunks,
                converted,
                section_info=Path(pdf_part).stem
            )
        )
        
        chunks = enhanced_chunks

    print(
        f"Chunks created: "
        f"{len(chunks)}"
    )

    all_chunk_count += len(chunks)



    for i, chunk in enumerate(chunks[:10]):
        print(f'\nCHUNK {i}')
        print(chunk)
        print("="*80)
    # =====================================
    # EMBEDDINGS
    # =====================================

    embeddings = (
        embedding_client.embed_batch(
            chunks
        )
    )

    print("Embeddings generated")

    # =====================================
    # CHROMA INSERT
    # =====================================

    vectordb.add_documents_batch(
        batch_size=256,
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
        source_file=Path(pdf_part).name,
        chunk_metadata=chunk_metadata
    )

    print("Inserted into Chroma")

    # =====================================
    # MEMORY CLEANUP
    # =====================================

    del chunks
    del embeddings

    gc.collect()


# =========================================================
# DONE
# =========================================================

print(
    f"\nCollection created successfully"
)

print(
    f"Total chunks: {all_chunk_count}"
)

# =========================================================
# CLEANUP TEMP FILES
# =========================================================
print("\nCleaning up temporary PDF parts...")
for f in split_files:
    try:
        if os.path.exists(f):
            os.remove(f)
            print(f"Removed {f}")
    except Exception as e:
        print(f"Warning: could not remove {f}: {e}")

# Try to remove the output directory if it's empty
try:
    output_dir = "temp_pdf_parts"
    if os.path.isdir(output_dir) and not os.listdir(output_dir):
        os.rmdir(output_dir)
        print(f"Removed directory {output_dir}")
except Exception as e:
    print(f"Warning: could not remove directory {output_dir}: {e}")