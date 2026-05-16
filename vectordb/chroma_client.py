import chromadb
from chromadb.config import Settings

from config import CHROMA_DB_PATH


class ChromaVectorDB:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

    def create_collection(
        self,
        collection_name: str,
        metadata: dict
    ):

        return self.client.get_or_create_collection(
            name=collection_name,
            metadata=metadata
        )

    def add_documents(
        self,
        collection,
        chunks,
        embeddings,
        source_file: str
    ):

        ids = [
            f"{source_file}_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": source_file,
                "chunk_index": i
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def add_documents_batch(
            self,
            collection,
            chunks,
            embeddings,
            source_file: str,
            batch_size: int = 128,
            chunk_metadata: list = None
    ):
        """
        Add documents in batches with optional chunk-level metadata.
        
        Args:
            collection: Chroma collection
            chunks: List of text chunks
            embeddings: List of embeddings
            source_file: Source file name
            batch_size: Batch size for insertion
            chunk_metadata: Optional list of dicts with metadata for each chunk
        """
        total = len(chunks)

        for i in range(0, total, batch_size):
            batch_chunks = chunks[i:i + batch_size]

            batch_embeddings = embeddings[i:i + batch_size]

            ids = [
                f"{source_file}_{i + j}"
                for j in range(len(batch_chunks))
            ]

            metadatas = []
            for j in range(len(batch_chunks)):
                metadata = {
                    "source": source_file,
                    "chunk_index": i + j
                }
                
                # Add chunk-specific metadata if provided
                if chunk_metadata and (i + j) < len(chunk_metadata):
                    chunk_meta = chunk_metadata[i + j]
                    # Chroma only supports string/int/float/bool metadata, not lists
                    # So we skip 'headers' list and keep only 'section' string
                    for key, value in chunk_meta.items():
                        # Skip list types (not supported by Chroma)
                        if isinstance(value, list):
                            continue
                        metadata[key] = value
                
                metadatas.append(metadata)

            collection.add(
                ids=ids,
                documents=batch_chunks,
                embeddings=batch_embeddings,
                metadatas=metadatas
            )

            print(
                f"Inserted batch "
                f"{i // batch_size + 1}"
            )