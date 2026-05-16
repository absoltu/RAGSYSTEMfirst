from openai import OpenAI

from config import (
    LM_STUDIO_BASE_URL,
    EMBEDDING_MODEL,
    TOKENIZER_MODEL
)
from utils.token_batcher import TokenBatcher

class EmbeddingClient:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL
    ):

        self.model_name = model_name

        self.client = OpenAI(
            base_url=LM_STUDIO_BASE_URL,
            api_key="lm-studio"
        )
        self.token_batcher = TokenBatcher(tokenizer_name=TOKENIZER_MODEL,
                                        max_tokens=4096)
    def embed(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        return [
            item.embedding
            for item in response.data
        ]

    def embed_batch(self, texts):

        batches = self.token_batcher.create_batches(
            texts
        )

        all_embeddings = []

        for i, batch in enumerate(batches):
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )

            batch_embeddings = [
                item.embedding
                for item in response.data
            ]

            all_embeddings.extend(batch_embeddings)

            print(
                f"Batch {i + 1}/{len(batches)} complete"
            )

        return all_embeddings
