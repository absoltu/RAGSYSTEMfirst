from transformers import AutoTokenizer
from config import TOKENIZER_MODEL

class TokenBatcher:

    def __init__(
        self,
        tokenizer_name: str,
        max_tokens: int = 8192
    ):

        self.max_tokens = max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name
        )

    def count_tokens(self, text: str):

        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

    def create_batches(self, texts):

        batches = []

        current_batch = []

        current_tokens = 0

        for text in texts:

            text_tokens = self.count_tokens(text)

            # skip giant chunks
            if text_tokens > self.max_tokens:
                continue

            if (
                current_tokens + text_tokens
                > self.max_tokens
            ):

                batches.append(current_batch)

                current_batch = [text]

                current_tokens = text_tokens

            else:

                current_batch.append(text)

                current_tokens += text_tokens

        if current_batch:
            batches.append(current_batch)

        return batches
