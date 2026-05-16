import chromadb  # type: ignore[import]
import math
from typing import List, Dict, Any

from config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL
)

from embeddings.embedding_client import EmbeddingClient


class ChromaSearcher:

    def __init__(
        self,
        collection_name: str,
        embedding_model: str = EMBEDDING_MODEL
    ):

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH)
        )

        self.collection = self.client.get_collection(
            name=collection_name
        )

        self.embedding_client = EmbeddingClient(
            model_name=embedding_model
        )

    @staticmethod
    def cosine_similarity(vec1, vec2):

        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))

        if norm_a == 0 or norm_b == 0:
            return 0

        return dot_product / (norm_a * norm_b)

    def search(
        self,
        query: str,
        top_k: int = 5,
        section: str = None,
        source: str = None,
        metadata_filter: dict = None
    ):

        # =========================
        # Query embedding
        # =========================

        query_embedding = self.embedding_client.embed(query)[0]

        # =========================
        # Optional metadata filtering
        # =========================

        filters = {}
        if metadata_filter:
            filters.update(metadata_filter)
        if section:
            filters["section"] = section
        if source:
            filters["source"] = source

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": [
                "documents",
                "embeddings",
                "metadatas",
                "distances"
            ]
        }

        if filters:
            query_kwargs["where"] = filters

        results = self.collection.query(**query_kwargs)

        documents = results.get("documents", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        final_results = []

        for doc, emb, meta, dist in zip(
            documents,
            embeddings,
            metadatas,
            distances
        ):

            similarity = self.cosine_similarity(
                query_embedding,
                emb
            )

            final_results.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "cosine_similarity": similarity,
                "section": meta.get("section"),
                "source": meta.get("source")
            })

        final_results.sort(
            key=lambda x: x["cosine_similarity"],
            reverse=True
        )

        return final_results

    def search_with_mmr(
        self,
        query: str,
        top_k: int = 5,
        lambda_param: float = 0.5,
        section: str = None,
        source: str = None,
        metadata_filter: dict = None
    ) -> List[Dict[str, Any]]:
        """
        Поиск с ранжированием MMR (Maximal Marginal Relevance).

        :param query: Запрос для поиска
        :param top_k: Количество результатов
        :param lambda_param: Параметр баланса (0-1): 1 - только релевантность, 0 - только разнообразие
        :param section: Фильтр по секции
        :param source: Фильтр по источнику
        :param metadata_filter: Дополнительные фильтры метаданных
        :return: Список результатов с MMR-ранжированием
        """

        # Получаем embedding запроса
        query_embedding = self.embedding_client.embed(query)[0]

        # Фильтры
        filters = {}
        if metadata_filter:
            filters.update(metadata_filter)
        if section:
            filters["section"] = section
        if source:
            filters["source"] = source

        # Получаем больше кандидатов для MMR (обычно 2-3 раза больше top_k)
        candidate_multiplier = 3
        n_candidates = min(top_k * candidate_multiplier, 50)  # Ограничение для производительности

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_candidates,
            "include": ["documents", "embeddings", "metadatas", "distances"]
        }

        if filters:
            query_kwargs["where"] = filters

        results = self.collection.query(**query_kwargs)

        documents = results.get("documents", [[]])[0]
        embeddings = results.get("embeddings", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Подготавливаем кандидатов
        candidates = []
        for doc, emb, meta, dist in zip(documents, embeddings, metadatas, distances):
            relevance = self.cosine_similarity(query_embedding, emb)
            candidates.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "relevance": relevance,
                "embedding": emb,
                "section": meta.get("section"),
                "source": meta.get("source")
            })

        # Если кандидатов меньше чем top_k, возвращаем всех отсортированных по релевантности
        if len(candidates) <= top_k:
            candidates.sort(key=lambda x: x["relevance"], reverse=True)
            return candidates[:top_k]

        # MMR ранжирование
        selected = []
        remaining = candidates.copy()

        for _ in range(min(top_k, len(candidates))):
            if not remaining:
                break

            best_score = -float('inf')
            best_candidate = None
            best_idx = -1

            for idx, candidate in enumerate(remaining):
                # Релевантность к запросу
                relevance_score = candidate["relevance"]

                # Максимальная схожесть к уже выбранным
                max_similarity = 0.0
                if selected:
                    similarities = [
                        self.cosine_similarity(candidate["embedding"], sel["embedding"])
                        for sel in selected
                    ]
                    max_similarity = max(similarities)

                # MMR score
                mmr_score = lambda_param * relevance_score - (1 - lambda_param) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
                    best_idx = idx

            if best_candidate:
                selected.append(best_candidate)
                remaining.pop(best_idx)

        return selected