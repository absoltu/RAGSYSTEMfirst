from typing import List, Dict, Any


class ContextGenerator:
    """
    Генератор контекста для RAG системы.
    Формирует контекст из результатов поиска для передачи в модель.
    """

    @staticmethod
    def generate_context(search_results: List[Dict[str, Any]], max_chunks: int = None) -> str:
        """
        Генерирует контекст из результатов поиска.

        :param search_results: Список результатов поиска
        :param max_chunks: Максимальное количество чанков для включения (если None, использует все)
        :return: Сформированный контекст как строка
        """
        if not search_results:
            return "No relevant information found."

        # Ограничиваем количество чанков если указано
        chunks_to_use = search_results[:max_chunks] if max_chunks else search_results

        context_parts = []

        for i, result in enumerate(chunks_to_use, 1):
            # Извлекаем текст и метаданные
            text = result.get("text", "")
            metadata = result.get("metadata", {})

            # Формируем заголовок для чанка
            chunk_header = f"Chunk {i}:"
            if "section" in metadata:
                chunk_header += f" {metadata['section']}"
            if "source" in metadata:
                chunk_header += f" (Source: {metadata['source']})"

            # Добавляем релевантность если есть
            if "relevance" in result:
                chunk_header += f" [Relevance: {result['relevance']:.3f}]"
            elif "cosine_similarity" in result:
                chunk_header += f" [Similarity: {result['cosine_similarity']:.3f}]"

            # Формируем часть контекста
            context_part = f"{chunk_header}\n\n{text}\n\n{'='*50}\n\n"
            context_parts.append(context_part)

        # Объединяем все части
        full_context = "".join(context_parts)

        return full_context.strip()

    @staticmethod
    def generate_context_with_query(query: str, search_results: List[Dict[str, Any]], max_chunks: int = None) -> str:
        """
        Генерирует контекст с включением исходного запроса.

        :param query: Исходный запрос пользователя
        :param search_results: Список результатов поиска
        :param max_chunks: Максимальное количество чанков
        :return: Контекст с запросом
        """
        context = ContextGenerator.generate_context(search_results, max_chunks)

        full_prompt = f"Query: {query}\n\nRelevant Context:\n\n{context}"

        return full_prompt