from retrieval.search import ChromaSearcher
from rag.context_generator import ContextGenerator
from rag.model_interface import ModelInterface
from config import (
    SEARCH_TYPE,
    MMR_LAMBDA,
    CONTEXT_TOP_N
)
from typing import Dict, Any, List, Optional


class RAGSystem:
    """
    Полная RAG система, объединяющая поиск, генерацию контекста и общение с моделью.
    """

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.searcher = None  # Инициализируем позже, чтобы избежать проблем с импортом
        self.context_generator = ContextGenerator()
        self.model_interface = ModelInterface()

    def _init_searcher(self):
        """Инициализирует поисковик при первом использовании."""
        if self.searcher is None:
            self.searcher = ChromaSearcher(collection_name=self.collection_name)

    def search_documents(
        self,
        user_query: str,
        search_type: str = None,
        mmr_lambda: float = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Выполняет только поиск документов без генерации ответа.

        :param user_query: Запрос пользователя
        :param search_type: Тип поиска ("def" или "mmr")
        :param mmr_lambda: Параметр MMR
        :param top_k: Количество результатов
        :return: Список результатов поиска
        """
        self._init_searcher()

        search_type = search_type or SEARCH_TYPE
        mmr_lambda = mmr_lambda or MMR_LAMBDA
        top_k = top_k or CONTEXT_TOP_N

        if search_type == "mmr":
            return self.searcher.search_with_mmr(
                query=user_query,
                top_k=top_k,
                lambda_param=mmr_lambda
            )
        else:
            return self.searcher.search(
                query=user_query,
                top_k=top_k
            )

    def generate_context(
        self,
        search_results: List[Dict[str, Any]],
        max_chunks: int = None
    ) -> str:
        """
        Генерирует контекст из результатов поиска.

        :param search_results: Результаты поиска
        :param max_chunks: Максимум чанков
        :return: Сформированный контекст
        """
        max_chunks = max_chunks or CONTEXT_TOP_N
        return self.context_generator.generate_context(search_results, max_chunks)

    def generate_response(
        self,
        context: str,
        query: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Генерирует ответ на основе контекста.

        :param context: Контекст
        :param query: Запрос
        :param max_tokens: Максимум токенов
        :param temperature: Температура
        :return: Ответ модели
        """
        return self.model_interface.generate_response(
            context=context,
            query=query,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def query(
        self,
        user_query: str,
        search_type: str = None,
        mmr_lambda: float = None,
        context_top_n: int = None,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Выполняет полный RAG запрос: поиск -> контекст -> генерация ответа.

        :param user_query: Запрос пользователя
        :param search_type: Тип поиска ("def" или "mmr"), если None использует из config
        :param mmr_lambda: Параметр MMR, если None использует из config
        :param context_top_n: Количество чанков в контексте, если None использует из config
        :param max_tokens: Максимум токенов в ответе
        :param temperature: Температура генерации
        :return: Словарь с результатами
        """
        # Используем параметры из config если не указаны
        search_type = search_type or SEARCH_TYPE
        mmr_lambda = mmr_lambda or MMR_LAMBDA
        context_top_n = context_top_n or CONTEXT_TOP_N

        # Выполняем поиск
        search_results = self.search_documents(
            user_query=user_query,
            search_type=search_type,
            mmr_lambda=mmr_lambda,
            top_k=context_top_n
        )

        # Генерируем контекст
        context = self.generate_context(
            search_results,
            max_chunks=context_top_n
        )

        # Генерируем ответ
        response = self.generate_response(
            context=context,
            query=user_query,
            max_tokens=max_tokens,
            temperature=temperature
        )

        return {
            "query": user_query,
            "search_type": search_type,
            "search_results": search_results,
            "context": context,
            "response": response,
            "metadata": {
                "mmr_lambda": mmr_lambda if search_type == "mmr" else None,
                "context_chunks": len(search_results)
            }
        }


# # Пример использования
# if __name__ == "__main__":
#     # Инициализируем систему
#     COLLECTION_NAME = "TeoryElConection_chtype_token_chsize_500_overlap_50_outtype_markdown"
#     rag_system = RAGSystem(collection_name=COLLECTION_NAME)

#     # Проверяем соединение с моделью
#     if not rag_system.model_interface.check_connection():
#         print("Warning: Cannot connect to the model. Make sure LM Studio is running.")
#         print("You can still test search and context generation.")
#     else:
#         print("Model connection OK.")

#     # Пример запроса
#     query = "Что такое пропускная способность?"

#     print(f"Query: {query}")
#     print("=" * 50)

#     # Сначала тестируем только поиск
#     print("Testing search...")
#     search_results = rag_system.search_documents(query, search_type="def")
#     print(f"Found {len(search_results)} results")
#     print()

#     # Тестируем генерацию контекста
#     print("Testing context generation...")
#     context = rag_system.generate_context(search_results[:3])  # Используем первые 3 результата
#     print("Generated context (first 200 chars):")
#     print(context[:200] + "..." if len(context) > 200 else context)
#     print()

#     # Если модель подключена, тестируем полный запрос
#     if rag_system.model_interface.check_connection():
#         print("Testing full RAG query with default search...")
#         result_def = rag_system.query(query, search_type="def")
#         print("Response:")
#         print(result_def["response"])
#         print()

#         print("Testing full RAG query with MMR search...")
#         result_mmr = rag_system.query(query, search_type="mmr", mmr_lambda=0.7)
#         print("Response:")
#         print(result_mmr["response"])
#     else:
#         print("Model not connected - skipping full query test")