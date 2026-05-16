from rag.context_generator import ContextGenerator
from rag.model_interface import ModelInterface
from config import CONTEXT_TOP_N

# Тест генератора контекста
def test_context_generator():
    # Пример результатов поиска
    sample_results = [
        {
            "text": "Пропускная способность - это максимальная скорость передачи информации.",
            "metadata": {"section": "Глава 1", "source": "book.pdf"},
            "relevance": 0.85
        },
        {
            "text": "Достоверность измеряется вероятностью ошибки.",
            "metadata": {"section": "Глава 2", "source": "book.pdf"},
            "relevance": 0.72
        }
    ]

    generator = ContextGenerator()

    # Тест генерации контекста
    context = generator.generate_context(sample_results, max_chunks=CONTEXT_TOP_N)
    print("Generated Context:")
    print(context)
    print("\n" + "="*50 + "\n")

    # Тест с запросом
    query = "Что такое пропускная способность?"
    full_context = generator.generate_context_with_query(query, sample_results)
    print("Context with Query:")
    print(full_context)

# Тест интерфейса модели
def test_model_interface():
    model = ModelInterface()

    # Проверяем соединение
    connected = model.check_connection()
    print(f"Model connection: {'OK' if connected else 'FAILED'}")

    if connected:
        # Тест генерации ответа
        context = "Пропускная способность - это максимальная скорость передачи информации при заданной достоверности."
        query = "Что такое пропускная способность?"

        response = model.generate_response(context, query, max_tokens=100)
        print("Model Response:")
        print(response)
    else:
        print("Skipping response generation - model not connected")

if __name__ == "__main__":
    print("Testing Context Generator...")
    test_context_generator()
    print("\nTesting Model Interface...")
    test_model_interface()