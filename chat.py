from rag import RAGSystem
from config import (
    RAG_ENABLED,
    RAG_COLLECTION_NAME,
    CHAT_TEMPERATURE,
    CHAT_MAX_TOKENS,
    SEARCH_TYPE,
    MMR_LAMBDA,
    CONTEXT_TOP_N
)


def main():
    print("Simple terminal chat")
    print("Type 'exit' or 'quit' to stop")
    print("RAG enabled:", RAG_ENABLED)
    if RAG_ENABLED:
        print("Search type:", SEARCH_TYPE)
        if SEARCH_TYPE == "mmr":
            print("MMR lambda:", MMR_LAMBDA)
        print("Context chunks:", CONTEXT_TOP_N)
        print("Collection:", RAG_COLLECTION_NAME)
    print("Max tokens:", CHAT_MAX_TOKENS)
    print("Temperature:", CHAT_TEMPERATURE)
    print("=" * 50)

    rag_system = RAGSystem(collection_name=RAG_COLLECTION_NAME)

    while True:
        user_query = input("You: ").strip()
        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit"}:
            print("Exiting chat...")
            break

        if RAG_ENABLED:
            result = rag_system.query(
                user_query=user_query,
                search_type=SEARCH_TYPE,
                mmr_lambda=MMR_LAMBDA,
                context_top_n=CONTEXT_TOP_N,
                max_tokens=CHAT_MAX_TOKENS,
                temperature=CHAT_TEMPERATURE
            )
        else:
            result = {
                "response": rag_system.generate_response(
                    context="",
                    query=user_query,
                    max_tokens=CHAT_MAX_TOKENS,
                    temperature=CHAT_TEMPERATURE
                )
            }

        print("Model:")
        print(result["response"])
        print("=" * 50)


if __name__ == "__main__":
    main()