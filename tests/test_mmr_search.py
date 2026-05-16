from retrieval.search import ChromaSearcher


COLLECTION_NAME = "TeoryElConection_chtype_token_chsize_500_overlap_50_outtype_markdown"

searcher = ChromaSearcher(
    collection_name=COLLECTION_NAME
)

query = "Пропускная способность"

# Поиск с MMR ранжированием
results = searcher.search_with_mmr(
    query=query,
    top_k=5,
    lambda_param=0.9  # Баланс между релевантностью и разнообразием
)

print("\nMMR SEARCH RESULTS:\n")

for i, result in enumerate(results, 1):
    print("=" * 80)
    print(f"RESULT #{i}")
    print(f"Relevance: {result['relevance']:.4f}")
    print(f"Distance: {result['distance']:.4f}")
    print(f"Metadata: {result['metadata']}")
    print("\nTEXT:\n")
    print(result["text"][:1000])

# Сравнение с обычным поиском
print("\n" + "="*80)
print("COMPARISON WITH REGULAR SEARCH:")
print("="*80)

regular_results = searcher.search(query=query, top_k=5)

for i, result in enumerate(regular_results, 1):
    print(f"Regular #{i} - Relevance: {result['cosine_similarity']:.4f}")