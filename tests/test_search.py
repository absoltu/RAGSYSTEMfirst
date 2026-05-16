from retrieval.search import ChromaSearcher


COLLECTION_NAME = "TeoryElConection_chtype_token_chsize_500_overlap_50_outtype_markdown"

searcher = ChromaSearcher(
    collection_name=COLLECTION_NAME
)

query = "Пропускная способность"

results = searcher.search(
    query=query,
    top_k=5,
)

print("\nTOP RESULTS:\n")

for i, result in enumerate(results, 1):

    print("=" * 80)

    print(f"RESULT #{i}")

    print(f"Cosine similarity: {result['cosine_similarity']:.4f}")

    print(f"Distance: {result['distance']:.4f}")

    print(f"Metadata: {result['metadata']}")

    print("\nTEXT:\n")

    print(result["text"][:1000])