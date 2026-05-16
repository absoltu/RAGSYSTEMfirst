from vectordb.chroma_client import ChromaVectorDB
vectordb = ChromaVectorDB()
collections_list = vectordb.client.list_collections()
names=[collections_list[i].name for i in range(len(collections_list))]
# print(vectordb.client.list_collections())
print(names)