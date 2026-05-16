import chromadb
import shutil
from config import CHROMA_DB_PATH


def list_collections():

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collections = client.list_collections()

    for collection in collections:

        print(collection.name)


def delete_collection(name: str):

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    client.delete_collection(name)


def delete_all_collections():

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH
    )

    collections = client.list_collections()

    for collection in collections:

        client.delete_collection(
            collection.name
        )
    shutil.rmtree(
        CHROMA_DB_PATH,
        ignore_errors=True
    )
    print("ALL Collections was deleted\nFolder deleted")