import chromadb

if __name__ == '__main__':
    client = chromadb.HttpClient(host='localhost', port=8000)
    client.heartbeat()

    # collection = client.get_collection('all-my-documents')
    collection = client.create_collection("all-my-documents")

    collection.add(
        documents=["This is document1", "This is document2"],
        metadatas=[{"source": "notion"}, {"source": "google-docs"}],
        ids=["doc13", "doc26"],  # unique for each doc
    )

    results = collection.query(
        query_texts=["This is a query document"],
        n_results=2,
    )

    print(results)
