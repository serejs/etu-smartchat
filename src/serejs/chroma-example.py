import chromadb

client = chromadb.HttpClient(host='localhost', port=8000)
client.heartbeat()

collection = client.create_collection("all-my-documents")

collection.add(
    documents=["This is document1", "This is document2"],
    metadatas=[{"source": "notion"}, {"source": "google-docs"}],
    ids=["doc1", "doc2"], # unique for each doc
)

results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
)