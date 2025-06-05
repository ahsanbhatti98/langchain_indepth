from langchain_community.retrievers import WikipediaRetriever


retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en",
)

query = "The geopolitical situation in the Middle East"


docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"\n-- Result {i+1} --")
    print(f"Title: {doc.page_content}...")
print(f"Total documents retrieved: {len(docs)}\n")
print("Retrieval complete.")
