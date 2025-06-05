from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


documents = [
    Document(
        page_content="Langchain is a framework for building applications with LLMs."
    ),
    Document(page_content="Chroma is a vector database for LLMs."),
    Document(page_content="Embeddings are a way to represent text as vectors."),
    Document(
        page_content="Hugging Face provides a wide range of models for NLP tasks."
    ),
]


vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
    collection_name="example_collection",
    persist_directory="chroma_db",  # Directory to persist the vector store
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 2},  # Number of documents to retrieve
)

query = "What is Chroma use for?"

result = retriever.invoke(query)
print("Retrieved Documents:")

for i, doc in enumerate(result):
    print(f"\n-- Result {i+1} --")
    print(f"Title: {doc.page_content}...")
