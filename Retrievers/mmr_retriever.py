from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


documents = [
    Document(
        page_content="Langchain is a framework for building applications with LLMs."
    ),
    Document(page_content="Langchain is used to build llms based applications."),
    Document(page_content="Chroma is a vector database for LLMs."),
    Document(page_content="Embeddings are a way to represent text as vectors."),
    Document(
        page_content="Hugging Face provides a wide range of models for NLP tasks."
    ),
    Document(page_content="MMR helps select the most relevant documents."),
    Document(page_content="LangChain support FAISS, Chroma, and other vector stores."),
]

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding,
)

retriever = vector_store.as_retriever(
    search_type="mmr",  # Use MMR for diverse retrieval
    search_kwargs={"k": 3, "lambda_mult": 1},  # Number of documents to retrieve
)
query = "What is Langchain used for?"
result = retriever.invoke(query)
print("Retrieved Documents:")
for i, doc in enumerate(result):
    print(f"\n-- Result {i+1} --")
    print(f"Title: {doc.page_content}...")
