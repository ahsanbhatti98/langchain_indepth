from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


document = [
    "The capital of Pakistan is Islamabad.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo.",
    "The capital of India is New Delhi.",
    "The capital of China is Beijing.",
]

query = "Karachi is the city of Pakistan or india?"

document_embeddings = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)


scores = cosine_similarity([query_embedding], document_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print(f"Query: {query}")
print(f"Most similar document: {document[index]}\nScore: {score}")
