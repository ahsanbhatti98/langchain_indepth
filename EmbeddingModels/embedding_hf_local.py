from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

docuemnt = [
    "The capital of Pakistan is Islamabad.",
    "The capital of France is Paris.",
    "The capital of Japan is Tokyo.",
]


result = embedding.embed_documents(docuemnt)

print(str(result))
