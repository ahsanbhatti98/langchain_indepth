from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain.retrievers import MultiQueryRetriever

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

all_docs = [
    Document(
        page_content="Regular walking boosts heart health and can reduce symptoms of depression.",
        metadata={"source": "H1"},
    ),
    Document(
        page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.",
        metadata={"source": "H2"},
    ),
    Document(
        page_content="Deep sleep is crucial for cellular repair and emotional regulation.",
        metadata={"source": "H3"},
    ),
    Document(
        page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.",
        metadata={"source": "H4"},
    ),
    Document(
        page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.",
        metadata={"source": "H5"},
    ),
    Document(
        page_content="The solar energy system in modern homes helps balance electricityl demand.",
        metadata={"source": "I1"},
    ),
    Document(
        page_content="Python balances readability with power, making it a popular system design language.",
        metadata={"source": "I2"},
    ),
    Document(
        page_content="Photosynthesis enables plants to produce energy by converting sunlight.",
        metadata={"source": "I3"},
    ),
    Document(
        page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.",
        metadata={"source": "I4"},
    ),
    Document(
        page_content="Black holes bend spacetime and store immense gravitational energy.",
        metadata={"source": "I5"},
    ),
]


vector_store = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding,
)

similarity_search_retriever = vector_store.as_retriever(
    search_type="similarity",  # Use similarity search for standard retrieval
    search_kwargs={"k": 3},  # Number of documents to retrieve
)

model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=ANTHROPOCENE_API_KEY,
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    llm=model,
    retriever=vector_store.as_retriever(
        search_kwargs={"k": 5},  # Number of documents to retrieve
    ),
)

query = "How to improve energy level and maintain balance ?"

# similarity_results = similarity_search_retriever.invoke(query)
multi_query_results = multi_query_retriever.invoke(query)

# print("Similarity Search Results:")
# for i, doc in enumerate(similarity_results):
#     print(f"\n-- Result {i+1} --")
#     print(f"Title: {doc.page_content}...")  # Print first 50 characters

print("\nMulti-Query Retriever Results:")
for i, doc in enumerate(multi_query_results):
    print(f"\n-- Result {i+1} --")
    print(f"Title: {doc.page_content}...")  # Print first 50 characters
