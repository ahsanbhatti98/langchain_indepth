from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")

# Recreate the document objects from the previous data
docs = [
    Document(
        page_content=(
            """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
        ),
        metadata={"source": "Doc1"},
    ),
    Document(
        page_content=(
            """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
        ),
        metadata={"source": "Doc2"},
    ),
    Document(
        page_content=(
            """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
        ),
        metadata={"source": "Doc3"},
    ),
    Document(
        page_content=(
            """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
        ),
        metadata={"source": "Doc4"},
    ),
]


embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
)

base_retriever = vector_store.as_retriever(
    search_type="similarity",  # Use similarity search for initial retrieval
    search_kwargs={"k": 5},  # Retrieve top 5 similar documents
)

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    anthropic_api_key=ANTHROPOCENE_API_KEY,
)

compressor = LLMChainExtractor.from_llm(
    llm=llm,
)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)


query = "What is photosynthesis?"

result = retriever.invoke(query)

print("Retrieved Documents:")
for i, doc in enumerate(result):
    print(f"\n-- Result {i+1} --")
    print(f"Title: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content: {doc.page_content}...")  # Print first 100 characters
