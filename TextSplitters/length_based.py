from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl-curriculum.pdf")

docs = loader.load()

text_splitter = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=5,
)


result = text_splitter.split_documents(docs)

print(result[0].page_content)
print("-" * 40)
print(result[1].page_content)
