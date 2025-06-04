from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

# documents = loader.load()

# for doc in documents:
#     print(doc.metadata)
#     print(doc.page_content[:100])  # Print the first 100 characters of the content
#     print("-" * 40)  # Separator for readability

# print(documents[0].metadata)
# print(documents[0].page_content)


documents = loader.lazy_load()

for doc in documents:
    print(doc.metadata)
