from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")
model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=ANTHROPOCENE_API_KEY,
)

parser = StrOutputParser()

loader = PyPDFLoader("dl-curriculum.pdf")

docs = loader.load()

print(f"Number of pages in the PDF: {len(docs)}")
print(f"First page content: {docs[0].page_content[500]}...")
