from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader


load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")
model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=ANTHROPOCENE_API_KEY,
)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a summary of the following poem : {poem}",
    input_variables=["poem"],
)

load_doc = TextLoader("cricket.txt", encoding="utf-8")

docs = load_doc.load()

poem = docs[0].page_content

# print(docs)

chain = prompt1 | model | parser

summary = chain.invoke({"poem": poem})
print("Summary of the poem:")
print(summary)
