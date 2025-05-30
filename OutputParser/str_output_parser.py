from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")

model = ChatAnthropic(model="claude-3-haiku-20240307", api_key=ANTHROPOCENE_API_KEY)


# 1st Prompt Template --> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

# 2nd Prompt Template --> summary
template2 = PromptTemplate(
    template="Write a summary 2 lines summary on the following text as a bullet point. \n {text}",
    input_variables=["text"],
)


parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "Black Holes"})

print(result)
