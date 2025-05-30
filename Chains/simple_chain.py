from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")


prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}", input_variables=["topic"]
)

model = ChatAnthropic(model="claude-3-haiku-20240307", api_key=ANTHROPOCENE_API_KEY)


parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "BJJ"})

print(result)

chain.get_graph().print_ascii()
