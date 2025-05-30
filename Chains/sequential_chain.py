from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")

prompt1 = PromptTemplate(
    template="Generate a detailed report on the topic: {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Summarize the report in a 5 pointer list in a concise manner: {report}",
    input_variables=["report"],
)

model = ChatAnthropic(model="claude-3-haiku-20240307", api_key=ANTHROPOCENE_API_KEY)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke(
    {"topic": "Judo techniques and their applications in self-defense"}
)
# print(chain.get_graph().print_ascii())
print(result)
