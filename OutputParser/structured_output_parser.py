from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")

model = ChatAnthropic(model="claude-3-haiku-20240307", api_key=ANTHROPOCENE_API_KEY)

schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)


template = PromptTemplate(
    template="Give 3 facts about the {topic} give only this format no other text \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)

chain = template | model | parser

result = chain.invoke({"topic": "Black Holes"})
print("result->>>", result)
