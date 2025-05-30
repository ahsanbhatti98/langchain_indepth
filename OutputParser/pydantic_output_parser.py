from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")

model = ChatAnthropic(model="claude-3-haiku-20240307", api_key=ANTHROPOCENE_API_KEY)


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City where the person lives")


parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person. give only this format no other text  \n{format_instructions}",
    input_variables=["place"],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)

# prompt = template.invoke({"place": "New York"})

# print("prompt->>>", prompt)

chain = template | model | parser

result = chain.invoke({"place": "Pakistan"})
print("result->>>", result)
