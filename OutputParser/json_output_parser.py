from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")

model = ChatAnthropic(model="claude-3-haiku-20240307", api_key=ANTHROPOCENE_API_KEY)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, city and age of a fictional person.give only this format no other text \n {format_instructions}",
    input_variables=[],
    partial_variables={
        "format_instructions": parser.get_format_instructions(),
    },
)

# method 1

prompt = template.format()

print("prompt->>>", prompt)

result = model.invoke(prompt)
print("result->>>", result)
print("result.content->>>", result.content)

final_result = parser.parse(result.content)

print("final_result->>>", final_result["name"])


# method 2
# chain = template | model | parser

# result = chain.invoke({})
# print("result->>>", result)
