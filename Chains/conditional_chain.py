from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give the sentiment of the feedback."
    )


load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")


model = ChatAnthropic(
    model="claude-3-haiku-20240307", api_key=ANTHROPOCENE_API_KEY, temperature=0
)

parser = StrOutputParser()

parserPydantic = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following text in to postive or negative keywords only not any other text: \n{feedback} \n {format_instructions}",
    input_variables=["feedback"],
    partial_variables={"format_instructions": parserPydantic.get_format_instructions()},
)


classifier_chain = prompt1 | model | parserPydantic


prompt_positive = PromptTemplate(
    template="Write an appropriate response to this positive feedback and not any other text: \n{feedback}",
    input_variables=["feedback"],
)

prompt_negative = PromptTemplate(
    template="Write an appropriate response to this negative feedback and not any other text: \n{feedback}",
    input_variables=["feedback"],
)


branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_positive | model | parser),
    (lambda x: x.sentiment == "negative", prompt_negative | model | parser),
    RunnableLambda(lambda x: "could not classify sentiment"),
)


chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "This is a great movie"})
print(result)

chain.get_graph().print_ascii()
