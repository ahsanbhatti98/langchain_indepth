from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

load_dotenv()
ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")
model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=ANTHROPOCENE_API_KEY,
)

parser = StrOutputParser()


prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Exaplain the following joke in simple terms: \n{joke}",
    input_variables=["joke"],
)

chain = RunnableSequence(prompt, model, parser, prompt2, model, parser)


print(chain.invoke({"topic": "cats"}))
