from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableSequence

load_dotenv()
ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")
model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=ANTHROPOCENE_API_KEY,
)

prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate a LinkedIn post about {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "linkedin_post": RunnableSequence(prompt2, model, parser),
    }
)

result = chain.invoke({"topic": "AI in healthcare"})

print(result)
