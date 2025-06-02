from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
    RunnableLambda,
)


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

joke_gen_chain = RunnableSequence(prompt, model, parser)


def word_count(joke: str) -> int:
    return len(joke.split())


parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "word_count": RunnableLambda(word_count),
    },
)

final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain,
)

result = final_chain.invoke({"topic": "cars"})

final_result = """{} \n word count: {}""".format(result["joke"], result["word_count"])
print(final_result)
