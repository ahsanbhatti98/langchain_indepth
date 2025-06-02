from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import (
    RunnableParallel,
    RunnableSequence,
    RunnablePassthrough,
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

prompt2 = PromptTemplate(
    template="Explain the following joke in simple terms: \n{joke}",
    input_variables=["joke"],
)


joke_gen_chain = RunnableSequence(prompt, model, parser)


parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explanation": RunnableSequence(
            prompt2,
            model,
            parser,
        ),
    }
)


final_chain = RunnableSequence(
    joke_gen_chain,
    parallel_chain,
)

print(final_chain.invoke({"topic": "UFC"}))
