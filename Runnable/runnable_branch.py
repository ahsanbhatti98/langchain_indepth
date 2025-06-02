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
    RunnableBranch,
)

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")
model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=ANTHROPOCENE_API_KEY,
)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Write a report on the : {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Write a summary of the report on the following  text : {text}",
    input_variables=["text"],
)


report_gen_chain = RunnableSequence(prompt1, model, parser)

summary_gen_chain = RunnableSequence(prompt2, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, summary_gen_chain), RunnablePassthrough()
)

final_chain = RunnableSequence(
    report_gen_chain,
    branch_chain,
)

result = final_chain.invoke({"topic": "Artificial Intelligence"})
print(result)
