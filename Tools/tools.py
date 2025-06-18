from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")
model = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=ANTHROPOCENE_API_KEY,
)

result_pkr = model.invoke("how much is 1 usd in pkr?")

# print("pkr=>>>>>>>", result_pkr.content)

# print("#" * 20)


@tool
def multiply(x: int, y: int) -> int:
    """Multiplies two numbers."""
    return x * y


# print(multiply.invoke({"x": 2, "y": 3}))  # Example usage of the tool

# print(multiply.name, multiply.description, multiply.args)


result = model.invoke("hi")

llm_with_tool = model.bind_tools([multiply])

query = HumanMessage(
    content="can you mutiply 2 with 3000",
)

messages = [query]

result_with_tool = llm_with_tool.invoke(messages)

messages.append(result_with_tool)

tool_result = multiply.invoke(result_with_tool.tool_calls[0])

messages.append(tool_result)


result_with_tool = llm_with_tool.invoke(messages)
messages.append(result_with_tool)
# print(result_with_tool)
# print("" * 20)
# print(messages)
