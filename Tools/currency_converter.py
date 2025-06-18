from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
import json
from langchain_core.tools import InjectedToolArg
from typing import Annotated
from langchain.agents import initialize_agent, AgentType

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")


# Create a simple currency converter tool.


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """This function fetches the currency conversion factor between a given base currency and a target currency."""

    url = f"https://v6.exchangerate-api.com/v6/0abd51613b313008210dd7a2/pair/{base_currency}/{target_currency}"

    response = requests.get(url=url)
    return response.json()


# print(
#     "->>>>>>>>>>",
#     get_conversion_factor.invoke({"base_currency": "USD", "target_currency": "PKR"}),
# )


@tool
def convert(
    base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """

    return base_currency_value * conversion_rate


result_conversion = convert.invoke(
    {"base_currency_value": 10, "conversion_rate": 281.9443}
)

# print("dasd00000", result_conversion)

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=ANTHROPOCENE_API_KEY,
)

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [
    HumanMessage(
        content=(
            "Step 1: Get the conversion factor from PKR to USD.\n"
            "Step 2: Using that conversion factor, convert 10 PKR to USD.\n"
            "Please perform both steps."
        )
    )
]


ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    # execute the 1st tool and get the value of conversion rate
    if tool_call["name"] == "get_conversion_factor":
        tool_message1 = get_conversion_factor.invoke(tool_call)
        # fetch this conversion rate
        conversion_rate = json.loads(tool_message1.content)["conversion_rate"]
        # append this tool message to messages list
        messages.append(tool_message1)
    # execute the 2nd tool using the conversion rate from tool 1
    if tool_call["name"] == "convert":
        # fetch the current arg
        tool_call["args"]["conversion_rate"] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)


result = llm_with_tools.invoke(messages)


agent_executor = initialize_agent(
    tools=[get_conversion_factor, convert],
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # using ReAct pattern
    verbose=True,  # shows internal thinking
)

response = agent_executor.invoke({"input": messages})

print("dadada", response)
