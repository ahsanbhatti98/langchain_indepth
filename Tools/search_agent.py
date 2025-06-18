from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

from langchain_community.tools import DuckDuckGoSearchRun

from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")


search_tool = DuckDuckGoSearchRun()


# result = search_tool.invoke("top news in Pakistan today?")

# print(result)

llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    api_key=ANTHROPOCENE_API_KEY,
)


prompt = hub.pull("hwchase17/react")

# print("prompt ->>>>>>>>", prompt)

agent = create_react_agent(llm=llm, tools=[search_tool], prompt=prompt)

agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

response = agent_executor.invoke(
    {"input": "what is the weather conditon in karachi today ?"}
    # {"input": "What is the capital of france and what its population?"}
)

print(response)
