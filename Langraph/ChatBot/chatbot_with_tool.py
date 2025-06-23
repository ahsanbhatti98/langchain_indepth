from typing import TypedDict, Annotated
from langgraph.graph import add_messages, StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

load_dotenv()

chatbot_node_key = "chatbot"
tool_node_key = "tool_node"


class BasicChatBot(TypedDict):
    messages: Annotated[list, add_messages]


search_tool = TavilySearchResults(max_results=2)
tools = [search_tool]

llm = ChatGroq(model="llama-3.1-8b-instant", verbose=True)

llm_with_tools = llm.bind_tools(tools=tools)


def chatbot(state: BasicChatBot):
    return {
        "messages": [llm_with_tools.invoke(state["messages"])],
    }


def tool_router(state: BasicChatBot):
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END


tool_node = ToolNode(tools=tools)

graph = StateGraph(BasicChatBot)

graph.add_node(chatbot_node_key, chatbot)
graph.add_node(tool_node_key, tool_node)
graph.set_entry_point(chatbot_node_key)


graph.add_conditional_edges(
    chatbot_node_key,
    tool_router,
    {
        tool_node_key: tool_node_key,
        END: END,
    },
)

graph.add_edge(tool_node_key, chatbot_node_key)

app = graph.compile()

while True:
    user_input = input("User: ")
    if user_input in ["exit", "end"]:
        break
    else:
        result = app.invoke({"messages": [HumanMessage(content=user_input)]})

        print(result)
