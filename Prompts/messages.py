from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(
    model="claude-3-haiku-20240307", api_key=os.getenv("ANTHROPOCENE_API_KEY")
)


chat_history = [
    SystemMessage(
        content="You are a helpful assistant that summarizes research papers."
    ),
]

while True:
    input_text = input("You: ")
    if input_text.lower() in ["exit", "quit"]:
        break
    chat_history.append(HumanMessage(content=input_text))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")

print("Chat history:")
