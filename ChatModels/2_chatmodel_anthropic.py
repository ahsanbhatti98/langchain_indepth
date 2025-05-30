from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")

model = ChatAnthropic(model="claude-3-haiku-20240307", api_key=ANTHROPOCENE_API_KEY)

result = model.invoke("What is the capital of Pakistan?")

print(result.content)
