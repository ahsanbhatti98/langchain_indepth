from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# template

template = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "{query}",
        ),
    ]
)


# append chat history to the template

chat_history = []

# load Chat history

with open(
    "Prompts/chat_history.txt",
) as f:
    chat_history.extend(f.readlines())


prompt = template.invoke(
    {
        "query": "What is the status of my order?",
        "chat_history": chat_history,
    }
)

print(prompt)
