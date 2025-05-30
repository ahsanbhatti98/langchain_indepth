from langchain_core.prompts import ChatPromptTemplate

# template

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful {domain} expert."),
        (
            "human",
            "Explain in simple terms, what is {topic}",
        ),
    ]
)

prompt = chat_template.invoke(
    {
        "domain": "Cricket",
        "topic": "Duckworth-Lewis-Stern (DLS) method",
    }
)
print(prompt)
