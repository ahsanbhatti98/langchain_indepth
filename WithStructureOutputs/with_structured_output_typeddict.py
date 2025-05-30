from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from typing import TypedDict, Optional, Literal, Annotated

load_dotenv()

ANTHROPOCENE_API_KEY = os.getenv("ANTHROPOCENE_API_KEY")

model = ChatAnthropic(model="claude-3-haiku-20240307", api_key=ANTHROPOCENE_API_KEY)


# Example of using TypedDict with LangChain Anthropic for structured output
# And it only provide the structured output not will validate it.

# Simple TypedDict for structured output


# class Review(TypedDict):
#     summary: str
#     sentiment: str


# structured_output = model.with_structured_output(Review)


# result = structured_output.invoke(
#     """ The hardware is great, but the software is buggy and crashes often. Overall, I would not recommend this product.
# """
# )

# print(result)
# print(result["summary"])
# print(result["sentiment"])


# Annotated TypedDict for structured output


class Review(TypedDict):

    key_theme: Annotated[
        str, "Write down all the key themes in the review as a list of strings."
    ]
    summary: Annotated[str, "Write a brief summary of the review."]
    sentiment: Annotated[
        Literal["pos", "neg"], "Write down the sentiment of the review."
    ]
    pros: Annotated[
        Optional[dict[str, str]],
        "Write down the pros of the product as a dictionary with keys as strings and values as strings.",
    ]
    cons: Annotated[
        Optional[dict[str, str]],
        "Write down the cons of the product as a dictionary with keys as strings and values as strings and if it is speratley metioned in the review as in title cons.",
    ]
    name: Annotated[
        Optional[str],
        "Write down the name of the reviewer if it is available otherwise don't add it in the review.",
    ]


structured_output = model.with_structured_output(Review)

result = structured_output.invoke(
    """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 

"""
)

print(result)
