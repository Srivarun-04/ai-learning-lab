import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from langchain_core.tools import tool

from langchain_core.messages import ToolMessage

load_dotenv()

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


@tool
def get_word_length(word: str) -> str:
    """get word length"""
    return len(word)

tools = [get_word_length]
llm_with_tools = llm.bind_tools(tools)

msg = [
    (   "human",
        "How many Characters are  in HelloWorld"
    )
]
ai_msg = llm_with_tools.invoke(msg)
print("AI Response:", ai_msg)
msg.append(ai_msg)

if not ai_msg.tool_calls:
    print("No tool call was requested by the model. Response:")
    print(ai_msg.content)
else:
    tool_call = ai_msg.tool_calls[0]

    tool_result = get_word_length.invoke(
        tool_call["args"]
    )

    msg.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    )

    final_res = llm_with_tools.invoke(msg)
    print("Final Response:")
    print(final_res.content)



"""
class Explanation(BaseModel):
    topic: str
    explanation: str
    example: str
    source: list[str]


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert teacher."
    ),
    (
        "human",
        "Explain {topic} for a {level} student. in few lines"
    )
])

structured_llm = llm.with_structured_output(Explanation)

chain = prompt | structured_llm

result = chain.invoke({
    "topic": "Recursion",
    "level": "beginner"
})

print(result.topic)
print(result.explanation)
print(result.example)
print(result.source)


"""














"""
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a {profession}."
        ),
        (
            "human",
            "Explain {topic} for a {level} learner. in few lines"
        )
    ]
)

chain = prompt | llm
response = chain.invoke(
    {
        "profession": "Teacher",
        "topic":"LCEL",
        "level":"beginner"
    }
)
print(response.content)"""
