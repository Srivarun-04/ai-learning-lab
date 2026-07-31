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
