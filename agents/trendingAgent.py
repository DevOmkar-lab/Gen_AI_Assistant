from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from tools.webSearchTool import web_search

load_dotenv()


def trending_Agent():
    model = init_chat_model(
        "openrouter:openai/gpt-4o-mini",
        temperature=0.2,
        max_tokens=1200,
    )

    return create_agent(
        model=model,
        tools=[web_search],
    )
