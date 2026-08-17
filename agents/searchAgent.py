from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from tools.webSearchTool import web_search

load_dotenv()

def search_Agent():
    model = init_chat_model(
        "openrouter:openai/gpt-4o-mini",
        temperature=0,
        max_tokens=1000
    )

    return create_agent(
        model=model,
        tools=[web_search]
    )
