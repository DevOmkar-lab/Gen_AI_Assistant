from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

from tools.webScapeTool import scrape_website

load_dotenv()

def reader_Agent():
    return create_agent(
        model=init_chat_model(
            "openrouter:openai/gpt-4o-mini",
            temperature=0.5,
            max_tokens=2000
        ),
        tools=[scrape_website]
    )
