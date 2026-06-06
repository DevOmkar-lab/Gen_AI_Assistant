import os
import base64
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

st.set_page_config(
    page_title="Gen AI Assistant",
    page_icon="🤖",
    layout="wide"
)

logo_path = Path(__file__).with_name("bot.png")
logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8") if logo_path.exists() else ""

st.markdown("""
<style>

html, body {
    margin: 0;
    padding: 0;
}

.app-loading-overlay{
    position: fixed;
    inset: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: center;
    background: radial-gradient(circle at center, rgba(245, 247, 250, 0.98), rgba(232, 236, 241, 0.98));
    animation: appLoadingFade 1.35s ease 1.1s forwards;
    pointer-events: none;
}

.app-loading-card{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.8rem;
    padding: 1.5rem 1.75rem;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.72);
    box-shadow: 0 18px 60px rgba(0, 0, 0, 0.12);
    backdrop-filter: blur(8px);
}

.app-loading-logo{
    width: 92px;
    height: 92px;
    object-fit: contain;
    border-radius: 24px;
}

.app-loading-text{
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #1f2937;
}

@keyframes appLoadingFade {
    from { opacity: 1; }
    to {
        opacity: 0;
        visibility: hidden;
    }
}

.block-container{
    padding-top:1rem;
}

[data-testid="stSidebar"]{
    display:none;
}

</style>
""", unsafe_allow_html=True)

if logo_b64:
    st.markdown(
        f"""
        <div class="app-loading-overlay" aria-label="Loading application">
            <div class="app-loading-card">
                <img class="app-loading-logo" src="data:image/png;base64,{logo_b64}" alt="App logo" />
                <div class="app-loading-text">Loading Gen AI Assistant</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "openai/gpt-4o-mini"

if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = "GPT 4O Mini"

if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = True


search_tool = TavilySearchResults(max_results=5)


st.title("🤖 Gen AI Assistant")

st.caption(
    f"Model: {st.session_state.selected_model_name} | "
    f"Web Search: {'ON' if st.session_state.web_search_enabled else 'OFF'}"
)

with st.popover("➕"):

    st.markdown("### Select Model")

    selected = st.radio(
        "",
        [
            "GPT 4O Mini",
            "GPT OSS",
            "Gemma 4",
            "Nemotron 3 Super"
        ],
        label_visibility="collapsed"
    )

    mapping = {
        "GPT 4O Mini": "openai/gpt-4o-mini",
        "GPT OSS": "openai/gpt-oss-120b",
        "Gemma 4": "google/gemma-4-27b-it",
        "Nemotron 3 Super": "nvidia/nemotron-3-super-120b-a12b"
    }

    st.session_state.selected_model = mapping[selected]
    st.session_state.selected_model_name = selected

    st.divider()

    st.session_state.web_search_enabled = st.toggle(
        "Web Search",
        value=st.session_state.web_search_enabled
    )

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

def get_model():

    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=st.session_state.selected_model,
        temperature=0.2
    )

def needs_web_search(query):

    router_model = get_model()

    router_prompt = f"""
    Determine whether answering this question requires
    current or internet information.

    Question:
    {query}

    Respond ONLY:
    YES
    or
    NO
    """

    result = router_model.invoke(router_prompt)

    return result.content.strip().upper() == "YES"

def get_search_context(query):

    results = search_tool.invoke(query)

    return f"""
    User Question:
    {query}

    Search Results:
    {results}

    Use the search results when answering.
    """

def get_response(query):

    model = get_model()

    messages = [
        SystemMessage(
            content="""
            You are a professional AI assistant.

            Use:
            - Markdown
            - Headings
            - Bullet points
            - Tables when needed
            - Code blocks for code
            """
        )
    ]

    messages.extend(st.session_state.messages)

    if (
        st.session_state.web_search_enabled
        and needs_web_search(query)
    ):

        messages.append(
            HumanMessage(
                content=get_search_context(query)
            )
        )

    else:

        messages.append(
            HumanMessage(
                content=query
            )
        )

    return model.invoke(messages)

prompt = st.chat_input("Ask anything...") 

if prompt:

    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        HumanMessage(content=prompt)
    )

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = get_response(prompt)

            st.markdown(response.content)

    st.session_state.messages.append(response)

