from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from tools.agentic_tools import calculator, get_stock_price, get_current_weather, get_current_location, search_tool
from langgraph.prebuilt import ToolNode, tools_condition
from pathlib import Path
import os
import sqlite3

load_dotenv(find_dotenv())


gemini_api_key = os.environ.get("GEMINI_API_KEY")
gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
groq_api_key = os.environ.get("GROQ_API_KEY")
groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def _build_models():
    models = []

    if gemini_api_key:
        gemini_candidates = [
            gemini_model,
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]
        seen_gemini = set()
        for model_name in gemini_candidates:
            if model_name and model_name not in seen_gemini:
                seen_gemini.add(model_name)
                models.append(
                    ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0.7,
                        max_retries=1,
                    )
                )

    if groq_api_key:
        groq_candidates = [
            groq_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]
        seen_groq = set()
        for model_name in groq_candidates:
            if model_name and model_name not in seen_groq:
                seen_groq.add(model_name)
                models.append(
                    ChatGroq(
                        model=model_name,
                        temperature=0.7,
                        max_retries=1,
                    )
                )

    if not models:
        raise RuntimeError(
            "No LLM API keys configured. Set GEMINI_API_KEY or GROQ_API_KEY in .env."
        )

    return models


_all_models = _build_models()
model = (
    _all_models[0].with_fallbacks(_all_models[1:])
    if len(_all_models) > 1
    else _all_models[0]
)

tools = [calculator, get_stock_price, get_current_weather, get_current_location, search_tool]

llm_with_tools = (
    _all_models[0]
    .bind_tools(tools)
    .with_fallbacks([m.bind_tools(tools) for m in _all_models[1:]])
    if len(_all_models) > 1
    else _all_models[0].bind_tools(tools)
)


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatbotState):
    """LLM node that may answer or use tools based on the user's input."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

# Anchored to the project root so the database is the same no matter where the app is launched from
DB_PATH = Path(__file__).resolve().parent.parent / "chatbot.db"

conn = sqlite3.connect(database=str(DB_PATH), check_same_thread=False)

checkpointer = SqliteSaver(conn)

# Tracked separately from the LangGraph-managed `checkpoints` table, which has no
# per-thread row to hang a creation timestamp on
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS threads (
        thread_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        is_deleted INTEGER NOT NULL DEFAULT 0
    )
    """
)

# Databases created before soft deletes still have the two-column table
if "is_deleted" not in {row[1] for row in conn.execute("PRAGMA table_info(threads)")}:
    conn.execute(
        "ALTER TABLE threads ADD COLUMN is_deleted INTEGER NOT NULL DEFAULT 0"
    )

conn.commit()

graph = StateGraph(ChatbotState)

graph.add_node('tools', tool_node)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition, 'tools')
graph.add_edge('tools', 'chat_node')

chatbot_workflow = graph.compile(checkpointer=checkpointer)
