from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path
import os
import sqlite3

load_dotenv(find_dotenv())



gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_model = os.environ["GEMINI_MODEL"]

model = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.7)


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatbotState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# Anchored to the project root so the database is the same no matter where the app is launched from
DB_PATH = Path(__file__).resolve().parent.parent / "chatbot.db"

conn = sqlite3.connect(database=str(DB_PATH), check_same_thread=False)

checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatbotState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot_workflow = graph.compile(checkpointer=checkpointer)
