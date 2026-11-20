from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, BaseMessage
from typing import TypedDict, Literal, Annotated
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv, find_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages

import os

load_dotenv(find_dotenv())

checkpointer = InMemorySaver()


gemini_api_key = os.environ["GEMINI_API_KEY"]
gemini_model = os.environ["GEMINI_MODEL"]

model = ChatGoogleGenerativeAI(model=gemini_model)


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatbotState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}


checkpointer = InMemorySaver()

graph = StateGraph(ChatbotState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot_workflow = graph.compile(checkpointer=checkpointer)


