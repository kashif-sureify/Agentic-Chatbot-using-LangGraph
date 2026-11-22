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


def generate_title(user_message: str, assistant_message: str = "") -> str:
    prompt = (
        "Generate a concise 3 to 5 word title for the following conversation. "
        "Return only the title text, without quotes or trailing punctuation.\n\n"
        f"User: {user_message}\n"
        f"Assistant: {assistant_message[:300]}"
    )
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        title = response.text.strip().strip("\"'.,!?")
        if title:
            return title
    except Exception:
        pass

    fallback = user_message.strip().splitlines()[0] if user_message.strip() else "New Chat"
    return fallback[:35] + "..." if len(fallback) > 35 else fallback


def generate_greeting(current_time: str) -> str:
    prompt = (
        f"The current local time is {current_time}. "
        "Generate one short, friendly, creative greeting (3 to 7 words) for an AI chat app "
        "welcome screen tailored to this time of day (similar in vibe to 'Hello, night owl' or "
        "'Good morning, early bird'). Return only the greeting text, with no quotes or emojis."
    )
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        greeting = response.text.strip().strip("\"'")
        if greeting:
            return greeting
    except Exception:
        pass

    return "How can I help you today?"


