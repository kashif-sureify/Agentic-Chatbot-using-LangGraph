from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, SystemMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from tools.agentic_tools import calculator, get_stock_price, get_market_status, get_current_weather, get_current_location, search_tool, rag_tool, purchase_stock
from langgraph.prebuilt import ToolNode, tools_condition
from db.database import checkpointer
from models.models import all_models

tools = [calculator, get_stock_price, get_market_status, get_current_weather, get_current_location, search_tool, rag_tool, purchase_stock]

llm_with_tools = (
    all_models[0]
    .bind_tools(tools)
    .with_fallbacks([m.bind_tools(tools) for m in all_models[1:]])
    if len(all_models) > 1
    else all_models[0].bind_tools(tools)
)


class ChatbotState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are Magnus AI, a helpful assistant with access to several tools.\n\n"

        "Tool usage instructions:\n"
        "- Use `rag_tool` for questions about PDFs uploaded in this conversation. Answer "
        "only from what it returns, and cite the file names and page numbers it reports. "
        "Ignore documents from other conversations. If it reports that nothing has been "
        "uploaded, ask the user to attach a PDF.\n"
        "- Use `tavily_search` for current events, recent information, or anything else "
        "that requires an internet search.\n"
        "- Use `calculator` for mathematical calculations. Do not work expressions out "
        "yourself when the calculator is available.\n"
        "- Use `get_current_weather` when the user asks about the weather for a named city "
        "or location.\n"
        "- Use `get_current_location` when the user asks where they are. It reads their "
        "approximate location from their IP address, so do not search the web for this.\n"
        "- Use `get_stock_price` when the user asks what a share costs. It takes one ticker "
        "symbol, so map the company to its symbol first, for example Apple to AAPL.\n"
        "- Use `get_market_status` when the user asks whether the markets are open.\n"
        "- Use `purchase_stock` when the user wants to buy shares. It prices the order and "
        "asks the user to approve it, so report the amounts it returns and never estimate "
        "the cost yourself.\n\n"

        "Answer directly when no tool is required. Never invent document contents, prices, "
        "weather, or locations. If a tool returns nothing useful or fails, tell the user "
        "plainly what you could not find rather than repeating the raw error. After a tool "
        "runs, reply with a clear, self-contained answer instead of restating its output."
    )
)


def chat_node(state: ChatbotState):
    """LLM node that may answer or use tools based on the user's input."""
    messages = [SYSTEM_MESSAGE, *state["messages"]]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatbotState)

graph.add_node('tools', tool_node)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition, 'tools')
graph.add_edge('tools', 'chat_node')

chatbot_workflow = graph.compile(checkpointer=checkpointer)
