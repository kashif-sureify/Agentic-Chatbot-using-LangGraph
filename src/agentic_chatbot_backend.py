from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from tools.agentic_tools import calculator, get_stock_price, get_current_weather, get_current_location, search_tool
from langgraph.prebuilt import ToolNode, tools_condition
from db.database import checkpointer
from models.models import _build_models

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

graph = StateGraph(ChatbotState)

graph.add_node('tools', tool_node)
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition, 'tools')
graph.add_edge('tools', 'chat_node')

chatbot_workflow = graph.compile(checkpointer=checkpointer)
