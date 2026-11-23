from langchain_core.messages import HumanMessage

from agentic_chatbot_backend import checkpointer, model


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


def get_all_threads():
    """Return thread IDs ordered by when each conversation was created (oldest first)."""
    created_at = {}

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        # The earliest checkpoint of a thread marks when that conversation started
        timestamp = checkpoint.checkpoint.get("ts", "")
        if thread_id not in created_at or timestamp < created_at[thread_id]:
            created_at[thread_id] = timestamp

    return sorted(created_at, key=lambda thread_id: created_at[thread_id])


def get_friendly_error_message(error: Exception) -> str:
    msg = str(error).lower()

    if any(k in msg for k in ("503", "unavailable", "high demand", "overloaded", "capacity")):
        return "Our AI models are currently busy due to high demand. Please try again in a moment."

    if any(k in msg for k in ("429", "rate limit", "rate_limit", "quota", "resource_exhausted")):
        return "Too many requests right now. Please wait a few seconds and try again."

    if any(k in msg for k in ("401", "403", "api key", "api_key", "permission_denied", "unauthorized")):
        return "There is a configuration issue with the AI service. Please try again later."

    if any(k in msg for k in ("timeout", "timed out", "connection", "network", "unreachable")):
        return "Unable to reach the AI service right now. Please check your connection and try again."

    return "Something went wrong while generating a response. Please try again."
