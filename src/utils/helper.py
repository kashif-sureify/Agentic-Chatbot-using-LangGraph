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
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)
