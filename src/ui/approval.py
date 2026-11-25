import streamlit as st

from agentic_chatbot_backend import chatbot_workflow
from ui import components

# The strings `purchase_stock` and `get_current_location` compare their resume value against
APPROVE = "yes"
DECLINE = "no"


def get_pending_approval(thread_id):
    """Return the request a paused tool is waiting on, or None when nothing is paused.

    `interrupt()` stops the graph mid-tool and writes the pause into the checkpoint, so the
    saved state is what survives a browser refresh or a switch between conversations.
    """
    try:
        snapshot = chatbot_workflow.get_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        # A brand new thread has no checkpoint to read yet
        return None

    interrupts = snapshot.interrupts or ()
    if not interrupts:
        return None

    request = interrupts[0].value
    if not isinstance(request, dict):
        request = {"question": str(request)}

    # The tool call the graph stopped inside, so the card can name what is about to run
    messages = snapshot.values.get("messages", [])
    tool_calls = getattr(messages[-1], "tool_calls", None) if messages else None
    pending_call = tool_calls[0] if tool_calls else {}

    return {
        "question": request.get("question", "Approve this action?"),
        "details": request.get("details", ""),
        "reason": request.get("reason", ""),
        "tool": pending_call.get("name", ""),
        "args": pending_call.get("args", {}),
    }


def render_approval_card(request):
    """Draw the approval prompt in the current container and return the decision.

    Returns None while the user decides. The card is cleared as soon as a button is
    pressed so it is not left on screen while the answer streams in below it.
    """
    placeholder = st.empty()
    decision = None

    with placeholder.container(key="approval_card"):
        st.markdown(components.approval_card(request), unsafe_allow_html=True)

        with st.container(horizontal=True, key="approval_actions", gap="small"):
            if st.button(
                "Approve", key="approve_action", type="primary", icon=":material/check:"
            ):
                decision = APPROVE

            if st.button("Decline", key="decline_action", icon=":material/close:"):
                decision = DECLINE

    if decision:
        placeholder.empty()

    return decision
