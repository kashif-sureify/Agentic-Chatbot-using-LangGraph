"""HTML snippets for the parts of the UI that Streamlit has no widget for.

Every builder returns one line of markup: Markdown treats a blank line as the end of an
HTML block and an indented line as code, so the pieces are joined without either.
Anything that came from the model, a tool or the user is escaped before it is inserted.
"""

import html
import json
import re

from ui.icons import icon

TOOL_ICONS = {
    "calculator": "calculator",
    "get_current_location": "map-pin",
    "get_current_weather": "cloud-sun",
    "get_market_status": "landmark",
    "get_stock_price": "trending-up",
    "purchase_stock": "shopping-cart",
    "rag_tool": "file-text",
    "tavily_search": "globe",
}

RESULT_PREVIEW_CHARS = 160


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def result_preview(raw) -> str:
    """One readable line of a tool result for the activity panel."""
    text = raw if isinstance(raw, str) else str(raw)

    # `purchase_stock` and the rejected-location path return a dict, which reaches the
    # UI as JSON; its "message" field is the sentence worth showing
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and parsed.get("message"):
        text = str(parsed["message"])

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > RESULT_PREVIEW_CHARS:
        text = text[: RESULT_PREVIEW_CHARS - 1].rstrip() + "…"
    return text


def thinking(label: str = "Thinking") -> str:
    return (
        '<div class="mg-thinking" role="status">'
        '<span class="mg-dots" aria-hidden="true"><i></i><i></i><i></i></span>'
        f"{esc(label)}…</div>"
    )


def tool_steps(steps) -> str:
    rows = "".join(
        '<div class="mg-step">'
        f'<span class="mg-step__chip">{icon(TOOL_ICONS.get(step["name"], "wrench"), 14)}{esc(step["name"])}</span>'
        f'<span class="mg-step__res" title="{esc(step["result"])}">{esc(step["result"])}</span>'
        "</div>"
        for step in steps
    )
    return f'<div class="mg-steps">{rows}</div>'


def file_chip(name: str, meta: str, state: str = "attached") -> str:
    status = icon("circle-check", 16, "mg-file__ok") if state == "indexed" else ""
    return (
        f'<div class="mg-file mg-file--{state}">'
        f'<span class="mg-file__tile">{icon("file-text", 16)}</span>'
        '<span class="mg-file__text">'
        f'<span class="mg-file__name" title="{esc(name)}">{esc(name)}</span>'
        f'<span class="mg-file__meta">{esc(meta)}</span>'
        "</span>"
        f"{status}</div>"
    )


def file_chips(names, state: str = "attached") -> str:
    meta = "Indexed for this chat" if state == "indexed" else "PDF"
    return '<div class="mg-files">' + "".join(file_chip(n, meta, state) for n in names) + "</div>"


def indexing_card(name: str, embedded: int, total: int, minutes_left: int) -> str:
    if total:
        percent = round(100 * embedded / total)
        label = f"Embedding {embedded} of {total} chunks"
        if minutes_left:
            label += f" · about {minutes_left} min left"
    else:
        percent = 0
        label = f"Reading {name} …"

    return (
        '<div class="mg-indexing" role="status">'
        '<div class="mg-indexing__top">'
        f'<span class="mg-file__tile">{icon("file-text", 16)}</span>'
        '<span class="mg-file__text">'
        f'<span class="mg-file__name" title="{esc(name)}">{esc(name)}</span>'
        f'<span class="mg-file__meta">{esc(label)}</span>'
        "</span>"
        f'<span class="mg-indexing__pct">{percent}%</span>'
        "</div>"
        f'<div class="mg-track"><i style="width:{percent}%"></i></div>'
        '<p class="mg-indexing__note">Large PDFs are embedded in batches with a pause between '
        "them to stay within the rate limit.</p>"
        "</div>"
    )


def decision_chip(tool: str, approved: bool) -> str:
    if approved:
        return f'<div class="mg-decision mg-decision--ok">{icon("check", 14)}You approved <code>{esc(tool)}</code></div>'
    return f'<div class="mg-decision mg-decision--no">{icon("x", 14)}You declined <code>{esc(tool)}</code></div>'


def topbar(title: str, docs) -> str:
    pill = ""
    if docs:
        label = docs[0] if len(docs) == 1 else f"{len(docs)} PDFs"
        pill = (
            f'<span class="mg-docpill" title="{esc(", ".join(docs))}">'
            f'{icon("file-text", 14)}<span>{esc(label)}</span></span>'
        )
    return f'<div class="mg-topbar"><div class="mg-topbar__title">{esc(title)}</div>{pill}</div>'


CAPABILITIES = [
    ("globe", "Web search"),
    ("cloud-sun", "Weather"),
    ("trending-up", "Stocks"),
    ("calculator", "Calculator"),
    ("file-text", "Your PDFs"),
    ("map-pin", "Location"),
]


def welcome(greeting: str, logo_b64: str) -> str:
    # The words after the first comma get the brand gradient ("Hello, <night owl>")
    lead, sep, rest = greeting.partition(",")
    if sep and rest.strip():
        heading = f'{esc(lead)}, <span class="mg-grad">{esc(rest.strip())}</span>'
    else:
        heading = esc(greeting)

    caps = "".join(f'<span class="mg-cap">{icon(i, 14)}{esc(label)}</span>' for i, label in CAPABILITIES)
    return (
        '<section class="mg-welcome">'
        f'<div class="mg-halo"><img src="data:image/png;base64,{logo_b64}" alt="Magnus logo"></div>'
        # A div with a heading role: Markdown would add an anchor link to a real <h2>
        f'<div class="mg-greeting" role="heading" aria-level="2">{heading}</div>'
        '<p class="mg-sub">How can I help you today?</p>'
        f'<div class="mg-caps" aria-label="What Magnus can use">{caps}</div>'
        "</section>"
    )


def suggestion_card(icon_name: str, tone: str, title: str, text: str) -> str:
    return (
        '<div class="mg-sug">'
        f'<span class="mg-tile mg-tile--{tone}">{icon(icon_name, 20)}</span>'
        f'<span class="mg-sug__text"><b>{esc(title)}</b><span>{esc(text)}</span></span>'
        "</div>"
    )


def approval_card(request) -> str:
    tool = request["tool"] or "a tool"
    details = (
        f'<div class="mg-approval__calc">{icon("calculator", 16)}<span>{esc(request["details"])}</span></div>'
        if request["details"]
        else ""
    )
    chips = "".join(
        f'<span class="mg-arg">{esc(name)}: <b>{esc(value)}</b></span>'
        for name, value in request["args"].items()
    )
    arguments = f'<div class="mg-approval__args">{chips}</div>' if chips else ""
    reason = f'<p class="mg-approval__reason">{esc(request["reason"])}</p>' if request["reason"] else ""

    return (
        '<div class="mg-approval__head">'
        f'<span class="mg-badge mg-badge--warn">{icon("shield-alert", 14)}Approval required</span>'
        "<span>Magnus wants to run</span>"
        f'<code class="mg-toolname">{esc(tool)}</code>'
        "</div>"
        f'<div class="mg-approval__question" role="heading" aria-level="3">{esc(request["question"])}</div>'
        f"{details}{arguments}{reason}"
    )


def overline(text: str) -> str:
    return f'<div class="mg-overline">{esc(text)}</div>'


def empty_threads(query: str) -> str:
    if query:
        return f'<p class="mg-empty">No conversations match “{esc(query)}”.</p>'
    return '<p class="mg-empty">Your conversations will appear here.</p>'
