from langchain_core.messages import HumanMessage
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from models.models import model
import os
import time

load_dotenv(find_dotenv())

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    output_dimensionality=EMBEDDING_DIMENSIONS,
)


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


# RAG Document Ingestion
# One folder per conversation, so a search in this thread cannot see another thread's PDFs
FAISS_INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "faiss_index"
K_RETRIEVER = 4

# Gemini meters embeddings per minute, so a long PDF is embedded one batch at a time
# with a pause in between instead of firing every batch at once
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "100"))
EMBEDDING_THROTTLE_SECONDS = float(os.environ.get("EMBEDDING_THROTTLE_SECONDS", "60"))


def _index_dir(thread_id: str) -> Path:
    if not thread_id or thread_id in {".", ".."} or "/" in thread_id or "\\" in thread_id:
        raise ValueError("Invalid thread id for the document index.")
    return FAISS_INDEX_PATH / thread_id


def has_rag_documents(thread_id: str) -> bool:
    return (_index_dir(thread_id) / "index.faiss").exists()


def _load_vectorstore(thread_id: str):
    vectorstore = FAISS.load_local(
        folder_path=str(_index_dir(thread_id)),
        embeddings=embeddings,
        allow_dangerous_deserialization=True
        )

    # Searching an index built at a different size fails deep inside FAISS, so say so here
    if vectorstore.index.d != EMBEDDING_DIMENSIONS:
        raise RuntimeError(
            f"The index at {_index_dir(thread_id)} stores {vectorstore.index.d}-dimension vectors "
            f"but EMBEDDING_DIMENSIONS is {EMBEDDING_DIMENSIONS}. Delete the index folder and "
            "upload the documents again."
        )

    return vectorstore


def _embed_in_batches(texts: list[str], on_progress=None) -> list[list[float]]:
    """Embed texts one batch per interval so long documents stay inside the rate limit."""
    vectors: list[list[float]] = []

    if on_progress:
        on_progress(0, len(texts))

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        vectors.extend(
            embeddings.embed_documents(texts[start : start + EMBEDDING_BATCH_SIZE])
        )

        if on_progress:
            on_progress(len(vectors), len(texts))

        if len(vectors) < len(texts):
            time.sleep(EMBEDDING_THROTTLE_SECONDS)

    return vectors


def ingest_rag_documents(
    file_path: str,
    thread_id: str,
    source_name: str | None = None,
    on_progress=None,
):
    loader = PyMuPDF4LLMLoader(file_path)
    docs = loader.load()

    # Uploads land in a temp file, so cite the name the user actually recognises
    if source_name:
        for doc in docs:
            doc.metadata["source"] = source_name

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    text_embeddings = list(zip(texts, _embed_in_batches(texts, on_progress)))

    # Added to this thread's index so several PDFs in the same chat stay searchable together
    index_dir = _index_dir(thread_id)
    if has_rag_documents(thread_id):
        vectorstore = _load_vectorstore(thread_id)
        vectorstore.add_embeddings(text_embeddings, metadatas=metadatas)
    else:
        vectorstore = FAISS.from_embeddings(text_embeddings, embeddings, metadatas=metadatas)

    vectorstore.save_local(str(index_dir))


def estimated_minutes(chunk_count: int) -> int:
    """Rough wall-clock estimate for embedding, dominated by the waits between batches."""
    pauses = max(0, -(-chunk_count // EMBEDDING_BATCH_SIZE) - 1)
    return round(pauses * EMBEDDING_THROTTLE_SECONDS / 60)

def get_rag_retriever(thread_id: str):
    vectorstore = _load_vectorstore(thread_id)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": K_RETRIEVER})
    return retriever
