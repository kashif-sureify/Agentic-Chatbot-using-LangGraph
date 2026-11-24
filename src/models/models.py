from dotenv import load_dotenv, find_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv(find_dotenv())


gemini_api_key = os.environ.get("GEMINI_API_KEY")
gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
groq_api_key = os.environ.get("GROQ_API_KEY")
groq_model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def _build_models():
    models = []

    if gemini_api_key:
        gemini_candidates = [
            gemini_model,
            "gemini-flash-lite-latest",
            "gemini-flash-latest",
        ]
        seen_gemini = set()
        for model_name in gemini_candidates:
            if model_name and model_name not in seen_gemini:
                seen_gemini.add(model_name)
                models.append(
                    ChatGoogleGenerativeAI(
                        model=model_name,
                        temperature=0.7,
                        max_retries=1,
                    )
                )

    if groq_api_key:
        groq_candidates = [
            groq_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ]
        seen_groq = set()
        for model_name in groq_candidates:
            if model_name and model_name not in seen_groq:
                seen_groq.add(model_name)
                models.append(
                    ChatGroq(
                        model=model_name,
                        temperature=0.7,
                        max_retries=1,
                    )
                )

    if not models:
        raise RuntimeError(
            "No LLM API keys configured. Set GEMINI_API_KEY or GROQ_API_KEY in .env."
        )

    return models


all_models = _build_models()

# Plain model for helper tasks such as titles and greetings; the graph binds its own
# tool-aware copy from `all_models`
model = (
    all_models[0].with_fallbacks(all_models[1:])
    if len(all_models) > 1
    else all_models[0]
)

