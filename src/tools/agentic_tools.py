from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
import math
import requests
import os
import time
from typing import Any
from dotenv import load_dotenv, find_dotenv
from langgraph.types import interrupt
from utils.helper import get_rag_retriever, has_rag_documents

load_dotenv(find_dotenv())

stock_api_key = os.environ.get("STOCK_API_KEY")
openweather_api_key = os.environ.get("OPENWEATHER_API_KEY")
current_location_url = os.environ.get("CURRENT_LOCATION_URL")

# Tools
search_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="advanced",
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for calculating math expressions.
    Input should be a valid math expression.
    Example: "1+1", math.sqrt(16), math.sin(90), etc.
    """

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "ceil": math.ceil,
            "floor": math.floor,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "sqrt": math.sqrt,
            "log": math.log,
            "exp": math.exp,
            "pow": math.pow,
            "pi": math.pi,
            "e": math.e,
            "min": min,
            "max": max,
            "sum": sum,
            "avg": lambda x: sum(x) / len(x),
        }
        result = eval(expression, {"__builtins__": {}}, allowed)

        return str(result)
    except Exception as e:
        return f"Calculator Error: {str(e)}"

# Resuming an approval re-runs `purchase_stock` from the top, so quotes are reused for a
# short while: the price the user approved is then the price that gets recorded, and one
# purchase costs one API call instead of two
QUOTE_CACHE_SECONDS = 300
_quote_cache: dict[str, tuple[float, dict]] = {}


def _get_quote(symbol: str) -> dict | None:
    """Return the latest Alpha Vantage quote for a symbol, or None when unavailable."""
    cached = _quote_cache.get(symbol)
    if cached and time.time() - cached[0] < QUOTE_CACHE_SECONDS:
        return cached[1]

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": stock_api_key,
            },
            timeout=10,
        )
        response.raise_for_status()

        # An unknown symbol or an exhausted daily limit returns a note instead of a quote
        quote = response.json().get("Global Quote") or {}
        float(quote["05. price"])
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None

    _quote_cache[symbol] = (time.time(), quote)
    return quote


def _money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


@tool
def get_stock_price(symbol: str) -> str:
    """
    Get the latest traded price for one stock symbol.

    Args:
        symbol: Ticker symbol, for example "AAPL", "GOOG" or "TSLA".

    Returns:
        The latest price with the day's move and trading range.
    """
    symbol = symbol.strip().upper()
    quote = _get_quote(symbol)

    if not quote:
        return (
            f"No price is available for {symbol} right now. The symbol may be wrong, or "
            "today's market data limit may have been reached."
        )

    return (
        f"{symbol} latest price: {_money(quote.get('05. price'))}\n"
        f"Change: {quote.get('09. change', 'N/A')} "
        f"({quote.get('10. change percent', 'N/A')})\n"
        f"Open: {_money(quote.get('02. open'))}  "
        f"High: {_money(quote.get('03. high'))}  "
        f"Low: {_money(quote.get('04. low'))}\n"
        f"Previous close: {_money(quote.get('08. previous close'))}\n"
        f"Latest trading day: {quote.get('07. latest trading day', 'N/A')}"
    )


@tool
def get_market_status() -> str:
    """
    Report which major financial markets are open or closed right now. Takes no arguments.
    """
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "MARKET_STATUS", "apikey": stock_api_key},
            timeout=10,
        )
        response.raise_for_status()
        markets = response.json().get("markets") or []
    except (requests.RequestException, ValueError) as e:
        return f"Market status is unavailable right now: {e}"

    if not markets:
        return "Market status is unavailable right now."

    lines = [
        f"{market.get('region', 'Unknown')} "
        f"({market.get('primary_exchanges', 'unknown exchanges')}): "
        f"{market.get('current_status', 'unknown')}"
        for market in markets
    ]

    return "Market status:\n" + "\n".join(lines)

@tool
def get_current_weather(location: str) -> str:
    """
    Get the current real-time weather for a given city or location.

    Args:
        location: City or location name, for example:
                  "Dhaka", "London, UK", or "New York, US".

    Returns:
        A formatted current weather report.
    """

    if not openweather_api_key:
        return ("OpenWeather API key is not set. Please set the OPENWEATHER_API_KEY environment variable.")

    try:
        geocoding_url = "https://api.openweathermap.org/geo/1.0/direct"
        geocoding_params = {
            "q": location,
            "limit": 1,
            "appid": openweather_api_key,
        }
        geocoding_response = requests.get(geocoding_url, params=geocoding_params, timeout=10)

        geocoding_response.raise_for_status()

        locations: list[dict[str, Any]] = geocoding_response.json()

        if not locations:
            return "No location found. Please try again with a different location."

        latitude = locations[0]["lat"]
        longitude = locations[0]["lon"]
        resolved_name = locations[0].get("name", location)
        country = locations[0].get("country", "")
        state = locations[0].get("state", "")

        weather_url = "https://api.openweathermap.org/data/2.5/weather"
        weather_params = {
            "lat": latitude,
            "lon": longitude,
            "appid": openweather_api_key,
            "units": "metric",
        }
        weather_response = requests.get(weather_url, params=weather_params, timeout=10)

        weather_response.raise_for_status()
        weather_data: dict[str, Any] = weather_response.json()

        temperature = weather_data["main"]["temp"]
        feels_like = weather_data["main"]["feels_like"]
        humidity = weather_data["main"]["humidity"]
        description = weather_data["weather"][0]["description"]
        wind_speed = weather_data["wind"]["speed"]
        wind_direction = weather_data["wind"]["deg"]
        wind_gust = weather_data["wind"]["gust"]
        wind_direction_text = weather_data["wind"]["dir"]
        visibility = weather_data["visibility"]
        pressure = weather_data["main"]["pressure"]

        visibility_km =(
            round(visibility / 1000, 1)
            if visibility is not None else "N/A"
        )

        location_parts = [resolved_name]

        if state:
            location_parts.append(state)
        if country:
            location_parts.append(country)

        full_location = ", ".join(location_parts)

        return (
            f"Current weather in {full_location}:\n"
            f"Temperature: {temperature}°C\n"
            f"Feels like: {feels_like}°C\n"
            f"Humidity: {humidity}%\n"
            f"Description: {description}\n"
            f"Wind Speed: {wind_speed} m/s\n"
            f"Wind Direction: {wind_direction}°\n"
            f"Wind Gust: {wind_gust} m/s\n"
            f"Wind Direction Text: {wind_direction_text}\n"
            f"Visibility: {visibility_km} km\n"
            f"Pressure: {pressure} hPa\n"
        )

    except requests.Timeout:
        return "Weather API request timed out. Please try again later."

    except requests.HTTPError as e:
        status_code = e.response.status_code if e.response else "Unknown"

        if status_code == 401:
            return "Weather API Error: Invalid API key. Please check your OPENWEATHER_API_KEY environment variable."
        elif status_code == 404:
            return "Weather API Error: Location not found. Please try again with a different location."
        else:
            return f"Weather API Error: HTTP Error {status_code}. Please check your API key and try again."
    except requests.RequestException as e:
        return f"Weather API Error: {str(e)}"
    except (KeyError, TypeError, ValueError) as e:
        return f"Weather API Error: {str(e)}"
    except Exception as e:
        return f"Weather API Error: {str(e)}"

@tool
def get_current_location() -> str | dict:
    """
    Get the user's current approximate geographic location (city, region, country, and coordinates)
    based on their public IP address. Takes no input arguments.
    Use this whenever the user asks about their current location or where they are.
    """
    try:
        if not current_location_url:
            return "Current location URL is not set. Please set the CURRENT_LOCATION_URL environment variable."
        
        decision = interrupt({
            "type": "approval",
            "reason": "Model is about to get the user's current location. Please approve or reject the location.",
            "question": "Get the user's current location?",
            "instruction": "Approve this location? (yes/no)"
        })
        if isinstance(decision, str) and decision.lower() == "no":
            return {"status": "rejected", "message": "Location retrieval rejected by the user."}
        elif isinstance(decision, str) and decision.lower() == "yes":
            response = requests.get(current_location_url, timeout=10)
            response.raise_for_status()
            data = response.json()

            city = data.get("city", "Unknown")
            region = data.get("region", "Unknown")
            country = data.get("country", "Unknown")
            loc = data.get("loc", "Unknown")
            timezone = data.get("timezone", "Unknown")

            return (
                f"Current location: {city}, {region}, {country}\n"
                f"Coordinates (lat, lon): {loc}\n"
                f"Timezone: {timezone}"
            )
        else:
            return {"status": "rejected", "message": "Location retrieval rejected by the user."}
    except requests.Timeout:
        return "Location lookup timed out. Please try again."
    except requests.RequestException as e:
        return f"Location API Error: {str(e)}"
    except (ValueError, KeyError, TypeError) as e:
        return f"Unexpected location response: {str(e)}"

@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Retrieve the most relevant documents from the vector store based on the query. Use this tool when the user asks about the content of the PDF.
    Input should be a valid query.
    Example: "What is Rainfall measurement?"

    Args:
        query: The query to retrieve the most relevant documents from the vector store.
    Returns:
        The most relevant documents from the vector store in a list of strings.

    """
    try:
        thread_id = (config.get("configurable") or {}).get("thread_id")
        if not thread_id or not has_rag_documents(thread_id):
            return "No document has been uploaded in this conversation yet. Ask the user to upload a PDF first."

        retriever = get_rag_retriever(thread_id)
        documents = retriever.invoke(query)

        if not documents:
            return "No relevant documents found for the query."

        formatted_documents = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "Unknown")
            page = document.metadata.get("page", "Unknown")
            formatted_documents.append(f"Document {index} \n"
            f"Source: {source} \n"
            f"Page: {page} \n"
            f"Content: {document.page_content}"
            )

        return "\n\n".join(formatted_documents)
    except Exception as e:
        return f"RAG Tool Error: {str(e)}"

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    The live share price and the order total are looked up here, so report the amounts
    this tool returns rather than working them out yourself.

    NOTE: This is a mock implementation:
    - No real brokerage API is called.
    - It simply returns a confirmation payload.
    """
    symbol = symbol.strip().upper()

    quote = _get_quote(symbol)
    unit_price = float(quote["05. price"]) if quote else None
    total_cost = round(unit_price * quantity, 2) if unit_price is not None else None

    if total_cost is None:
        question = f"Purchase {quantity} shares of {symbol}?"
        details = "Live price unavailable, so the order total cannot be confirmed."
    else:
        question = f"Purchase {quantity} shares of {symbol} for ${total_cost:,.2f}?"
        details = f"{quantity} × ${unit_price:,.2f} per share = ${total_cost:,.2f}"

    # The amount goes in the request so the user approves a known cost, not just a quantity
    decision = interrupt({
        "type": "approval",
        "reason": "Model is about to purchase a stock. Please approve or reject the purchase.",
        "question": question,
        "details": details,
        "instruction": "Approve this purchase? (yes/no)"
    })

    if isinstance(decision, str) and decision.lower() == "yes":
        if total_cost is None:
            message = (
                f"Purchase order placed for {quantity} shares of {symbol}. "
                "The live price was unavailable, so the total cost is unknown."
            )
        else:
            message = (
                f"Purchase order placed for {quantity} shares of {symbol} at "
                f"${unit_price:,.2f} each, totalling ${total_cost:,.2f}."
            )

        return {
            "status": "success",
            "message": message,
            "symbol": symbol,
            "quantity": quantity,
            "unit_price": unit_price,
            "total_cost": total_cost,
        }

    return {
        "status": "rejected",
        "message": f"Purchase order rejected for {quantity} shares of {symbol}.",
        "symbol": symbol,
        "quantity": quantity,
    }
