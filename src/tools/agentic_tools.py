from langchain_tavily import TavilySearch
from langchain_core.tools import tool
import math
import requests
import os
from typing import Any
from dotenv import load_dotenv, find_dotenv
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

@tool
def get_stock_price() -> dict:
    """
    Useful for getting the price of a stock.
    Input should be a valid stock symbol.
    Example: "AAPL", "GOOG", "MSFT", etc.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=MARKET_STATUS&apikey={stock_api_key}"
        response = requests.get(url)
        data = response.json()
        return data
    except Exception as e:
        return f"Stock Price Error: {str(e)}"

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
def get_current_location() -> str:
    """
    Get the user's current approximate geographic location (city, region, country, and coordinates)
    based on their public IP address. Takes no input arguments.
    Use this whenever the user asks about their current location or where they are.
    """
    try:
        if not current_location_url:
            return "Current location URL is not set. Please set the CURRENT_LOCATION_URL environment variable."
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
    except requests.Timeout:
        return "Location lookup timed out. Please try again."
    except requests.RequestException as e:
        return f"Location API Error: {str(e)}"
    except (ValueError, KeyError, TypeError) as e:
        return f"Unexpected location response: {str(e)}"

@tool
def rag_tool(query: str) -> str:
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
        if not has_rag_documents():
            return "No document has been uploaded yet. Ask the user to upload a PDF first."

        retriever = get_rag_retriever()
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
