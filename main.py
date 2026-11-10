
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai.errors import ClientError, ServerError
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env
load_dotenv()

# Read the Gemini API key securely
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found. Check your .env file."
    )

# Configure the Gemini client
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=30000),
)
#MODEL_NAME = "gemini-3.8-flash"
MODEL_NAME = "gemini-3.5-flash-lite"


# Create the FastAPI application
app = FastAPI(
    title="My First LLM Application",
    description="A simple API powered by Google Gemini",
    version="1.0.0",
)


# Define the incoming request structure
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)


# Basic welcome endpoint
@app.get("/")
def home():
    return {
        "message": "Welcome to my first LLM application!"
    }


# LLM chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=request.message,
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response.",
            )

        return {
            "model": MODEL_NAME,
            "response": response.text,
        }

    except HTTPException:
        raise
    except ClientError as exc:
        logging.warning("Gemini request rejected: %s", exc)
        raise HTTPException(
            status_code=429,
            detail=(
                "Gemini rejected the request. Your free-tier quota "
                "may be exhausted. Check your Google AI Studio usage."
            ),
        )
    except ServerError as exc:
        logging.warning(
            "Gemini service unavailable: %s",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is temporarily overloaded. "
                "Please try again shortly."
            ),
        )

    except Exception:
        logging.exception("Gemini API request failed")
        raise HTTPException(
            status_code=502,
            detail=(
                "Gemini API request failed. "
                "Check the server logs and API configuration."
            ),
        )
