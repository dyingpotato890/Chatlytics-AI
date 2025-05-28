import datetime
import os
import re
import asyncio
import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from contextlib import asynccontextmanager

load_dotenv()

# Global variables for keep-alive
keep_alive_task = None
app_url = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global keep_alive_task, app_url
    
    # Try to determine the app URL from environment variables
    app_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("APP_URL")
    
    if app_url:
        print(f"Starting keep-alive service for URL: {app_url}")
        keep_alive_task = asyncio.create_task(keep_alive_service())
    else:
        print("No external URL configured. Keep-alive service disabled.")
    
    yield
    
    # Shutdown
    if keep_alive_task:
        keep_alive_task.cancel()
        try:
            await keep_alive_task
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title="Chat Analyzer API",
    version="1.0.0",
    lifespan=lifespan
)

async def keep_alive_service():
    while True:
        try:
            await asyncio.sleep(14 * 60)
            
            if app_url:
                timeout = aiohttp.ClientTimeout(total=30)  # Fixed: Use ClientTimeout object
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        async with session.get(f"{app_url}/health") as response:
                            if response.status == 200:
                                print(f"Keep-alive ping successful at {datetime.datetime.now()}")
                            else:
                                print(f"Keep-alive ping failed with status {response.status}")
                    except Exception as e:
                        print(f"Keep-alive ping error: {e}")
        except asyncio.CancelledError:
            print("Keep-alive service stopped")
            break
        except Exception as e:
            print(f"Keep-alive service error: {e}")
            await asyncio.sleep(60)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_1")
if not GEMINI_API_KEY:
    print("Error: KEY not found")
    
# Configure Google Gemini API
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY) # type: ignore
        gemini_model = genai.GenerativeModel('gemini-1.5-flash') # type: ignore
        print("Google Gemini API configured and model loaded.")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
else:
    print("Google Gemini API not configured due to missing API key.")

# Models
class ChatMessages(BaseModel):
    messages: List[str]
    
stop_words = set(ENGLISH_STOP_WORDS)

@app.get("/")
async def read_root():
    return {
        "message": "Chat Analyzer API is running.",
        "timestamp": datetime.datetime.now(),
        "keep_alive_active": keep_alive_task is not None and not keep_alive_task.done(),
        "app_url": app_url
    }

@app.get("/wake")
async def wake_endpoint():
    return {
        "message": "Service is awake and ready",
        "timestamp": datetime.datetime.now(),
        "gemini_ready": gemini_model is not None,
        "keep_alive_active": keep_alive_task is not None and not keep_alive_task.done()
    }

@app.post("/analyze_chat/")
async def analyze_chat_endpoint(chat_data: ChatMessages, background_tasks: BackgroundTasks):
    messages = chat_data.messages
    
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No messages found for analysis."
        )

    if not gemini_model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini model not initialized. Please check API key and server logs."
        )

    # Add a background task to log usage (optional)
    background_tasks.add_task(log_analysis_usage, len(messages))

    chat_dialogue = "\n".join(messages)
    summary = None
    sentiment = None
    topics = []
    error_message = None
    used_model = "Google Gemini"

    try:
        # Generate Summary
        summary_prompt = f"Summarize the following chat dialogue concisely, focusing on key events and topics. Keep it under 150 words.\n\nChat Dialogue:\n{chat_dialogue}"
        summary_response = await gemini_model.generate_content_async(summary_prompt)
        summary = summary_response.text.strip()
        
        # Analyze Sentiment
        sentiment_prompt = f"Analyze the overall sentiment of the following chat dialogue. Respond with a single word: 'Positive', 'Negative', or 'Neutral'. Do not add any other text or punctuation.\n\nChat Dialogue:\n{chat_dialogue}"
        sentiment_response = await gemini_model.generate_content_async(sentiment_prompt)
        raw_sentiment = sentiment_response.text.strip().lower()
        
        if "positive" in raw_sentiment:
            sentiment = "Positive"
        elif "negative" in raw_sentiment:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        topics_prompt = f"From the following chat dialogue, identify and list up to 5 main topics or keywords. List them as a comma-separated string, e.g., 'topic1, topic2, topic3'.\n\nChat Dialogue:\n{chat_dialogue}"
        topics_response = await gemini_model.generate_content_async(topics_prompt)
        topics_raw = topics_response.text.strip()
        topics = [t.strip() for t in topics_raw.split(',') if t.strip()]

    except Exception as e:
        print(f"Gemini API call failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemini analysis failed: {e}. Please check the chat dialogue length or API key/rate limits."
        )

    message_count = len(messages)
    word_count = len(" ".join(messages).split())
    avg_message_length = sum(len(m) for m in messages) / message_count if message_count > 0 else 0.0

    return {
        'summary': summary,
        'sentiment': sentiment,
        'topics': topics,
        'messageCount': message_count,
        'wordCount': word_count,
        'avgMessageLength': avg_message_length,
        'error': error_message, # None if successful
        'model': used_model,
        'timestamp': datetime.datetime.now()
    }

async def log_analysis_usage(message_count: int):
    print(f"Analysis completed for {message_count} messages at {datetime.datetime.now()}")

@app.get("/health")
async def health_check():
    return {
        "api_status": "running",
        "timestamp": datetime.datetime.now(),
        "gemini_ready": gemini_model is not None,
        "keep_alive_active": keep_alive_task is not None and not keep_alive_task.done()
    }

@app.get("/test_connection")
async def test_connection_endpoint():
    is_ready = bool(GEMINI_API_KEY and gemini_model)
    
    return {
        "timestamp": datetime.datetime.now(),
        "status": "success" if is_ready else "error",
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_model_loaded": gemini_model is not None,
        "keep_alive_active": keep_alive_task is not None and not keep_alive_task.done(),
        "message": "API ready for analysis" if is_ready else "Missing API key or model failed to load"
    }

@app.get("/status")
async def get_status():
    is_ready = bool(GEMINI_API_KEY and gemini_model)
    return {
        "timestamp": datetime.datetime.now(),
        "status": "ready" if is_ready else "not_ready",
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_ready": gemini_model is not None,
        "keep_alive_running": keep_alive_task is not None and not keep_alive_task.done(),
        "message": "All systems operational" if is_ready else "Configuration incomplete"
    }