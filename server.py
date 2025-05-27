import os
import re
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List, Dict, Union
import google.generativeai as genai
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

load_dotenv()

app = FastAPI(
    title = "Chat Analyzer API",
    version = "1.0.0"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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

def _extract_words_from_text(text: str) -> List[str]:
    # Tokenize using regex, convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())

    # Filter out stopwords and short words
    filtered_words = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]

    # Count word frequencies
    word_counts = {}
    for word in filtered_words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Return top 5 most frequent words
    sorted_words = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    return [word for word, count in sorted_words[:5]]


@app.get("/")
async def read_root():
    return {"message": "Chat Analyzer API is running. Go to /docs for API documentation."}

@app.post("/analyze_chat/")
async def analyze_chat_endpoint(chat_data: ChatMessages):
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
        'model': used_model
    }

@app.get("/health")
async def health_check():
    status_dict = {
        "api_status": "running",
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_model_loaded": gemini_model is not None
    }
    if not status_dict["gemini_model_loaded"]:
        status_dict["warning"] = "Gemini model failed to load. Analysis endpoint will fail."
    return status_dict

@app.get("/test_connection")
async def test_connection_endpoint():
    """
    Test API connection and Gemini model availability without consuming tokens.
    Uses model configuration checks instead of actual API calls.
    """
    test_results = {
        "api_status": "running",
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_model_loaded": gemini_model is not None,
        "status": "success",
        "message": "Connection test completed without consuming tokens"
    }
    
    # Check if all prerequisites are met
    if not GEMINI_API_KEY:
        test_results["status"] = "error"
        test_results["message"] = "Gemini API key not configured"
        test_results["error_details"] = "GEMINI_API_KEY environment variable is missing"
        return test_results
    
    if not gemini_model:
        test_results["status"] = "error"
        test_results["message"] = "Gemini model not initialized"
        test_results["error_details"] = "Model failed to load during startup. Check API key validity and network connection."
        return test_results
    
    # Optional: Test basic model configuration (still no token usage)
    try:
        # This just checks if the model object has the expected attributes
        model_name = gemini_model.model_name if hasattr(gemini_model, 'model_name') else "gemini-1.5-flash"
        test_results["model_name"] = model_name
        test_results["message"] = f"API ready. Model '{model_name}' is configured and ready for analysis."
        
    except Exception as e:
        test_results["status"] = "warning"
        test_results["message"] = "Model configured but may have issues"
        test_results["warning_details"] = str(e)
    
    return test_results

@app.get("/validate_setup")
async def validate_setup():
    validation = {
        "timestamp": "2024-01-01T00:00:00Z",  # You might want to add actual timestamp
        "checks": {
            "environment": {
                "gemini_api_key_exists": bool(os.getenv("GEMINI_API_KEY")),
                "gemini_api_key_length": len(GEMINI_API_KEY) if GEMINI_API_KEY else 0,
                "gemini_api_key_format_valid": bool(GEMINI_API_KEY and GEMINI_API_KEY.startswith('AI')),
            },
            "api_configuration": {
                "genai_configured": bool(GEMINI_API_KEY),
                "model_initialized": gemini_model is not None,
                "model_type": type(gemini_model).__name__ if gemini_model else None,
            },
            "dependencies": {
                "fastapi_available": True,  # If we're running, FastAPI is available
                "google_generativeai_available": 'genai' in globals(),
                "sklearn_available": 'ENGLISH_STOP_WORDS' in globals(),
                "dotenv_loaded": bool(os.getenv("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")),
            }
        },
        "overall_status": "unknown",
        "recommendations": []
    }
    
    # Determine overall status
    critical_checks = [
        validation["checks"]["environment"]["gemini_api_key_exists"],
        validation["checks"]["api_configuration"]["genai_configured"],
        validation["checks"]["api_configuration"]["model_initialized"],
    ]
    
    if all(critical_checks):
        validation["overall_status"] = "ready"
        validation["message"] = "All systems ready for chat analysis"
    elif validation["checks"]["environment"]["gemini_api_key_exists"]:
        validation["overall_status"] = "partial"
        validation["message"] = "API key present but model initialization failed"
        validation["recommendations"].append("Check API key validity and network connectivity")
    else:
        validation["overall_status"] = "not_ready"
        validation["message"] = "Missing required configuration"
        validation["recommendations"].append("Set GEMINI_API_KEY environment variable")
    
    # Add specific recommendations based on failed checks
    if not validation["checks"]["environment"]["gemini_api_key_format_valid"]:
        validation["recommendations"].append("Verify API key format (should start with 'AI')")
    
    if validation["checks"]["environment"]["gemini_api_key_length"] > 0 and validation["checks"]["environment"]["gemini_api_key_length"] < 20:
        validation["recommendations"].append("API key seems too short, verify it's complete")
    
    return validation