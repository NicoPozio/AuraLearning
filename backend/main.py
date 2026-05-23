from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from groq import AsyncGroq
import httpx
import os
import io
from dotenv import load_dotenv

# Safely load environment variables from the .env file
load_dotenv()

app = FastAPI(title="Aura Proxy Backend")

# Strict CORS configuration to allow cross-origin requests from the web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LLM (Large Language Model) client
LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY mancante nel file .env")
llm_client = AsyncGroq(api_key=LLM_API_KEY)

# Initialize ElevenLabs API Key and specific Voice ID for Text-to-Speech
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "N2lVS1w4EtoT3dr4eOWO"  # Callum's voice ID

# --- DATA MODELS ---

# Represents a single message in the chat history sent by the frontend
class ChatMessage(BaseModel):
    role: str                          # Either "user" or "model"
    parts: List[dict]                  # E.g., [{"text": "..."}]

# Payload structure for the chat request
class ChatRequest(BaseModel):
    user_text: str
    emotion_state: str
    slide_context: str
    chat_history: Optional[List[ChatMessage]] = []   # Last 10 chat messages

# Payload structure for the Text-to-Speech request
class TTSRequest(BaseModel):
    text: str

# --- ENDPOINTS ---
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Inject slide context if provided by the frontend
        context_injection = ""
        if request.slide_context:
            context_injection = f"Ecco il testo delle slide:\n{request.slide_context}\n"

        # Construct the System Prompt guiding the AI tutor's behavior
        system_prompt = f"""Sei Aura, un brillante tutor universitario. 
            L'utente ha un livello di attenzione: {request.emotion_state}.
            {context_injection}
            
            REGOLE STRETTE DI RISPOSTA:
            1. NON limitarti a tradurre o riassumere il testo. Spiega il CONCETTO in modo naturale e discorsivo, come se stessi parlando a uno studente.
            2. Usa un italiano eccellente e colloquiale. Mantieni i termini tecnici specifici in inglese (es. "Control-Value Theory", "gaze-away") senza inventare traduzioni macchinose o inesistenti (VIETATO usare parole come "addressato").
            3. Se l'utente usa parole come "questo" o "questa cosa", sostituiscile tu nella spiegazione con il concetto reale a cui si riferisce il testo.
            4. Vai dritto al punto senza presentarti. Sii concisa (massimo 5 frasi brevi) per permettere una sintesi vocale fluida.
            5. ASSOLUTAMENTE NESSUNA FORMATTAZIONE (niente Markdown, elenchi, o simboli speciali). Solo testo leggibile a voce."""
        # Build the messages array: system prompt + chat history + current user message
        # The frontend sends history using Gemini's role format ("user"/"model").
        # We map "model" to "assistant" to make it compatible with Groq/OpenAI APIs.
        messages = [{"role": "system", "content": system_prompt}]

        for msg in (request.chat_history or []):
            groq_role = "assistant" if msg.role == "model" else "user"
            text_content = msg.parts[0].get("text", "") if msg.parts else ""
            if text_content.strip():
                messages.append({"role": groq_role, "content": text_content})

        # Append the current message from the user
        messages.append({"role": "user", "content": request.user_text})

        # Asynchronous call to the LLaMA 3 model via Groq
        response = await llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=150  # Keep the response short and concise
        )

        # Extract and return the generated text
        answer = response.choices[0].message.content.strip()
        return {"text": answer}

    except Exception as e:
        # Print the actual stack trace in the terminal for debugging purposes
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore LLM: {str(e)}")

@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    # Validate that the ElevenLabs API key is present
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API Key mancante nel server.")

    # Prepare the ElevenLabs API request URL and headers
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?optimize_streaming_latency=3"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    # Prepare the TTS payload using the requested text and a specific multilingual model
    payload = {
        "text": request.text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    # Make an asynchronous HTTP POST request to ElevenLabs
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=30.0)

    # Handle potential errors from the TTS provider
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Errore dal provider TTS")

    # Return the generated audio as a streaming response
    return StreamingResponse(io.BytesIO(response.content), media_type="audio/mpeg")

# Mount the frontend directory to serve static HTML/JS/CSS files at the root URL
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

# Entry point to run the application using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)