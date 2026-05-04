from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from groq import AsyncGroq
import httpx
import os
import io
from dotenv import load_dotenv

# Caricamento sicuro delle variabili d'ambiente
load_dotenv()

app = FastAPI(title="Aura Proxy Backend")

# Configurazione rigorosa del CORS per permettere le chiamate dal frontend Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inizializzazione LLM
LLM_API_KEY = os.getenv("LLM_API_KEY")
if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY mancante nel file .env")
llm_client = AsyncGroq(api_key=LLM_API_KEY)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "N2lVS1w4EtoT3dr4eOWO"  # Voce di Callum

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    user_text: str
    emotion_state: str
    slide_context: str

class TTSRequest(BaseModel):
    text: str

# --- ENDPOINTS ---
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        context_injection = ""
        if request.slide_context:
            context_injection = f"Ecco il testo delle slide:\n{request.slide_context}\n"

        # Costruzione del System Prompt
        system_prompt = f"""Sei Aura, un tutor didattico virtuale.
            L'utente sembra: {request.emotion_state}.
            {context_injection}
            REGOLE:
            1. NON presentarti mai.
            2. Vai dritto al sodo e basa le tue risposte sul contenuto delle slide.
            3. Sii concisa (max 3 frasi).
            4. Solo testo semplice. ASSOLUTAMENTE NESSUNA FORMATTAZIONE. Vietato usare Markdown, vietati gli asterischi, vietati i simboli matematici speciali. Scrivi tutto a parole in italiano."""

        #Groq ha una struttura a ruoli
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.user_text}
        ]
        
        #Chiamata asincrona a LLaMA 3 
        response = await llm_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.7,
            max_tokens=150 # Manteniamo la risposta concisa
        )
        
        #Estrazione del testo generato
        answer = response.choices[0].message.content.strip()
        return {"text": answer}
        
    except Exception as e:
        # AGGIUNTA PER IL DEBUG: stampa l'errore vero nel terminale
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Errore LLM: {str(e)}")

@app.post("/api/tts")
async def tts_endpoint(request: TTSRequest):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API Key mancante nel server.")
        
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?optimize_streaming_latency=3"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    payload = {
        "text": request.text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload, timeout=30.0)
        
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Errore dal provider TTS")
        
    return StreamingResponse(io.BytesIO(response.content), media_type="audio/mpeg")

app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)