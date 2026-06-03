"""
Aura backend proxy.

Thin FastAPI server that sits between the browser frontend and the two
external services Aura uses:

  /api/chat → Groq LLM (chat completions)
  /api/tts  → ElevenLabs (text-to-speech)

Reasons for proxying instead of calling the providers directly from the
browser:

  - API keys are kept off the client.
  - One CORS origin to whitelist instead of two.
  - The chat endpoint can inject a system prompt and reshape the
    conversation history (Gemini-style → Groq-style) on the fly.

Everything is async (AsyncGroq + httpx.AsyncClient) so the event loop is
never blocked while waiting for the LLM stream to complete.
"""

import os
from typing import List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import AsyncGroq
from pydantic import BaseModel

# Load credentials from .env before instantiating the clients below.
load_dotenv()

GROQ_API_KEY = os.getenv("LLM_API_KEY")
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# Callum voice — neutral male, well-suited to the multilingual v2 model.
ELEVEN_VOICE_ID = "N2lVS1w4EtoT3dr4eOWO"

app = FastAPI()

# Permissive CORS for development. Tighten allow_origins to the
# production domain before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared async Groq client. Reusing one client across requests lets the
# underlying httpx connection pool be amortised across calls.
groq_client = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# Request schemas.

class HistoryItem(BaseModel):
    """One turn of the chat history as sent by the frontend (Gemini-style)."""
    role: str               # "user" or "model"
    parts: List[dict]       # each dict has a "text" field


class ChatRequest(BaseModel):
    """Payload accepted by /api/chat."""
    user_text: str
    emotion_state: str = "Normale"
    slide_context: Optional[str] = ""
    chat_history: List[HistoryItem] = []


class TTSRequest(BaseModel):
    """Payload accepted by /api/tts."""
    text: str


# Endpoints.

@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Forward a chat turn to Groq.

    Steps:
      1. Build a system prompt that injects the live emotional state and
         the slide context extracted via gaze tracking.
      2. Convert the Gemini-style chat history into the Groq message
         schema (role + content string).
      3. Call llama-3.1-8b-instant with a short, fast configuration
         (max_tokens=300, temperature=0.6).
      4. Return the assistant's reply as plain JSON.
    """
    if not groq_client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY non configurata.")

    system_prompt = f"""Sei Aura, un tutor virtuale empatico e proattivo. Stai aiutando uno studente che sta studiando un PDF.

STATO EMOTIVO ATTUALE: {req.emotion_state}
CONTESTO DELLA SLIDE (estratto dallo sguardo): "{req.slide_context}"

REGOLE FONDAMENTALI:
1. Rispondi SEMPRE in italiano, mantenendo il tono caldo e accademico.
2. Se la risposta richiede termini tecnici inglesi (es. "feature map", "softmax"), usali invariati.
3. Sii concisa: una/due frasi quando possibile.
4. Se il contesto della slide è utile, fai riferimento al concetto specifico che lo studente sta leggendo.
5. Non chiedere "cosa non capisci, vuoi che ti spieghi, ...": OFFRI spiegazioni mirate e precise."""

    # Convert chat history from Gemini-style (role/parts) to Groq-style
    # (role/content). The "model" role becomes "assistant".
    messages = [{"role": "system", "content": system_prompt}]
    for h in req.chat_history:
        role = "assistant" if h.role == "model" else "user"
        text = h.parts[0].get("text", "") if h.parts else ""
        if text:
            messages.append({"role": role, "content": text})
    messages.append({"role": "user", "content": req.user_text})

    try:
        completion = await groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.6,
            max_tokens=300,
        )
        return {"text": completion.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Errore Groq: {e}")


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest):
    """
    Synthesise `text` via ElevenLabs and stream the resulting MP3 back
    to the browser.

    optimize_streaming_latency=3 trades a small amount of audio quality
    for a noticeably shorter time-to-first-byte, which matters because
    the avatar's mouth animation is gated on the audio actually starting.
    """
    if not ELEVEN_API_KEY:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY non configurata.")

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
        f"/stream?optimize_streaming_latency=3"
    )
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": req.text,
        "model_id": "eleven_multilingual_v2",
        # stability vs similarity_boost: standard balanced values that
        # work well across italian utterances of varying length.
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    try:
        # New AsyncClient per request: avoids state leaking across
        # otherwise-unrelated TTS calls. The cost of the handshake is
        # negligible compared to the synthesis itself.
        client = httpx.AsyncClient(timeout=30.0)
        r = await client.post(url, headers=headers, json=payload)

        if r.status_code == 401:
            # Surface ElevenLabs' own error code (quota exhausted) so
            # the frontend can switch to the browser-native fallback.
            await client.aclose()
            raise HTTPException(status_code=401, detail="ElevenLabs: crediti esauriti.")
        if r.status_code != 200:
            await client.aclose()
            raise HTTPException(status_code=502, detail=f"ElevenLabs HTTP {r.status_code}")

        # Stream the bytes back without buffering everything in memory.
        async def gen():
            try:
                async for chunk in r.aiter_bytes():
                    yield chunk
            finally:
                await client.aclose()

        return StreamingResponse(gen(), media_type="audio/mpeg")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Errore TTS: {e}")