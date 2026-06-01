# Aura Learning Assistant

A Multimodal Affective Tutoring System — project for the course **Multimodal Interaction**, Sapienza University of Rome.

Developed by **Leonardo Costantini** and **Niccolò Pozio**.

Aura is a browser-based proactive tutoring system that detects signs of cognitive difficulty in students studying PDF slide decks, using only a standard webcam. It combines real-time facial microexpression analysis, gaze-based text extraction, and an LLM-driven Embodied Conversational Agent (ECA) to deliver spoken, contextually grounded interventions. The entire perception pipeline runs client-side, so no biometric data ever leaves the device.

## How it works

The system is organized into five modules communicating through shared state:

1. **Sensing** — MediaPipe extracts 478 facial landmarks per webcam frame (~30 FPS).
2. **Affect Analysis** — eight geometric detectors, each mapped to a FACS Action Unit, normalized against the user's personal baseline via z-score and governed by bidirectional hysteresis to suppress false positives.
3. **Gaze Pipeline** — an iris-displacement estimator, corrected for head pose and depth, mapped to screen pixels through a nine-point Thin-Plate Spline calibration and smoothed with the One-Euro filter.
4. **Context Extraction** — every 500 ms the gaze coordinate is matched against the PDF.js text layer to identify the exact span being read.
5. **Interaction** — when a difficulty state persists for five seconds, the affect state and reading context are sent to the LLM, and the reply is spoken by the ECA avatar.

## Activation modes

- **Explicit** — the user says the wake word *"Aura"* followed by a request; the prompt is didactic.
- **Proactive** — the agent speaks on its own when difficulty persists past threshold; the prompt is short and empathic. A 30-second cooldown prevents repeated interruptions.



## Technology stack

| Component | Technology |
| --- | --- |
| Face landmarks | MediaPipe FaceLandmarker (WASM) |
| PDF rendering | PDF.js 3.x |
| 3D avatar | Three.js |
| Speech recognition | Web Speech API |
| Backend | FastAPI (Python) |
| LLM inference | Groq (`llama-3.1-8b-instant`) |
| Text-to-speech | ElevenLabs (multilingual v2) |

## Setup

Backend:

```
cd backend
pip install fastapi uvicorn httpx
export GROQ_API_KEY=<your-groq-key>
export ELEVENLABS_API_KEY=<your-elevenlabs-key>
uvicorn main:app --reload --port 8000
```

Frontend (must be served over HTTP; use a Chromium-based browser for the Web Speech API):

```
cd frontend
python -m http.server 5500
```

Then open `http://localhost:5500`.

## Usage

1. **Gaze calibration** — fixate each of the nine points, holding the space bar.
2. **Emotion calibration** — hold a neutral expression for ~4 seconds.
3. **Load a PDF** — drop a slide deck into the study area.
4. **Study** — read normally; Aura intervenes on sustained difficulty, or say *"Aura"* to ask explicitly.



## License

Educational use, Multimodal Interaction, Sapienza University of Rome.
