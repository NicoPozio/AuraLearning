Then open `http://localhost:5500` in a Chromium-based browser (required for the Web Speech API).

## Usage

1. **Gaze calibration** — click *Calibrazione Sguardo* and fixate each of the nine points, holding the space bar while looking at each one.
2. **Emotion calibration** — click *Calibrazione Emotiva* and hold a neutral expression for about four seconds.
3. **Load a PDF** — drop a slide deck into the study area. Aura confirms when the semantic layer is ready.
4. **Study** — read normally. Aura intervenes on its own when it detects sustained difficulty, or you can ask explicitly by saying the wake word *"Aura"* followed by your question.

## Two activation modes

- **Explicit** — the user says the wake word *"Aura"* followed by a request. The prompt is didactic and the user controls the timing.
- **Proactive** — the agent speaks without being addressed, because a difficulty state has persisted past its threshold. The prompt instructs the model to produce a short, empathic, supportive sentence rather than a lecture. A 30-second cooldown prevents repeated interruptions.

## Validation

Validation was performed informally through systematic expression-testing sessions: after a neutral calibration, each microexpression was held in isolation while observing the live z-score debug sidebar to confirm correct detection. The full pipeline was also verified end-to-end on real slide decks, with the ECA delivering an intervention within roughly 7–8 seconds of expression onset — consistent with the sum of the activation window and the five-second persistence timer. A proper evaluation of *learning benefit* would require a user study with ground-truth difficulty annotations, which is left as future work.
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


