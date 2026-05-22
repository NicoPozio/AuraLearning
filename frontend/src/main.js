import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import { GazeEstimator } from "./sensors/GazeEstimator.js";
import { FaceMetricsExtractor } from "./sensors/FaceMetricsExtractor.js";
import { AffectAnalyzer } from "./analysis/AffectAnalyzer.js";
import { GazeCalibrator } from "./calibration/GazeCalibrator.js";
import { CalibrationUI } from "./calibration/CalibrationUI.js";
import { OneEuroFilter } from "./utils/OneEuroFilter.js";
import { VoiceInput } from "./sensors/VoiceInput.js";
import { ECAController } from "./eca/ECAController.js";

// ───────────────────────────────────────────────────────────────────────────
// DOM REFERENCES (resolved on DOMContentLoaded)
// ───────────────────────────────────────────────────────────────────────────
let video, canvas, ctx, gazeDot;
let btnCalGaze, btnCalEmotion, calOverlay;

// ───────────────────────────────────────────────────────────────────────────
// SESSION / RUNTIME STATE
// ───────────────────────────────────────────────────────────────────────────

// Per-frame telemetry rows, exported as CSV on demand for offline analysis
let sessionData = [];

// Latest filtered gaze position in screen pixels (output of OneEuroFilter)
let currentSmoothPos = { x: 0, y: 0 };

// Most recent finalized speech transcript (cleared after a few seconds)
let currentFinalTranscript = "";

// Text snippet currently under the user's gaze on the PDF (deictic context)
let currentGazedText = "";

// Memoria degli ultimi 5 snippet di testo letti dall'utente (Reading Trail)
let recentGazeHistory = [];

// Memoria a breve termine per gli interventi proattivi
let lockedEcaContext = "";
let lockedEcaContextTime = 0;

// Timestamp (ms) of the last successful gaze-to-text extraction
let lastGazedTextTimestamp = 0;

// Freshness window: gaze snippets older than this are considered stale
const GAZE_CONTEXT_TTL_MS = 3000;

// Last snippet sent to the LLM, used to detect significant content change
let _lastExtractedSnippet = "";

// Interval handle for the gaze-text extraction polling loop
let textExtractionInterval = null;

// MediaPipe FaceLandmarker instance (loaded asynchronously in init)
let faceLandmarker;

// Frame deduplication: only process a frame when video.currentTime advances
let lastVideoTime = -1;

// High-resolution timestamp of the previous processed frame (for dt computation)
let lastFrameTimeMs = performance.now();

// True while the user is going through the 9-point gaze calibration
let isGazeCalibrating = false;

// Latest head-pose-corrected iris vector in normalised camera space
let currentNormalizedIris = null;

// Becomes true after a PDF deck has been successfully loaded and indexed
let isPdfLoaded = false;

// ── Proactive intervention state ──────────────────────────────────────────

// Timestamp of the last proactive intervention (used for cooldown)
let lastProactiveIntervention = 0;

// Minimum gap between two consecutive proactive interventions
const PROACTIVE_COOLDOWN_MS = 30000;

// True while the ECA is delivering an unprompted intervention
let isProactiveInterventionActive = false;

// First-intervention flag: the very first one bypasses the cooldown
let firstInterventionDone = false;

// True while the ECA is processing/answering an explicit user question
let isAnsweringUser = false;

// ── Negative-state persistence ────────────────────────────────────────────

// Timestamp when the current "negative" affective state started; 0 if none
let negativeStateStartTime = 0;

// Negative state must persist this long before triggering an intervention
const NEGATIVE_STATE_PERSIST_MS = 5000;

// ── Multi-frame gaze calibration ──────────────────────────────────────────

// Number of frames averaged per calibration anchor to reduce sample noise
const GAZE_SAMPLE_COUNT = 30;

// Reference IOD captured at calibration time, used as the depth-correction
// anchor for all subsequent frames. Fixed once and never overwritten.
let gazeBaseIod = 0.20;

// Buffer of (x, y) samples for the current calibration anchor
let gazeSampleBuffer = [];

// True while the user is holding SPACE on a calibration anchor
let isCollectingGazeSample = false;

// ── Speech-to-text state ──────────────────────────────────────────────────

// Wake-word matcher with word boundaries: prevents false matches inside
// words like "paura", "Laura", "restaura", etc.
const WAKE_WORD_RE = /\baura\b/i;

// Microphone gating timestamp: any transcript arriving before this is dropped.
// Used as an anti-echo measure right after the ECA finishes speaking.
let ignoreMicUntil = 0;

// Debounce timer that returns the ECA from LISTENING back to IDLE after silence
let userMicTimeout = null;

// Rolling conversation history sent as context to the LLM
let chatHistory = [];

// Coarse affect label injected into the LLM prompt: "Normale" | "In difficoltà"
let currentEmotionState = "Normale";

// ───────────────────────────────────────────────────────────────────────────
// SINGLETON COMPONENTS
// ───────────────────────────────────────────────────────────────────────────
const affectAnalyzer = new AffectAnalyzer();
const eca = new ECAController('eca-container');
const gazeCalibrator = new GazeCalibrator();

// 1-Euro filter tuned for cursor-like signals (60 Hz, smooth at rest, low lag in motion)
const uiFilter = new OneEuroFilter(60, 0.1, 0.001, 1.0);
const gazeEstimator = new GazeEstimator();

let calibrationUI;
let voiceInput;

// ───────────────────────────────────────────────────────────────────────────
// HELPERS
// ───────────────────────────────────────────────────────────────────────────

/**
 * Append a message to the on-screen chat panel and to the chat history
 * forwarded to the LLM.
 *
 * @param {'user'|'ai'} sender - Author of the message.
 * @param {string} text - Message body.
 */
function addChatMessage(sender, text) {
    const chatBox = document.getElementById('chat-box');
    if (!chatBox) return;
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('chat-message', sender === 'user' ? 'msg-user' : 'msg-ai');
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    // History uses Gemini-style role/parts; the backend converts to Groq format
    chatHistory.push({ role: sender === 'user' ? 'user' : 'model', parts: [{ text }] });
}

/**
 * Speak a phrase through the ECA's TTS while gating the microphone to avoid
 * self-transcription. After playback ends the STT engine is flushed so that
 * any residual TTS audio in the recogniser buffer is discarded.
 *
 * @param {string} text - Phrase to synthesise.
 */
async function speakECA(text) {
    // Hold the mic closed for the entire duration of the synthesis
    ignoreMicUntil = Infinity;
    updateMicStatusUI();
    try {
        await eca.speak(text);
    } catch (e) {
        console.error("TTS Error:", e);
    } finally {
        // Flush the STT engine's audio buffer before re-opening the gate,
        // otherwise residual TTS audio leaks back as user input
        if (voiceInput) voiceInput.flush();
        ignoreMicUntil = performance.now() + 1500;
    }
}

/**
 * Refresh the microphone-status badge in the sidebar. The label reflects
 * the highest-priority reason the mic is currently muted or active.
 */
function updateMicStatusUI() {
    const el = document.getElementById('voice-status');
    if (!el) return;

    const now = performance.now();
    if (now < ignoreMicUntil) {
        el.innerText = "Anti-eco attivo...";
        el.style.color = '#f59e0b';
    } else if (!isPdfLoaded) {
        el.innerText = "Carica un PDF per attivare";
        el.style.color = '#94a3b8';
    } else if (affectAnalyzer.isCalibrating) {
        el.innerText = "Calibrazione in corso...";
        el.style.color = '#94a3b8';
    } else if (isAnsweringUser) {
        el.innerText = "Elaborazione risposta...";
        el.style.color = '#3b82f6';
    } else if (isProactiveInterventionActive) {
        el.innerText = "Intervento Aura...";
        el.style.color = '#8b5cf6';
    } else {
        el.innerText = "In ascolto — di' \"Aura, ...\"";
        el.style.color = '#10b981';
    }
}

/**
 * Return the currently gazed text only if it was captured recently enough.
 * Stale snippets are treated as empty to avoid sending obsolete context
 * to the LLM.
 *
 * @returns {string} Fresh gaze snippet, or "" if no PDF is loaded or the
 * snippet is older than GAZE_CONTEXT_TTL_MS.
 */
function _getFreshGazeContext() {
    if (!isPdfLoaded) return "";
    return (performance.now() - lastGazedTextTimestamp) > GAZE_CONTEXT_TTL_MS
        ? "" : currentGazedText.trim();
}

/**
 * Decide whether a newly extracted text snippet differs enough from the
 * previous one to justify a new context injection. Comparison is based on
 * the Jaccard overlap of unique tokens: less than 70% common words is
 * considered a significant change.
 *
 * @param {string} a - New snippet.
 * @param {string} b - Previous snippet.
 * @returns {boolean} True when the change is significant.
 */
function _snippetChangedSignificantly(a, b) {
    if (!b) return true;
    const wA = new Set(a.toLowerCase().split(/\s+/));
    const wB = new Set(b.toLowerCase().split(/\s+/));
    let overlap = 0;
    for (const w of wA) { if (wB.has(w)) overlap++; }
    return (overlap / Math.max(wA.size, wB.size)) < 0.7;
}

/**
 * Numerical median of an array. Used to aggregate gaze samples per anchor
 * because the median is robust to single-frame outliers (saccades, blinks).
 *
 * @param {number[]} arr - Non-empty array of numbers.
 * @returns {number} Median value.
 */
function median(arr) {
    const s = [...arr].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}

/**
 * Concatenate the entire text of the slide closest to the centre of the
 * viewport. Used as a fallback when the gaze cursor does not fall on any
 * textLayer span (e.g. it lies on whitespace between paragraphs).
 *
 * @returns {string} Concatenated visible-slide text, or "" if none.
 */
function getVisibleSlideText() {
    const wrappers = Array.from(document.querySelectorAll('.pdf-slide-wrapper'));
    if (wrappers.length === 0) return "";

    // Reference y-coordinate: the vertical middle of the current viewport
    const viewportMid = window.scrollY + window.innerHeight / 2;

    // Find the slide whose centre is closest to the viewport centre
    let bestWrapper = null;
    let bestDist = Infinity;
    for (const w of wrappers) {
        const rect = w.getBoundingClientRect();
        const absTop = rect.top + window.scrollY;
        const mid = absTop + rect.height / 2;
        const dist = Math.abs(mid - viewportMid);
        if (dist < bestDist) { bestDist = dist; bestWrapper = w; }
    }

    if (!bestWrapper) return "";

    const textLayer = bestWrapper.querySelector('.textLayer');
    if (!textLayer) return "";

    // Concatenate every non-empty span inside the textLayer in DOM order
    const text = Array.from(textLayer.children)
        .map(el => el.textContent.trim())
        .filter(t => t.length > 0)
        .join(' ')
        .trim();

    return text;
}

/**
 * Detect whether the slide currently in view contains no extractable text
 * (image-only slide). Drives the prompt-selection logic for proactive
 * interventions: such slides require a generic, content-agnostic question.
 *
 * @returns {boolean} True if the visible slide has no text layer or its
 * text layer is empty.
 */
function isVisibleSlideImageOnly() {
    const wrappers = Array.from(document.querySelectorAll('.pdf-slide-wrapper'));
    if (wrappers.length === 0) return false;

    const viewportMid = window.scrollY + window.innerHeight / 2;
    let bestWrapper = null;
    let bestDist = Infinity;
    for (const w of wrappers) {
        const rect = w.getBoundingClientRect();
        const absTop = rect.top + window.scrollY;
        const mid = absTop + rect.height / 2;
        const dist = Math.abs(mid - viewportMid);
        if (dist < bestDist) { bestDist = dist; bestWrapper = w; }
    }

    if (!bestWrapper) return false;

    const textLayer = bestWrapper.querySelector('.textLayer');
    if (!textLayer) return true; // No text layer at all → definitely image-only

    const text = Array.from(textLayer.children)
        .map(el => el.textContent.trim())
        .filter(t => t.length > 0)
        .join('');

    return text.length === 0;
}

// ───────────────────────────────────────────────────────────────────────────
// INITIALISATION
// ───────────────────────────────────────────────────────────────────────────

/**
 * Bootstrap the whole pipeline: load the MediaPipe model, wire the
 * calibration UI, attach the speech recogniser, load the ECA avatar,
 * open the webcam and start the render loop. Runs once after the DOM
 * is ready.
 */
async function init() {
    // Load the MediaPipe WASM runtime and the FaceLandmarker model
    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
            delegate: "CPU"
        },
        runningMode: "VIDEO",
        numFaces: 1
    });

    // Calibration UI: callback fires when all 9 anchors have been collected
    calibrationUI = new CalibrationUI(() => {
        gazeCalibrator.calculateModel();
        isGazeCalibrating = false;
        gazeDot.style.display = 'block';

        // Snapshot the current IOD as the depth-correction reference.
        // Preference order: emotion-calibration baseline IOD → fallback 0.20
        if (affectAnalyzer.isCalibrated && affectAnalyzer.baseline.iod > 0) {
            gazeBaseIod = affectAnalyzer.baseline.iod;
        } else if (gazeCalibrator.calibrationPoints.length > 0) {
            gazeBaseIod = 0.20;
        }
        console.log(`[Gaze] gazeBaseIod fissato a ${gazeBaseIod.toFixed(4)}`);
    });

    // Voice input: continuous Italian speech recognition with two callbacks
    voiceInput = new VoiceInput(
        // onTranscript: invoked on every interim/final transcript chunk
        (interim, final) => {
            // ── Mic gates, evaluated in priority order ──────────────────

            // 1. Anti-echo gate (highest priority) — drop everything while
            //    the ECA's own voice could still be in the buffer
            if (performance.now() < ignoreMicUntil) return;

            // 2. Emotion baseline acquisition — ignore all input
            if (affectAnalyzer.isCalibrating) return;

            // 3. No PDF loaded — the assistant is not actionable yet
            if (!isPdfLoaded) return;

            // 4. ECA is already answering a user question — let it finish
            if (isAnsweringUser) return;

            // 5. ECA is mid-intervention — ONLY the wake word can interrupt
            if (isProactiveInterventionActive) {
                if (final) {
                    const lower = final.trim().toLowerCase();
                    if (WAKE_WORD_RE.test(lower) && final.trim().length > 5) {
                        // Cancel the ongoing TTS and switch to user-question mode
                        eca.currentAudio.pause();
                        isProactiveInterventionActive = false;
                        negativeStateStartTime = 0;
                        manageUserQuestion(final.trim());
                    }
                }
                return;
            }

            // ── All gates passed: handle the transcript normally ─────────
            updateMicStatusUI();
            const voiceDiv = document.getElementById('val-voice');

            // Switch the avatar to LISTENING as soon as speech is detected
            if ((interim || final) && eca.currentState === 'IDLE') eca.setState('LISTENING');
            if (userMicTimeout) clearTimeout(userMicTimeout);

            if (final) {
                const text = final.trim();
                const lower = text.toLowerCase();
                currentFinalTranscript = text;
                voiceDiv.innerHTML = `<span style="color:#0f172a;font-weight:600;">${text}</span>`;

                // Wake-word check: must contain "Aura" as a whole word and be long enough
                if (text.length > 5 && WAKE_WORD_RE.test(lower)) {
                    manageUserQuestion(text);
                } else {
                    eca.setState('IDLE');
                    if (text.length > 3) {
                        // Hint the user about the wake-word requirement
                        voiceDiv.innerHTML += `<span style="font-size:0.75rem;color:#94a3b8;display:block;">
                            (Di' "Aura" per attivare l'assistente)</span>`;
                    }
                }

                // Clear the cached transcript after a few seconds
                setTimeout(() => { if (currentFinalTranscript === text) currentFinalTranscript = ""; }, 3000);

            } else if (interim) {
                // Live-typing feedback while the user is still speaking
                voiceDiv.innerHTML = `<span style="font-style:italic;color:#475569;">${interim}...</span>`;
                // Return to IDLE if no further speech arrives within 1.5 s
                userMicTimeout = setTimeout(() => {
                    if (eca.currentState === 'LISTENING') eca.setState('IDLE');
                }, 1500);
            }
        },
        // onStatusChange: surfaces recogniser errors in the sidebar
        (statusMessage) => {
            if (statusMessage.startsWith("Errore")) {
                const el = document.getElementById('voice-status');
                if (el) { el.innerText = statusMessage; el.style.color = '#ef4444'; }
            }
        }
    );

    // Load the 3D avatar; non-fatal if it fails (system runs without it)
    try { await eca.loadModel('./models/personaggio.fbx'); }
    catch (error) { console.error("Error loading 3D model:", error); }

    // Request webcam access at VGA resolution / 60 fps; audio disabled
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    // Start the render loop as soon as the first frame is decodable
    video.addEventListener('loadeddata', loop);

    // Restore previously saved calibrations from sessionStorage, if any
    if (gazeCalibrator.loadFromStorage?.()) gazeDot.style.display = 'block';

    if (affectAnalyzer.loadBaselineFromStorage()) {
        document.getElementById('val-status').innerText = "Baseline caricata.";
        updateMicStatusUI();
    }

    // Mic-status badge needs periodic refresh because some of its
    // conditions are time-based (anti-echo timer)
    setInterval(updateMicStatusUI, 1000);
}

// ───────────────────────────────────────────────────────────────────────────
// LLM INTEGRATION
// ───────────────────────────────────────────────────────────────────────────

/**
 * POST a chat turn to the FastAPI backend, including emotion state and
 * gaze-derived slide context for prompt injection.
 *
 * @param {string} userText - Either a user utterance or a system-generated
 * prompt for a proactive intervention.
 * @returns {Promise<string>} The LLM's reply text, or a fallback error string.
 */
async function fetchLLMResponse(userText) {
    const recentHistory = chatHistory.slice(-10);

    // Unisci gli ultimi 5 elementi letti separandoli con uno spazio (o un separatore)
    let contextToSend = recentGazeHistory.join(" [...] ");

    // Se l'utente risponde entro 15 secondi da un intervento dell'ECA,
    // usiamo il contesto bloccato in precedenza
    if (performance.now() - lockedEcaContextTime < 15000 && lockedEcaContext !== "") {
        contextToSend = lockedEcaContext;
    } else if (contextToSend.trim() === "") {
        // Fallback di sicurezza: se per qualche motivo la cronologia è vuota, manda tutta la slide
        contextToSend = getVisibleSlideText();
    }

    // --- LOG DEL CONTESTO INVIATO ALL'LLM ---
    console.log("=========================================");
    console.log("[LLM CONTEXT] Testo inviato in slide_context:");
    console.log(contextToSend);
    console.log("=========================================");

    const payload = {
        user_text: userText,
        emotion_state: currentEmotionState,
        slide_context: contextToSend,
        chat_history: recentHistory
    };
    try {
        const res = await fetch("http://localhost:8000/api/chat", {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return (await res.json()).text;
    } catch (e) {
        console.error("[LLM]", e);
        return "Errore di connessione con il sistema centrale.";
    }
}

/**
 * Handle an explicit user question end-to-end: append it to the chat,
 * play a short acknowledgement to mask the LLM latency, fetch the reply,
 * speak it through TTS and reset all UI/state flags.
 *
 * Re-entry guard: if a previous answer is still in flight the call is
 * silently dropped.
 *
 * @param {string} questionText - The transcribed user utterance.
 */
/**
 * Handle an explicit user question end-to-end: append it to the chat,
 * play a short acknowledgement to mask the LLM latency, fetch the reply,
 * speak it through TTS and reset all UI/state flags.
 */
async function manageUserQuestion(questionText) {
    if (isAnsweringUser) return;
    isAnsweringUser = true;

    // AZZERA SOLO LO STRESS QUANDO PARLI TU
    negativeStateStartTime = 0;

    addChatMessage('user', questionText);
    // Filler line: gives the user audible feedback while the LLM responds
    await speakECA("Certo, dammi un secondo.");
    eca.setState('THINKING');
    const reply = await fetchLLMResponse(questionText);
    addChatMessage('ai', reply);
    await speakECA(reply);
    // Resetting the cooldown timer prevents an immediate proactive follow-up
    lastProactiveIntervention = performance.now();
    isAnsweringUser = false;
    eca.setState('IDLE');
    updateMicStatusUI();
}

// ───────────────────────────────────────────────────────────────────────────
// RENDER LOOP
// ───────────────────────────────────────────────────────────────────────────

/**
 * Draw the facial landmark wireframe (brows, eyes, lips) on top of the
 * mirrored webcam canvas. Purely cosmetic feedback for the user.
 *
 * @param {Array<{x:number,y:number}>} landmarks - 468/478 normalised
 * landmarks as returned by MediaPipe FaceLandmarker.
 */
function drawFaceMeshSegments(landmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Local helper: render a polyline through the given landmark indices
    const drawPath = (indices, color, close = false) => {
        ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.5;
        indices.forEach((idx, i) => {
            const pt = landmarks[idx];
            if (i === 0) ctx.moveTo(pt.x * canvas.width, pt.y * canvas.height);
            else ctx.lineTo(pt.x * canvas.width, pt.y * canvas.height);
        });
        if (close) ctx.closePath();
        ctx.stroke();
    };

    // Colour scheme: brows blue, eyes green, lips red — matches the design tokens
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.brows, 'rgba(43,87,151,0.8)');
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.leftEye, 'rgba(0,180,0,0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.rightEye, 'rgba(0,180,0,0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.outerLips, 'rgba(185,29,71,0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.innerLips, 'rgba(185,29,71,0.8)', true);
}

/**
 * Update the top-level "student status" card based on the affect analyser
 * decision. Adds/removes the .alert CSS modifier and renders the list of
 * active expressions when in difficulty.
 *
 * @param {Object} state - Output of AffectAnalyzer.update().
 */
function updateStatusCard(state) {
    const card = document.getElementById('card-status');
    const val = document.getElementById('val-status');
    // Reset emphasis classes before re-applying the current one
    card.classList.remove('alert', 'warning', 'info');
    if (state.isInDifficulty) {
        card.classList.add('alert');
        // Show at most the first two active expressions for readability
        const exprLabel = state.activeExpressions.length > 0
            ? ` (${state.activeExpressions.slice(0, 2).join(', ')})` : '';
        val.innerText = "In difficoltà" + exprLabel;
    } else {
        val.innerText = "Normale";
    }
}

/**
 * Main per-frame loop. Runs at the browser's animation cadence (~60 fps)
 * and orchestrates: face detection, gaze estimation/calibration, affect
 * update, proactive-intervention decision, debug UI refresh and CSV
 * row capture. Always re-arms itself via requestAnimationFrame in the
 * `finally` block, even on unexpected errors.
 */
async function loop() {
    try {
        const ts = performance.now();

        // Only process a frame when the video has actually advanced — saves CPU
        if (video.currentTime !== lastVideoTime) {
            // Clamp dt to 100 ms so that hidden-tab pauses don't inject huge
            // integration steps into the affect bucket on resume
            const dtSec = Math.min((ts - lastFrameTimeMs) / 1000.0, 0.1);
            lastFrameTimeMs = ts;
            lastVideoTime = video.currentTime;

            // Advance the ECA animation clock (mixer.update needs dt)
            eca.update(dtSec);
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // Run face landmark detection on the current frame
            const results = faceLandmarker.detectForVideo(video, ts);

            // Notify the analyser if the face went off-camera (used for the
            // "Distracted" label and for the gaze-away counter)
            const hasFace = results.faceLandmarks?.length > 0;
            affectAnalyzer.updateFaceAbsent(!hasFace, Math.min((performance.now() - lastFrameTimeMs + 16) / 1000, 0.1));

            if (hasFace) {
                const landmarks = results.faceLandmarks[0];
                drawFaceMeshSegments(landmarks);

                // Extract IOD-normalised facial metrics; skip frame on failure
                const rawMetrics = FaceMetricsExtractor.extractRawMetrics(landmarks);
                if (!rawMetrics) { return; } // `finally` still calls requestAnimationFrame

                // Head-pose-corrected iris vector in normalised camera space
                currentNormalizedIris = gazeEstimator.getRobustGazeVector(landmarks);

                // ── Multi-frame gaze sample collection (during calibration) ──
                if (isCollectingGazeSample && currentNormalizedIris) {
                    // Apply depth correction even during calibration so the
                    // model is trained on the same coordinate space used at runtime
                    const _dcCal = gazeBaseIod / rawMetrics.iod;
                    gazeSampleBuffer.push({ x: currentNormalizedIris.x * _dcCal, y: currentNormalizedIris.y * _dcCal });
                    calibrationUI.updateProgress(Math.min(gazeSampleBuffer.length / GAZE_SAMPLE_COUNT, 1));

                    // Once enough samples are collected, store their median
                    // (robust to outliers) and advance to the next anchor
                    if (gazeSampleBuffer.length >= GAZE_SAMPLE_COUNT) {
                        const medX = median(gazeSampleBuffer.map(p => p.x));
                        const medY = median(gazeSampleBuffer.map(p => p.y));
                        const target = calibrationUI.getNextPointCoords();
                        gazeCalibrator.recordDataPoint(medX, medY, target.x, target.y);
                        calibrationUI.advance();
                        gazeSampleBuffer = [];
                        isCollectingGazeSample = false;
                    }
                }

                // ── Run-time gaze prediction ─────────────────────────────────
                if (!isGazeCalibrating && gazeCalibrator.regressionModel) {
                    // Depth-correct the iris vector before feeding the TPS,
                    // so that leaning closer/farther doesn't bias the cursor
                    const rawPos = gazeCalibrator.predict(
                        currentNormalizedIris.x * (gazeBaseIod / rawMetrics.iod),
                        currentNormalizedIris.y * (gazeBaseIod / rawMetrics.iod)
                    );
                    if (rawPos) {
                        // Adaptive smoothing of the screen-pixel coordinate
                        const smooth = uiFilter.filter(rawPos.x, rawPos.y, ts);
                        currentSmoothPos = smooth;
                        gazeDot.style.left = `${smooth.x}px`;
                        gazeDot.style.top = `${smooth.y}px`;
                        // Fade the dot near extrapolation regions to convey uncertainty
                        if (rawPos.confidence !== undefined)
                            gazeDot.style.opacity = (0.4 + 0.6 * rawPos.confidence).toFixed(2);
                        document.getElementById('val-gaze').innerText =
                            `X: ${Math.round(smooth.x)}, Y: ${Math.round(smooth.y)}`;
                    }
                }

                // ── Emotion baseline acquisition ─────────────────────────────
                if (affectAnalyzer.isCalibrating) {
                    const done = affectAnalyzer.processCalibrationSample(rawMetrics);
                    if (done) {
                        // Keep the overlay visible while we wait for the TTS network call,
                        // so the user doesn't see a confusing silent gap. The overlay is
                        // removed only when the ECA is actually about to speak.
                        document.getElementById('val-status').innerText = "Calibrazione completata. Preparo Aura...";
                        isProactiveInterventionActive = true;
                        updateMicStatusUI();
                        eca.setState('THINKING');
                        (async () => {
                            try {
                                const msg = "Calibrazione completata. Ciao, io sono Aura. Sono un sistema di assistenza proattivo. Rilevo la tua attenzione e se vedo che sei in difficoltà interverrò per darti una mano. Per qualsiasi domanda, chiedi pure.";
                                addChatMessage('ai', msg);
                                // Hide the overlay right before the audio actually starts playing
                                calOverlay.style.display = 'none';
                                await speakECA(msg);
                            } finally {
                                isProactiveInterventionActive = false;
                                updateMicStatusUI();
                            }
                        })();
                    }

                    // ── Calibrated regime: continuous affect update ──────────────
                } else if (affectAnalyzer.isCalibrated) {
                    const state = affectAnalyzer.update(rawMetrics, dtSec);

                    // Coarse label used by the LLM prompt
                    currentEmotionState = state.isInDifficulty ? "In difficoltà" : "Normale";

                    // Live Z-score readouts in the sidebar
                    document.getElementById('val-au4').innerText = `${state.zCorrugator.toFixed(2)} σ`;
                    document.getElementById('val-ear').innerText = `${state.zEar.toFixed(2)} σ`;



                    // Avoid flipping the status card during the ECA's own actions
                    if (!isProactiveInterventionActive && !isAnsweringUser) updateStatusCard(state);

                    // ── Debug sidebar refresh ────────────────────────────────
                    if (state.debugSignals) {
                        const el = document.getElementById('val-debug');
                        if (el) {
                            const me = state.debugSignals;
                            const actives = state.activeExpressions;
                            const z = state.rawZ;
                            // Per-signal formatter:
                            //   ● red bold  → confirmed active (above threshold AND debounced)
                            //   ◐ amber     → instantaneous signal but not yet confirmed
                            //   ○ grey      → below threshold
                            const fmt = (name, label, signal, zVal, me_active) => {
                                const dot = me_active ? '●' : (signal ? '◐' : '○');
                                const col = me_active ? '#ef4444' : (signal ? '#f59e0b' : '#94a3b8');
                                const bold = me_active ? 'font-weight:700;' : '';
                                return `<span style="color:${col};${bold}">${dot} ${label}: ${zVal.toFixed(2)}σ</span>`;
                            };
                            el.innerHTML = [
                                fmt('browFurrow', 'AU4 Fronte', me.browFurrow, z.zCorrugator, actives.includes('browFurrow')),
                                fmt('eyeSquint', 'EAR Occhi', me.eyeSquint, z.zEar, actives.includes('eyeSquint')),
                                fmt('mouthFrown', 'Bocca giù', me.mouthFrown, z.zMouthCurvature, actives.includes('mouthFrown')),
                                fmt('lipPress', 'Labbra prem.', me.lipPress, z.zLipPress, actives.includes('lipPress')),
                                fmt('noseWrinkle', 'Naso AU9', me.noseWrinkle, z.zNoseWrinkle, actives.includes('noseWrinkle')),
                                fmt('browRaise', 'AU1 Sopr.', me.browRaise, z.zBrowRaise, actives.includes('browRaise')),
                                fmt('mouthOpen', 'Bocca aperta', me.mouthOpen, z.zMouthOpen, actives.includes('mouthOpen')),
                            ].join('<br>');
                        }
                    }

                    // ── Negative-state persistence timer ─────────────────────
                    // We only fire a proactive intervention if the negative
                    // state has been sustained for NEGATIVE_STATE_PERSIST_MS.
                    const isNegativeNow = state.isInDifficulty;
                    if (!isNegativeNow) {
                        negativeStateStartTime = 0;
                    } else if (negativeStateStartTime === 0 && !isProactiveInterventionActive && !isAnsweringUser) {
                        negativeStateStartTime = ts;
                    }

                    const negativeMs = negativeStateStartTime > 0 ? ts - negativeStateStartTime : 0;
                    const isPersistent = negativeMs >= NEGATIVE_STATE_PERSIST_MS;
                    // The very first intervention bypasses the cooldown
                    const canIntervene = !firstInterventionDone || (ts - lastProactiveIntervention) > PROACTIVE_COOLDOWN_MS;

                    if (isPdfLoaded && isPersistent && canIntervene && !isProactiveInterventionActive && !isAnsweringUser) {
                        console.log(`[AURA] Intervento dopo ${(negativeMs / 1000).toFixed(1)}s: ${currentEmotionState}`);
                        isProactiveInterventionActive = true;
                        firstInterventionDone = true;
                        lastProactiveIntervention = ts;
                        negativeStateStartTime = 0;
                        eca.setState('THINKING');

                        // ── Three-branch prompt selection ────────────────────
                        // The intervention prompt is tailored to what we know
                        // about the slide the student is looking at.

                        // Proviamo a usare la cronologia di lettura
                        let gazeCtx = recentGazeHistory.join(" [...] ");
                        const exprList = state.activeExpressions.join(', ') || 'espressione di difficoltà';
                        let prompt;

                        if (gazeCtx.length > 5) {
                            // CASE A — gaze is on text: use the precise snippet
                            console.log(`[AURA] Caso A — snippet gaze: "${gazeCtx.substring(0, 60)}..."`);
                            prompt = `L'utente mostra ${exprList} mentre studia. Sta leggendo questo passaggio dalla slide:
                            "${gazeCtx}"

                            REGOLE STRETTE:
                            1. Scrivi UNA SOLA frase che termina con un PUNTO INTERROGATIVO.
                            2. Massimo 15 parole TOTALI.
                            3. Formato OBBLIGATORIO: "Vuoi che ti spieghi <concetto>?" oppure "Posso aiutarti con <concetto>?" oppure "Vuoi che riprenda <concetto>?".
                            4. Sostituisci <concetto> con il concetto specifico del passaggio.
                            5. NON aggiungere altre frasi dopo il punto interrogativo. NON spiegare nulla.
                            6. VIETATE: "Cosa non capisci", "Che significa", "Qual è il significato".`;

                        } else if (isVisibleSlideImageOnly()) {
                            // CASE C — image-only slide: generic empathic ping
                            console.log(`[AURA] Caso C — slide solo immagine, messaggio standard`);
                            prompt = `L'utente mostra ${exprList} guardando una slide che contiene solo immagini.

                            REGOLE STRETTE per la risposta:
                            1. Scrivi UNA SOLA frase empatica di massimo 12 parole.
                            2. Devi OFFRIRE aiuto, NON fare una domanda di verifica.
                            3. Esempi validi: "Vedo che stai riflettendo, vuoi che riprendiamo l'argomento?", "Vuoi che ti riepiloghi il concetto precedente?".
                            4. NON chiedere MAI "cosa non capisci" o "che significa". Sono VIETATE.`;

                        } else {
                            // CASE B — gaze off the text but the slide has text:
                            // fall back to the whole visible-slide text
                            const slideText = getVisibleSlideText();
                            if (slideText.length > 5) {
                                console.log(`[AURA] Caso B — testo intera slide: "${slideText.substring(0, 60)}..."`);
                                prompt = `L'utente mostra ${exprList}. Sta guardando una slide che contiene questo testo:
                                "${slideText}"

                                REGOLE STRETTE:
                                1. Scrivi UNA SOLA frase che termina con un PUNTO INTERROGATIVO.
                                2. Massimo 15 parole TOTALI.
                                3. Formato OBBLIGATORIO: "Vuoi che ti spieghi <concetto>?" oppure "Posso aiutarti con <concetto>?" oppure "Vuoi che riprenda <concetto>?".
                                4. Sostituisci <concetto> con il concetto principale della slide.
                                5. NON aggiungere altre frasi dopo il punto interrogativo. NON spiegare nulla.
                                6. VIETATE: "Cosa non capisci", "Che significa", "Qual è il significato".`;
                            } else {
                                // Final fallback: no usable textual context
                                console.log(`[AURA] Fallback — nessun testo rilevabile`);
                                prompt = `L'utente mostra ${exprList}. Non sto rilevando cosa stia guardando.

                                REGOLE STRETTE per la risposta:
                                1. Scrivi UNA SOLA frase empatica di massimo 12 parole.
                                2. Devi OFFRIRE aiuto, NON fare una domanda di verifica.
                                3. Esempi validi: "Vedo che fai fatica, vuoi che riprendiamo l'ultimo concetto?", "Vuoi che ti aiuti a riprendere il filo?".
                                4. NON fare MAI domande del tipo "cosa non capisci" o "che significa".`;
                            }
                        }
                        // ── Esecuzione della chiamata LLM per l'intervento ──
                        // ── Esecuzione della chiamata LLM per l'intervento ──
                        (async () => {
                            try {
                                // BLOCCO LA MEMORIA: Salvo il testo che l'ECA sta per usare
                                lockedEcaContext = gazeCtx.length > 5 ? gazeCtx : getVisibleSlideText();
                                lockedEcaContextTime = performance.now();

                                // 1. Manda il prompt nascosto all'LLM
                                const reply = await fetchLLMResponse(prompt);

                                // 2. Aggiungi la risposta alla chat visibile
                                addChatMessage('ai', reply);

                                // 3. Fai parlare l'avatar
                                await speakECA(reply);
                            } catch (err) {
                                console.error("[AURA] Errore durante l'intervento proattivo:", err);
                            } finally {
                                // 4. Rimetti l'avatar a riposo e resetta le variabili
                                isProactiveInterventionActive = false;
                                eca.setState('IDLE');
                                updateMicStatusUI();
                                lastProactiveIntervention = performance.now(); // Fa ripartire il cooldown


                            }
                        })();
                        // ── CSV telemetry row (only while the gaze model is trained) ──
                        if (gazeCalibrator.regressionModel) {
                            sessionData.push({
                                timestamp: ts.toFixed(2),
                                dtSec: dtSec.toFixed(4),
                                rawGazeX: currentNormalizedIris?.x.toFixed(5) ?? "0",
                                rawGazeY: currentNormalizedIris?.y.toFixed(5) ?? "0",
                                smoothGazeX: currentSmoothPos.x.toFixed(2),
                                smoothGazeY: currentSmoothPos.y.toFixed(2),
                                iod: rawMetrics.iod.toFixed(4),
                                zCorrugator: state.zCorrugator.toFixed(2),
                                zEar: state.zEar.toFixed(2),
                                zLipPress: state.zLipPress.toFixed(2),
                                zMouthOpen: state.zMouthOpen.toFixed(2),
                                zMouthCurvature: state.zMouthCurvature.toFixed(2),
                                zNoseWrinkle: state.zNoseWrinkle.toFixed(2),
                                zBrowRaise: state.zBrowRaise.toFixed(2),
                                blinkRate: state.blinkRate.toFixed(3),
                                isInDifficulty: state.isInDifficulty ? 1 : 0,
                                activeExpressions: state.activeExpressions.join(';'),
                                gazeAwayCount: state.gazeAwayCount,
                                gazeContextFresh: (_getFreshGazeContext().length > 0) ? 1 : 0,
                                emotionState: currentEmotionState,
                                spokenText: currentFinalTranscript
                            });
                        }
                    }
                }
            }
        }
    } catch (err) {
        // Catch-all: a failure inside one frame must never stop the loop
        console.error('[loop] Errore non gestito:', err);
    } finally {
        // ALWAYS schedule the next frame, even after an exception
        requestAnimationFrame(loop);
    }
}

// ───────────────────────────────────────────────────────────────────────────
// DOMContentLoaded: DOM bindings, calibration buttons, PDF upload pipeline
// ───────────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    // Cache DOM references for the rest of the file
    video = document.getElementById('webcam');
    canvas = document.getElementById('output-canvas');
    ctx = canvas.getContext('2d');
    gazeDot = document.getElementById('gaze-dot');
    btnCalGaze = document.getElementById('btn-cal-gaze');
    btnCalEmotion = document.getElementById('btn-cal-emotion');
    calOverlay = document.getElementById('cal-overlay');

    // ── Step 1: start gaze calibration ─────────────────────────────────────
    btnCalGaze.onclick = () => {
        btnCalGaze.blur(); // Avoid the SPACE key accidentally re-triggering this button
        gazeCalibrator.reset(); uiFilter.reset();
        gazeBaseIod = 0.20;
        isGazeCalibrating = true;
        isCollectingGazeSample = false; gazeSampleBuffer = [];
        calibrationUI.start();
    };

    // ── Step 2: start emotion baseline acquisition ─────────────────────────
    btnCalEmotion.onclick = () => {
        btnCalEmotion.blur();
        // Boot the speech recogniser the first time the user interacts
        // (browser autoplay policies require a user gesture)
        if (voiceInput && !voiceInput.isListening) voiceInput.start();
        // Play a sub-audible 1-frame WAV to unlock the AudioContext on iOS/Safari
        eca.currentAudio.src = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";
        eca.currentAudio.play().catch(() => { });
        calOverlay.style.display = 'block';
        document.getElementById('val-status').innerText = "Acquisizione Baseline...";
        affectAnalyzer.startCalibration();
        updateMicStatusUI();
    };

    // ── SPACE handling during gaze calibration ─────────────────────────────
    window.addEventListener('keydown', (e) => {
        if (e.code !== 'Space' || !isGazeCalibrating) return;
        e.preventDefault(); // Prevent page-scroll on SPACE
        if (e.repeat) return; // Ignore auto-repeat events

        // The calibration intro screen consumes the first SPACE press
        if (calibrationUI._phase === 'intro') {
            calibrationUI.onSpaceDown();
            return;
        }

        // From the second press onward, SPACE triggers sample collection
        if (!isCollectingGazeSample && currentNormalizedIris) {
            isCollectingGazeSample = true; gazeSampleBuffer = [];
            calibrationUI.updateProgress(0);
        }
    });

    window.addEventListener('keyup', (e) => {
        if (e.code !== 'Space' || !isGazeCalibrating) return;
        // If the user releases SPACE before the buffer is full, discard
        // partial samples — this prevents fragmentary anchors in the TPS
        if (isCollectingGazeSample && gazeSampleBuffer.length < GAZE_SAMPLE_COUNT) {
            isCollectingGazeSample = false; gazeSampleBuffer = [];
            calibrationUI.updateProgress(0);
        }
    });

    // On window resize the gaze coordinate system is invalidated;
    // clear cached snippets so we don't act on stale text positions
    window.addEventListener('resize', () => {
        currentGazedText = ""; lastGazedTextTimestamp = 0; _lastExtractedSnippet = "";
    });

    // ── Step 3: CSV export of the session telemetry ────────────────────────
    document.getElementById('btn-export')?.addEventListener('click', () => {
        if (sessionData.length === 0) { alert("Nessun dato registrato."); return; }
        const headers = Object.keys(sessionData[0]).join(",");
        const rows = sessionData.map(r => {
            // Strip commas from spoken text to avoid corrupting CSV cells
            if (r.spokenText) r.spokenText = r.spokenText.replace(/,/g, ';');
            return Object.values(r).join(",");
        }).join("\n");
        // Trigger a client-side download via a synthetic <a download> click
        const link = document.createElement("a");
        link.href = "data:text/csv;charset=utf-8," + encodeURI(headers + "\n" + rows);
        link.download = "aura_test_data.csv";
        document.body.appendChild(link); link.click(); document.body.removeChild(link);
    });

    // ── Step 4: switch the workspace from webcam view to PDF view ──────────
    document.getElementById('btn-show-pdf')?.addEventListener('click', () => {
        document.getElementById('video-wrapper').style.display = 'none';
        document.getElementById('pdf-wrapper').style.display = 'block';
        document.getElementById('btn-show-pdf').style.display = 'none';
        document.getElementById('btn-show-cam').style.display = 'inline-block';
        // Metric cards stay visible in PDF mode (live affective telemetry for testing).
        // The chat panel is also shown, placed below the metric cards in index.html.
        const cc = document.getElementById('chat-container');
        if (cc) cc.style.display = 'flex';
        updateMicStatusUI();
    });

    // Reverse of the above: from PDF view back to webcam view
    document.getElementById('btn-show-cam')?.addEventListener('click', () => {
        document.getElementById('pdf-wrapper').style.display = 'none';
        document.getElementById('video-wrapper').style.display = 'block';
        document.getElementById('btn-show-cam').style.display = 'none';
        document.getElementById('btn-show-pdf').style.display = 'inline-block';
        // Metric cards are already visible — only hide the chat panel
        const cc = document.getElementById('chat-container');
        if (cc) cc.style.display = 'none';
    });

    // ── PDF DROP ZONE WIRING ───────────────────────────────────────────────
    const dropZone = document.getElementById('pdf-drop-zone');
    const inputPdf = document.getElementById('input-pdf');
    const slidesContainer = document.getElementById('slides-container');
    const pdfSpinner = document.getElementById('pdf-loading-spinner');

    if (dropZone && inputPdf && slidesContainer) {
        // Click on the drop zone opens the file picker
        dropZone.onclick = () => inputPdf.click();
        // Drag-and-drop accepts a single PDF file
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
        dropZone.ondrop = (e) => { e.preventDefault(); handlePdfUpload(e.dataTransfer.files[0]); };
        inputPdf.onchange = (e) => handlePdfUpload(e.target.files[0]);
    }

    /**
     * Load a user-supplied PDF, rasterise every page onto a canvas and
     * mount an invisible-but-hit-testable PDF.js text layer on top so
     * that the gaze coordinate can later identify the exact span of
     * text under the user's eyes.
     *
     * @param {File} file - The dropped or selected PDF file.
     */
    async function handlePdfUpload(file) {
        if (!file || file.type !== "application/pdf") return alert("Carica un PDF.");

        // Configure PDF.js worker (CDN-hosted to avoid bundling)
        const pdfjsLib = window['pdfjs-dist/build/pdf'];
        pdfjsLib.GlobalWorkerOptions.workerSrc =
            'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

        // Tear down any previous PDF session before mounting the new one
        if (textExtractionInterval) { clearInterval(textExtractionInterval); textExtractionInterval = null; }
        isPdfLoaded = false;
        currentGazedText = ""; lastGazedTextTimestamp = 0; _lastExtractedSnippet = "";

        // UI swap: drop zone out, spinner in
        if (dropZone) dropZone.style.display = 'none';
        if (slidesContainer) slidesContainer.style.display = 'none';
        if (pdfSpinner) pdfSpinner.style.display = 'block';

        // Speak a status line in parallel with the rendering work so the user
        // perceives the page load as faster than it actually is
        const auraPromise = speakECA("Sto analizzando le slide, preparo il livello semantico.");

        const fileReader = new FileReader();
        fileReader.onerror = () => {
            if (pdfSpinner) pdfSpinner.style.display = 'none';
            alert("Errore nella lettura del file.");
        };

        fileReader.onload = async function () {
            try {
                let pdf;
                try {
                    pdf = await pdfjsLib.getDocument(new Uint8Array(this.result)).promise;
                } catch (e) { throw new Error(`PDF non valido: ${e.message}`); }

                slidesContainer.innerHTML = '';
                slidesContainer.style.display = 'block';
                // Aggregate counter for image-only PDF detection
                let totalTextItems = 0;

                // ── Page-by-page rendering loop ────────────────────────────
                for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                    const page = await pdf.getPage(pageNum);
                    // 1.5× scale: a good trade-off between sharpness and memory
                    const viewport = page.getViewport({ scale: 1.5 });

                    // Wrapper div: positioning anchor for the canvas + text layer
                    const pageWrapper = document.createElement('div');
                    pageWrapper.style.cssText = `position:relative;margin:0 auto 30px auto;width:${viewport.width}px;height:${viewport.height}px;`;
                    pageWrapper.className = 'pdf-slide-wrapper';
                    pageWrapper.dataset.pageNum = pageNum;

                    // Visible bitmap of the page
                    const slideCanvas = document.createElement('canvas');
                    slideCanvas.className = 'pdf-slide-canvas';
                    slideCanvas.height = viewport.height; slideCanvas.width = viewport.width;
                    slideCanvas.style.display = 'block';
                    pageWrapper.appendChild(slideCanvas);
                    slidesContainer.appendChild(pageWrapper);

                    // Fetch text content AND rasterise the page in parallel
                    const [textContent] = await Promise.all([
                        page.getTextContent(),
                        page.render({ canvasContext: slideCanvas.getContext('2d'), viewport }).promise
                    ]);

                    totalTextItems += textContent.items.length;

                    // Image-only page → display a badge and skip the text layer
                    if (textContent.items.length === 0) {
                        const badge = document.createElement('div');
                        badge.style.cssText = 'position:absolute;top:8px;left:8px;background:rgba(245,158,11,0.85);color:white;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-family:system-ui,sans-serif;pointer-events:none;z-index:5;';
                        badge.innerText = `Pagina ${pageNum}: solo immagine`;
                        pageWrapper.appendChild(badge);
                        continue;
                    }

                    // Build an invisible text layer that mirrors the PDF text geometry.
                    // pointer-events:auto is REQUIRED for elementFromPoint to hit it.
                    const textLayerDiv = document.createElement('div');
                    textLayerDiv.className = 'textLayer';
                    textLayerDiv.style.cssText = 'position:absolute;left:0;top:0;right:0;bottom:0;overflow:hidden;opacity:0;pointer-events:auto;';
                    pageWrapper.appendChild(textLayerDiv);

                    try {
                        const rt = pdfjsLib.renderTextLayer({ textContentSource: textContent, container: textLayerDiv, viewport, textDivs: [] });
                        if (rt?.promise) await rt.promise;
                    } catch (e) { console.warn(`[PDF] textLayer p${pageNum}:`, e.message); }
                }

                // Wait for the parallel TTS line to finish before queueing the next one
                await auraPromise;
                if (pdfSpinner) pdfSpinner.style.display = 'none';

                // Confirmation message: distinguishes text-PDFs from image-only ones
                let msg;
                if (totalTextItems === 0) {
                    msg = "Attenzione: questo PDF non contiene testo selezionabile. Il rilevamento dello sguardo non potrà estrarre contesto. Considera di usare un PDF con testo incorporato.";
                } else {
                    msg = "Caricamento completato. Ora posso capire esattamente quale frase stai leggendo.";
                }
                addChatMessage('ai', msg);
                await speakECA(msg);

                isPdfLoaded = true;
                // Reset the cooldown so we don't immediately fire a proactive call
                lastProactiveIntervention = performance.now();
                // Poll the gaze position twice per second to update the deictic context
                textExtractionInterval = setInterval(extractGazedText, 500);
                updateMicStatusUI();

            } catch (error) {
                console.error("[PDF]", error);
                if (pdfSpinner) pdfSpinner.style.display = 'none';
                // Restore the initial drop zone so the user can try a different file
                slidesContainer.style.display = 'none';
                if (dropZone) dropZone.style.display = 'block';
                alert(`Errore durante il parsing del PDF:\n${error.message}`);
            }
        };

        fileReader.readAsArrayBuffer(file);
    }

    /**
     * Sample the DOM element under the current smoothed gaze position and,
     * if it belongs to a PDF text layer, extract a context window of
     * neighbouring spans. The window is asymmetric — a few words before
     * the focus and more after — to bias towards what the user is about
     * to read.
     *
     * Called from a setInterval (500 ms) while a PDF is loaded.
     */
    function extractGazedText() {
        // Skip when gaze data is not actionable
        if (!isPdfLoaded || isGazeCalibrating) return;
        if (!gazeCalibrator.regressionModel) return;
        if (currentSmoothPos.x === 0 && currentSmoothPos.y === 0) return;

        // Temporarily disable pointer-events on the gaze dot so that
        // elementFromPoint returns the underlying text span rather than the dot itself
        gazeDot.style.pointerEvents = 'none';
        const element = document.elementFromPoint(currentSmoothPos.x, currentSmoothPos.y);
        gazeDot.style.pointerEvents = '';

        if (!element) return;
        // Only spans hosted inside a .textLayer represent slide text
        if (!element.parentNode?.classList?.contains('textLayer')) return;

        const siblings = Array.from(element.parentNode.children);
        const index = siblings.indexOf(element);
        if (index === -1) return;

        // Context window: 4 spans before, 8 after — asymmetric on purpose
        const snippet = siblings
            .slice(Math.max(0, index - 4), Math.min(siblings.length, index + 8))
            .map(el => el.textContent.trim()).filter(t => t.length > 0).join(' ');

        if (snippet.trim().length <= 5) return;

        // Only update the context when the new snippet differs enough,
        // to keep the LLM prompt stable while the user re-reads the same line
        // Only update the context when the new snippet differs enough
        if (_snippetChangedSignificantly(snippet, _lastExtractedSnippet)) {
            currentGazedText = snippet;
            _lastExtractedSnippet = snippet;

            // --- NUOVA LOGICA: Aggiorna la cronologia degli ultimi 5 snippet ---
            // Evitiamo di inserire lo stesso snippet due volte di fila
            if (recentGazeHistory.length === 0 || recentGazeHistory[recentGazeHistory.length - 1] !== snippet) {
                recentGazeHistory.push(snippet);
                if (recentGazeHistory.length > 5) {
                    recentGazeHistory.shift(); // Rimuove il più vecchio per tenerne solo 5
                }
            }
            // -------------------------------------------------------------------

            console.log(`[GAZE] Testo rilevato: "${snippet.substring(0, 80)}..."`);
        }
        // Always refresh the freshness timestamp, even when the snippet is unchanged
        lastGazedTextTimestamp = performance.now();
    }

    // Kick off the whole system
    init();
});