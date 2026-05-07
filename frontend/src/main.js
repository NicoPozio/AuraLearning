import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import { GazeEstimator } from "./sensors/GazeEstimator.js";
import { FaceMetricsExtractor } from "./sensors/FaceMetricsExtractor.js";
import { AffectAnalyzer } from "./analysis/AffectAnalyzer.js";
import { GazeCalibrator } from "./calibration/GazeCalibrator.js";
import { CalibrationUI } from "./calibration/CalibrationUI.js";
import { OneEuroFilter } from "./utils/OneEuroFilter.js";
import { VoiceInput } from "./sensors/VoiceInput.js";
import { ECAController } from "./eca/ECAController.js";

let video, canvas, ctx, gazeDot;
let btnCalGaze, btnCalEmotion, calOverlay;
let sessionData = [];
let currentSmoothPos = { x: 0, y: 0 };
let currentFinalTranscript = "";
let currentGazedText = "";
let lastGazedTextTimestamp = 0;
const GAZE_CONTEXT_TTL_MS = 4000;
let _lastExtractedSnippet = "";
let textExtractionInterval = null;

let faceLandmarker;
let lastVideoTime = -1;
let lastFrameTimeMs = performance.now();
let isGazeCalibrating = false;
let currentNormalizedIris = null;
let isPdfLoaded = false;

// Proactive intervention
let lastProactiveIntervention = 0;
const PROACTIVE_COOLDOWN_MS = 30000;
let isProactiveInterventionActive = false;
let firstInterventionDone = false;
let isAnsweringUser = false;

// Negative state persistence timer
let negativeStateStartTime = 0;
const NEGATIVE_STATE_PERSIST_MS = 5000;

// Multi-frame gaze calibration
const GAZE_SAMPLE_COUNT = 30;
// IOD di riferimento per il depth correction del gaze.
// Fissato al momento della calibrazione gaze e MAI più cambiato.
// Usare affectAnalyzer.baseline.iod causerebbe derive ogni volta
// che la calibrazione emotiva aggiorna la baseline.
let gazeBaseIod = 0.20;
let gazeSampleBuffer = [];
let isCollectingGazeSample = false;

// ── STT state ──────────────────────────────────────────────────────────────
// FIX: wake word usa regex con word boundary — evita "paura", "laura", ecc.
const WAKE_WORD_RE = /\baura\b/i;

// Motivo del blocco mic — mostrato nella UI al posto del generico "Spento"
// Valori: null | 'echo' | 'pdf' | 'answering' | 'intervention' | 'calibrating'
let micBlockReason = null;

let ignoreMicUntil = 0;
let userMicTimeout = null;
let chatHistory = [];
let currentEmotionState = "Normale"; // "Normale" | "In difficoltà"

const affectAnalyzer = new AffectAnalyzer();
const eca = new ECAController('eca-container');
const gazeCalibrator = new GazeCalibrator();
const uiFilter = new OneEuroFilter(60, 0.1, 0.001, 1.0);
const gazeEstimator = new GazeEstimator();

let calibrationUI;
let voiceInput;

// ── Helpers ────────────────────────────────────────────────────────────────

function addChatMessage(sender, text) {
    const chatBox = document.getElementById('chat-box');
    if (!chatBox) return;
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('chat-message', sender === 'user' ? 'msg-user' : 'msg-ai');
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    chatHistory.push({ role: sender === 'user' ? 'user' : 'model', parts: [{ text }] });
}

// FIX: flush del buffer STT dopo ECA speech per eliminare l'eco accumulato
async function speakECA(text) {
    ignoreMicUntil = Infinity;
    micBlockReason = 'echo';
    updateMicStatusUI();
    try {
        await eca.speak(text);
    } catch (e) {
        console.error("TTS Error:", e);
    } finally {
        // Svuota il buffer audio del motore STT prima di riaprire il gate
        if (voiceInput) voiceInput.flush();
        ignoreMicUntil = performance.now() + 1500;
        // micBlockReason viene resettato nel prossimo aggiornamento UI
        // (dopo che ignoreMicUntil scade, il prossimo frame di stato lo corregge)
    }
}

// Aggiorna la label di stato microfono in base al motivo del blocco
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

function _getFreshGazeContext() {
    if (!isPdfLoaded) return "";
    return (performance.now() - lastGazedTextTimestamp) > GAZE_CONTEXT_TTL_MS
        ? "" : currentGazedText.trim();
}

function _snippetChangedSignificantly(a, b) {
    if (!b) return true;
    const wA = new Set(a.toLowerCase().split(/\s+/));
    const wB = new Set(b.toLowerCase().split(/\s+/));
    let overlap = 0;
    for (const w of wA) { if (wB.has(w)) overlap++; }
    return (overlap / Math.max(wA.size, wB.size)) < 0.7;
}

function median(arr) {
    const s = [...arr].sort((a, b) => a - b);
    const m = Math.floor(s.length / 2);
    return s.length % 2 === 0 ? (s[m - 1] + s[m]) / 2 : s[m];
}

// ── NEW: recupera tutto il testo della slide attualmente visibile nel viewport ──
// Usato come fallback quando il gaze non punta a nessun textLayer.
// Restituisce il testo della slide il cui wrapper è più al centro del viewport,
// oppure stringa vuota se non trovato.
function getVisibleSlideText() {
    const wrappers = Array.from(document.querySelectorAll('.pdf-slide-wrapper'));
    if (wrappers.length === 0) return "";

    const viewportMid = window.scrollY + window.innerHeight / 2;

    // Trova il wrapper più vicino al centro del viewport
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

    // Estrae tutto il testo dal textLayer di quella slide
    const textLayer = bestWrapper.querySelector('.textLayer');
    if (!textLayer) return ""; // slide senza testo (solo immagine) → stringa vuota

    const text = Array.from(textLayer.children)
        .map(el => el.textContent.trim())
        .filter(t => t.length > 0)
        .join(' ')
        .trim();

    return text;
}

// ── NEW: controlla se la slide visibile è solo immagine (nessun textLayer con testo) ──
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
    if (!textLayer) return true;

    const text = Array.from(textLayer.children)
        .map(el => el.textContent.trim())
        .filter(t => t.length > 0)
        .join('');

    return text.length === 0;
}

// ── Init ───────────────────────────────────────────────────────────────────

async function init() {
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

    calibrationUI = new CalibrationUI(() => {
        gazeCalibrator.calculateModel();
        isGazeCalibrating = false;
        gazeDot.style.display = 'block';
        // Salva l'IOD corrente come riferimento fisso per il depth correction.
        // rawMetrics potrebbe non essere disponibile qui (siamo in callback),
        // usiamo il valore della baseline emotiva se già calibrata, altrimenti
        // il valore accumulato nei campioni gaze (mediana degli IOD raccolti).
        if (affectAnalyzer.isCalibrated && affectAnalyzer.baseline.iod > 0) {
            gazeBaseIod = affectAnalyzer.baseline.iod;
        } else if (gazeCalibrator.calibrationPoints.length > 0) {
            // Fallback: median IOD dai punti di calibrazione gaze stessi
            // (non disponibile direttamente, teniamo il default 0.20)
            gazeBaseIod = 0.20;
        }
        console.log(`[Gaze] gazeBaseIod fissato a ${gazeBaseIod.toFixed(4)}`);
    });

    voiceInput = new VoiceInput(
        // onTranscript
        (interim, final) => {
            // ── Gate: ordine di priorità ────────────────────────────────
            // 1. Anti-echo sempre prioritario
            if (performance.now() < ignoreMicUntil) return;

            // 2. Calibrazione emotiva: ignora tutto
            if (affectAnalyzer.isCalibrating) return;

            // 3. PDF non caricato
            if (!isPdfLoaded) return;

            // 4. ECA sta rispondendo a una domanda dell'utente: ignora
            if (isAnsweringUser) return;

            // 5. Intervento proattivo: SOLO wake word può interrompere
            //    (l'utente può dire "Aura, ho capito" per fermare l'intervento)
            if (isProactiveInterventionActive) {
                if (final) {
                    const lower = final.trim().toLowerCase();
                    if (WAKE_WORD_RE.test(lower) && final.trim().length > 5) {
                        // Interrompe l'intervento e gestisce la domanda
                        eca.currentAudio.pause();
                        isProactiveInterventionActive = false;
                        negativeStateStartTime = 0;
                        manageUserQuestion(final.trim());
                    }
                }
                return;
            }

            // ── Tutti i gate superati ────────────────────────────────────
            updateMicStatusUI();
            const voiceDiv = document.getElementById('val-voice');

            if ((interim || final) && eca.currentState === 'IDLE') eca.setState('LISTENING');
            if (userMicTimeout) clearTimeout(userMicTimeout);

            if (final) {
                const text = final.trim();
                const lower = text.toLowerCase();
                currentFinalTranscript = text;
                voiceDiv.innerHTML = `<span style="color:#0f172a;font-weight:600;">${text}</span>`;

                // FIX wake word: word boundary regex, non substring
                if (text.length > 5 && WAKE_WORD_RE.test(lower)) {
                    manageUserQuestion(text);
                } else {
                    eca.setState('IDLE');
                    // Feedback: l'utente ha parlato ma senza wake word
                    if (text.length > 3) {
                        voiceDiv.innerHTML += `<span style="font-size:0.75rem;color:#94a3b8;display:block;">
                            (Di' "Aura" per attivare l'assistente)</span>`;
                    }
                }

                setTimeout(() => { if (currentFinalTranscript === text) currentFinalTranscript = ""; }, 3000);

            } else if (interim) {
                voiceDiv.innerHTML = `<span style="font-style:italic;color:#475569;">${interim}...</span>`;
                userMicTimeout = setTimeout(() => {
                    if (eca.currentState === 'LISTENING') eca.setState('IDLE');
                }, 1500);
            }
        },
        // onStatusChange
        (statusMessage) => {
            // VoiceInput manda solo errori critici (mic negato ecc.)
            // Lo stato normale è gestito da updateMicStatusUI()
            if (statusMessage.startsWith("Errore")) {
                const el = document.getElementById('voice-status');
                if (el) { el.innerText = statusMessage; el.style.color = '#ef4444'; }
            }
        }
    );

    try { await eca.loadModel('./models/personaggio.fbx'); }
    catch (error) { console.error("Error loading 3D model:", error); }

    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    video.addEventListener('loadeddata', loop);

    if (gazeCalibrator.loadFromStorage?.()) gazeDot.style.display = 'block';

    // Tenta di ricaricare la baseline emotiva da sessione precedente
    if (affectAnalyzer.loadBaselineFromStorage()) {
        document.getElementById('val-status').innerText = "Baseline caricata.";
        updateMicStatusUI();
    }

    // Aggiorna la label mic ogni secondo (per far scadere lo stato anti-echo)
    setInterval(updateMicStatusUI, 1000);
}

// ── LLM ───────────────────────────────────────────────────────────────────

async function fetchLLMResponse(userText) {
    // Ultime 10 chat (escluso il messaggio corrente che viene aggiunto dopo)
    const recentHistory = chatHistory.slice(-10);
    const payload = {
        user_text: userText,
        emotion_state: currentEmotionState,
        slide_context: _getFreshGazeContext(),
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

async function manageUserQuestion(questionText) {
    if (isAnsweringUser) return;
    isAnsweringUser = true;
    addChatMessage('user', questionText);
    await speakECA("Certo, dammi un secondo.");
    eca.setState('THINKING');
    const reply = await fetchLLMResponse(questionText);
    addChatMessage('ai', reply);
    await speakECA(reply);
    lastProactiveIntervention = performance.now();
    isAnsweringUser = false;
    eca.setState('IDLE');
    updateMicStatusUI();
}

// ── Render loop ────────────────────────────────────────────────────────────

function drawFaceMeshSegments(landmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
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
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.brows, 'rgba(43,87,151,0.8)');
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.leftEye, 'rgba(0,180,0,0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.rightEye, 'rgba(0,180,0,0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.outerLips, 'rgba(185,29,71,0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.innerLips, 'rgba(185,29,71,0.8)', true);
}

function updateStatusCard(state) {
    const card = document.getElementById('card-status');
    const val = document.getElementById('val-status');
    card.classList.remove('alert', 'warning', 'info');
    if (state.isInDifficulty) {
        card.classList.add('alert');
        const exprLabel = state.activeExpressions.length > 0
            ? ` (${state.activeExpressions.slice(0, 2).join(', ')})` : '';
        val.innerText = "In difficoltà" + exprLabel;
    } else {
        val.innerText = "Normale";
    }
}

async function loop() {
    try {
        const ts = performance.now();

        if (video.currentTime !== lastVideoTime) {
            const dtSec = Math.min((ts - lastFrameTimeMs) / 1000.0, 0.1);
            lastFrameTimeMs = ts;
            lastVideoTime = video.currentTime;

            eca.update(dtSec);
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            const results = faceLandmarker.detectForVideo(video, ts);

            const hasFace = results.faceLandmarks?.length > 0;
            affectAnalyzer.updateFaceAbsent(!hasFace, Math.min((performance.now() - lastFrameTimeMs + 16) / 1000, 0.1));

            if (hasFace) { // era: if (results.faceLandmarks?.length > 0)
                const landmarks = results.faceLandmarks[0];
                drawFaceMeshSegments(landmarks);

                const rawMetrics = FaceMetricsExtractor.extractRawMetrics(landmarks);
                if (!rawMetrics) { return; } // finally chiama requestAnimationFrame

                currentNormalizedIris = gazeEstimator.getRobustGazeVector(landmarks);

                // Multi-frame calibration
                if (isCollectingGazeSample && currentNormalizedIris) {
                    // FIX: salva valori depth-corrected, coerenti con predict()
                    const _dcCal = gazeBaseIod / rawMetrics.iod;
                    gazeSampleBuffer.push({ x: currentNormalizedIris.x * _dcCal, y: currentNormalizedIris.y * _dcCal });
                    calibrationUI.updateProgress(Math.min(gazeSampleBuffer.length / GAZE_SAMPLE_COUNT, 1));
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

                // Gaze prediction
                if (!isGazeCalibrating && gazeCalibrator.regressionModel) {
                    const rawPos = gazeCalibrator.predict(
                        currentNormalizedIris.x * (gazeBaseIod / rawMetrics.iod),
                        currentNormalizedIris.y * (gazeBaseIod / rawMetrics.iod)
                    );
                    if (rawPos) {
                        const smooth = uiFilter.filter(rawPos.x, rawPos.y, ts);
                        currentSmoothPos = smooth;
                        gazeDot.style.left = `${smooth.x}px`;
                        gazeDot.style.top = `${smooth.y}px`;
                        if (rawPos.confidence !== undefined)
                            gazeDot.style.opacity = (0.4 + 0.6 * rawPos.confidence).toFixed(2);
                        document.getElementById('val-gaze').innerText =
                            `X: ${Math.round(smooth.x)}, Y: ${Math.round(smooth.y)}`;
                    }
                }

                // Emotion calibration
                if (affectAnalyzer.isCalibrating) {
                    const done = affectAnalyzer.processCalibrationSample(rawMetrics);
                    if (done) {
                        calOverlay.style.display = 'none';
                        document.getElementById('val-status').innerText = "Calibrazione completata.";
                        updateMicStatusUI();
                        setTimeout(async () => {
                            isProactiveInterventionActive = true;
                            try {
                                const msg = "Calibrazione completata. Ciao, io sono Aura. Sono un sistema di assistenza proattivo. Rilevo la tua attenzione e se vedo che sei in difficoltà interverrò per darti una mano. Per qualsiasi domanda, chiedi pure.";
                                addChatMessage('ai', msg);
                                await speakECA(msg);
                            } finally {
                                isProactiveInterventionActive = false;
                                updateMicStatusUI();
                            }
                        }, 1000);
                    }

                } else if (affectAnalyzer.isCalibrated) {
                    const state = affectAnalyzer.update(rawMetrics, dtSec);

                    currentEmotionState = state.isInDifficulty ? "In difficoltà" : "Normale";

                    document.getElementById('val-au4').innerText = `${state.zCorrugator.toFixed(2)} σ`;
                    document.getElementById('val-ear').innerText = `${state.zEar.toFixed(2)} σ`;
                    // Gaze-away: notifica all'analyzer quando il contesto non è fresco
                    // e il pdf è caricato (proxy: utente sta guardando altrove)
                    if (isPdfLoaded && _getFreshGazeContext().length === 0 && lastGazedTextTimestamp > 0) {
                        affectAnalyzer.notifyGazeAway();
                    }

                    if (!isProactiveInterventionActive && !isAnsweringUser) updateStatusCard(state);

                    // Aggiorna debug sidebar
                    if (state.debugSignals) {
                        const el = document.getElementById('val-debug');
                        if (el) {
                            const me = state.debugSignals;
                            const actives = state.activeExpressions;
                            const z = state.rawZ;
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

                    // Negative state persistence timer
                    const isNegativeNow = state.isInDifficulty;
                    if (!isNegativeNow) {
                        negativeStateStartTime = 0;
                    } else if (negativeStateStartTime === 0 && !isProactiveInterventionActive && !isAnsweringUser) {
                        negativeStateStartTime = ts;
                    }

                    const negativeMs = negativeStateStartTime > 0 ? ts - negativeStateStartTime : 0;
                    const isPersistent = negativeMs >= NEGATIVE_STATE_PERSIST_MS;
                    const canIntervene = !firstInterventionDone || (ts - lastProactiveIntervention) > PROACTIVE_COOLDOWN_MS;

                    if (isPdfLoaded && isPersistent && canIntervene && !isProactiveInterventionActive && !isAnsweringUser) {
                        console.log(`[AURA] Intervento dopo ${(negativeMs / 1000).toFixed(1)}s: ${currentEmotionState}`);
                        isProactiveInterventionActive = true;
                        firstInterventionDone = true;
                        lastProactiveIntervention = ts;
                        negativeStateStartTime = 0;
                        eca.setState('THINKING');

                        // ── NEW: logica a 3 casi per il contesto dell'intervento proattivo ──
                        const gazeCtx = _getFreshGazeContext();
                        const exprList = state.activeExpressions.join(', ') || 'espressione di difficoltà';
                        let prompt;

                        if (gazeCtx.length > 5) {
                            // CASO A: il gaze punta a del testo → usa lo snippet estratto
                            console.log(`[AURA] Caso A — snippet gaze: "${gazeCtx.substring(0, 60)}..."`);
                            prompt = `L'utente mostra ${exprList} mentre studia. Sta leggendo questo testo tratto dalla slide:\n"${gazeCtx}"\nFormula una brevissima domanda o proposta empatica (massimo 15 parole) basata su quel contenuto specifico.`;

                        } else if (isVisibleSlideImageOnly()) {
                            // CASO C: slide senza testo (solo immagine) → messaggio standard
                            console.log(`[AURA] Caso C — slide solo immagine, messaggio standard`);
                            prompt = `L'utente mostra ${exprList} guardando una slide che contiene solo immagini, senza testo. Chiedigli brevemente (massimo 10 parole) se ha bisogno di aiuto, in modo empatico.`;

                        } else {
                            // CASO B: gaze fuori dal testo ma la slide ha testo → manda tutto il testo della slide
                            const slideText = getVisibleSlideText();
                            if (slideText.length > 5) {
                                console.log(`[AURA] Caso B — testo intera slide: "${slideText.substring(0, 60)}..."`);
                                prompt = `L'utente mostra ${exprList}. Non riesco a rilevare esattamente dove stia guardando, ma la slide che sta visualizzando contiene questo testo:\n"${slideText}"\nFormula una domanda diretta e breve (massimo 15 parole) per capire se ha difficoltà con quel contenuto.`;
                            } else {
                                // Fallback: slide senza testo rilevabile (raro, safety net)
                                console.log(`[AURA] Fallback — nessun testo rilevabile`);
                                prompt = `L'utente mostra ${exprList}. Non sto rilevando cosa stia guardando. Fagli una domanda diretta e breve per capire se ha bisogno di aiuto.`;
                            }
                        }
                        // ── END NEW ──

                        fetchLLMResponse(prompt).then(phrase => {
                            addChatMessage('ai', phrase);
                            speakECA(phrase).then(() => {
                                isProactiveInterventionActive = false;
                                negativeStateStartTime = 0;
                                updateMicStatusUI();
                            });
                        });
                    }

                    if (gazeCalibrator.regressionModel) {
                        sessionData.push({
                            timestamp: ts.toFixed(2), dtSec: dtSec.toFixed(4),
                            rawGazeX: currentNormalizedIris?.x.toFixed(5) ?? "0",
                            rawGazeY: currentNormalizedIris?.y.toFixed(5) ?? "0",
                            smoothGazeX: currentSmoothPos.x.toFixed(2),
                            smoothGazeY: currentSmoothPos.y.toFixed(2),
                            iod: rawMetrics.iod.toFixed(4),
                            zAU4: state.zCorrugator?.toFixed(2) ?? "0",
                            zEar: state.zEar?.toFixed(2) ?? "0",
                            zLip: state.zLip?.toFixed(2) ?? "0",          // ← era senza ?.
                            zAU1: state.zInnerBrowRaise?.toFixed(2) ?? "0", // ← era senza ?.
                            zAU9: state.zNoseWrinkle?.toFixed(2) ?? "0",
                            blinkRate: state.blinkRate?.toFixed(3) ?? "0",
                            isInDifficulty: state.isInDifficulty ? 1 : 0,
                            activeExpressions: state.activeExpressions.join(';'),
                            gazeAwayCount: state.gazeAwayCount,
                            isSpeaking: state.isSpeaking ? 1 : 0,
                            gazeContextFresh: (_getFreshGazeContext().length > 0) ? 1 : 0,
                            emotionState: currentEmotionState,
                            spokenText: currentFinalTranscript
                        });
                    }
                }
            }
        }
    } catch (err) {
        // Un errore nel loop non deve mai fermare il rendering.
        // Logghiamo e riprendiamo nel prossimo frame.
        console.error('[loop] Errore non gestito:', err);
    } finally {
        requestAnimationFrame(loop);
    }
}

// ── DOMContentLoaded ───────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
    video = document.getElementById('webcam');
    canvas = document.getElementById('output-canvas');
    ctx = canvas.getContext('2d');
    gazeDot = document.getElementById('gaze-dot');
    btnCalGaze = document.getElementById('btn-cal-gaze');
    btnCalEmotion = document.getElementById('btn-cal-emotion');
    calOverlay = document.getElementById('cal-overlay');

    btnCalGaze.onclick = () => {
        btnCalGaze.blur();
        gazeCalibrator.reset(); uiFilter.reset();
        gazeBaseIod = 0.20; // reset: verrà risettato al completamento
        isGazeCalibrating = true;
        isCollectingGazeSample = false; gazeSampleBuffer = [];
        calibrationUI.start();
    };

    btnCalEmotion.onclick = () => {
        btnCalEmotion.blur();
        if (voiceInput && !voiceInput.isListening) voiceInput.start();
        eca.currentAudio.src = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";
        eca.currentAudio.play().catch(() => { });
        calOverlay.style.display = 'block';
        document.getElementById('val-status').innerText = "Acquisizione Baseline...";
        affectAnalyzer.startCalibration();
        updateMicStatusUI();
    };

    window.addEventListener('keydown', (e) => {
        if (e.code !== 'Space' || !isGazeCalibrating) return;
        e.preventDefault();
        if (e.repeat) return;

        // Fase intro: SPAZIO avvia la calibrazione vera
        if (calibrationUI._phase === 'intro') {
            calibrationUI.onSpaceDown();
            return;
        }

        // Fase dots: avvia raccolta multi-frame
        if (!isCollectingGazeSample && currentNormalizedIris) {
            isCollectingGazeSample = true; gazeSampleBuffer = [];
            calibrationUI.updateProgress(0);
        }
    });

    window.addEventListener('keyup', (e) => {
        if (e.code !== 'Space' || !isGazeCalibrating) return;
        if (isCollectingGazeSample && gazeSampleBuffer.length < GAZE_SAMPLE_COUNT) {
            isCollectingGazeSample = false; gazeSampleBuffer = [];
            calibrationUI.updateProgress(0);
        }
    });

    window.addEventListener('resize', () => {
        currentGazedText = ""; lastGazedTextTimestamp = 0; _lastExtractedSnippet = "";
    });

    document.getElementById('btn-export')?.addEventListener('click', () => {
        if (sessionData.length === 0) { alert("Nessun dato registrato."); return; }
        const headers = Object.keys(sessionData[0]).join(",");
        const rows = sessionData.map(r => {
            if (r.spokenText) r.spokenText = r.spokenText.replace(/,/g, ';');
            return Object.values(r).join(",");
        }).join("\n");
        const link = document.createElement("a");
        link.href = "data:text/csv;charset=utf-8," + encodeURI(headers + "\n" + rows);
        link.download = "aura_test_data.csv";
        document.body.appendChild(link); link.click(); document.body.removeChild(link);
    });

    document.getElementById('btn-show-pdf')?.addEventListener('click', () => {
        document.getElementById('video-wrapper').style.display = 'none';
        document.getElementById('pdf-wrapper').style.display = 'block';
        document.getElementById('btn-show-pdf').style.display = 'none';
        document.getElementById('btn-show-cam').style.display = 'inline-block';
        document.querySelectorAll('.metric-card').forEach(el => el.style.display = 'none');
        const cc = document.getElementById('chat-container');
        if (cc) cc.style.display = 'flex';
        updateMicStatusUI();
    });

    document.getElementById('btn-show-cam')?.addEventListener('click', () => {
        document.getElementById('pdf-wrapper').style.display = 'none';
        document.getElementById('video-wrapper').style.display = 'block';
        document.getElementById('btn-show-cam').style.display = 'none';
        document.getElementById('btn-show-pdf').style.display = 'inline-block';
        document.querySelectorAll('.metric-card').forEach(el => el.style.display = 'block');
        const cc = document.getElementById('chat-container');
        if (cc) cc.style.display = 'none';
    });

    const dropZone = document.getElementById('pdf-drop-zone');
    const inputPdf = document.getElementById('input-pdf');
    const slidesContainer = document.getElementById('slides-container');
    const pdfSpinner = document.getElementById('pdf-loading-spinner');

    if (dropZone && inputPdf && slidesContainer) {
        dropZone.onclick = () => inputPdf.click();
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('dragover'); };
        dropZone.ondrop = (e) => { e.preventDefault(); handlePdfUpload(e.dataTransfer.files[0]); };
        inputPdf.onchange = (e) => handlePdfUpload(e.target.files[0]);
    }

    async function handlePdfUpload(file) {
        if (!file || file.type !== "application/pdf") return alert("Carica un PDF.");

        const pdfjsLib = window['pdfjs-dist/build/pdf'];
        pdfjsLib.GlobalWorkerOptions.workerSrc =
            'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

        if (textExtractionInterval) { clearInterval(textExtractionInterval); textExtractionInterval = null; }
        isPdfLoaded = false;
        currentGazedText = ""; lastGazedTextTimestamp = 0; _lastExtractedSnippet = "";

        if (dropZone) dropZone.style.display = 'none';
        if (slidesContainer) slidesContainer.style.display = 'none';
        if (pdfSpinner) pdfSpinner.style.display = 'block';

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
                let totalTextItems = 0;

                for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                    const page = await pdf.getPage(pageNum);
                    const viewport = page.getViewport({ scale: 1.5 });

                    const pageWrapper = document.createElement('div');
                    pageWrapper.style.cssText = `position:relative;margin:0 auto 30px auto;width:${viewport.width}px;height:${viewport.height}px;`;
                    pageWrapper.className = 'pdf-slide-wrapper';
                    pageWrapper.dataset.pageNum = pageNum;

                    const slideCanvas = document.createElement('canvas');
                    slideCanvas.className = 'pdf-slide-canvas';
                    slideCanvas.height = viewport.height; slideCanvas.width = viewport.width;
                    slideCanvas.style.display = 'block';
                    pageWrapper.appendChild(slideCanvas);
                    slidesContainer.appendChild(pageWrapper);

                    const [textContent] = await Promise.all([
                        page.getTextContent(),
                        page.render({ canvasContext: slideCanvas.getContext('2d'), viewport }).promise
                    ]);

                    totalTextItems += textContent.items.length;

                    if (textContent.items.length === 0) {
                        const badge = document.createElement('div');
                        badge.style.cssText = 'position:absolute;top:8px;left:8px;background:rgba(245,158,11,0.85);color:white;padding:3px 8px;border-radius:4px;font-size:0.75rem;font-family:system-ui,sans-serif;pointer-events:none;z-index:5;';
                        badge.innerText = `Pagina ${pageNum}: solo immagine`;
                        pageWrapper.appendChild(badge);
                        continue;
                    }

                    const textLayerDiv = document.createElement('div');
                    textLayerDiv.className = 'textLayer';
                    textLayerDiv.style.cssText = 'position:absolute;left:0;top:0;right:0;bottom:0;overflow:hidden;opacity:0;pointer-events:auto;';
                    pageWrapper.appendChild(textLayerDiv);

                    try {
                        const rt = pdfjsLib.renderTextLayer({ textContentSource: textContent, container: textLayerDiv, viewport, textDivs: [] });
                        if (rt?.promise) await rt.promise;
                    } catch (e) { console.warn(`[PDF] textLayer p${pageNum}:`, e.message); }
                }

                await auraPromise;
                if (pdfSpinner) pdfSpinner.style.display = 'none';

                let msg;
                if (totalTextItems === 0) {
                    msg = "Attenzione: questo PDF non contiene testo selezionabile. Il rilevamento dello sguardo non potrà estrarre contesto. Considera di usare un PDF con testo incorporato.";
                } else {
                    msg = "Caricamento completato. Ora posso capire esattamente quale frase stai leggendo.";
                }
                addChatMessage('ai', msg);
                await speakECA(msg);

                isPdfLoaded = true;
                lastProactiveIntervention = performance.now();
                textExtractionInterval = setInterval(extractGazedText, 500);
                updateMicStatusUI();

            } catch (error) {
                console.error("[PDF]", error);
                if (pdfSpinner) pdfSpinner.style.display = 'none';
                slidesContainer.style.display = 'none';
                if (dropZone) dropZone.style.display = 'block';
                alert(`Errore durante il parsing del PDF:\n${error.message}`);
            }
        };

        fileReader.readAsArrayBuffer(file);
    }

    function extractGazedText() {
        if (!isPdfLoaded || isGazeCalibrating) return;
        if (!gazeCalibrator.regressionModel) return;
        if (currentSmoothPos.x === 0 && currentSmoothPos.y === 0) return;

        gazeDot.style.pointerEvents = 'none';
        const element = document.elementFromPoint(currentSmoothPos.x, currentSmoothPos.y);
        gazeDot.style.pointerEvents = '';

        if (!element) return;
        if (!element.parentNode?.classList?.contains('textLayer')) return;

        const siblings = Array.from(element.parentNode.children);
        const index = siblings.indexOf(element);
        if (index === -1) return;

        const snippet = siblings
            .slice(Math.max(0, index - 4), Math.min(siblings.length, index + 8))
            .map(el => el.textContent.trim()).filter(t => t.length > 0).join(' ');

        if (snippet.trim().length <= 5) return;

        if (_snippetChangedSignificantly(snippet, _lastExtractedSnippet)) {
            currentGazedText = snippet;
            _lastExtractedSnippet = snippet;
            console.log(`[GAZE] Testo rilevato: "${snippet.substring(0, 80)}..."`); // ← aggiunta
        }
        lastGazedTextTimestamp = performance.now();
    }

    init();
});
