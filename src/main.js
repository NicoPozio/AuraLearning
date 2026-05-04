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

let pdfTextData = [];
let currentSlideContext = "";
let lastGazeContextTs = 0;

let faceLandmarker;
let lastVideoTime = -1;
let lastFrameTimeMs = performance.now();
let isGazeCalibrating = false;
let currentNormalizedIris = null;

// 🟢 NUOVO: Blocca il microfono e l'IA finché il PDF non è pronto
let isPdfLoaded = false;

// --- VARIABILI FASE PROATTIVA E CONVERSAZIONALE ---
let lastProactiveIntervention = 0;
const PROACTIVE_COOLDOWN_MS = 30000;
let isProactiveInterventionActive = false;
let primoInterventoFatto = false;

let isAnsweringUser = false;

// 🛡️ Filtri e Smoothing per Microfono ed Eco
let ignoreMicUntil = 0;
let userMicTimeout = null;
// --------------------------------------------------

let lastLogTs = 0;

const GEMINI_API_KEY = "AIzaSyDl-6nqknNCjx1dxX9FRFRgH-jGHKgatTg";
let chatHistory = [];
let currentEmotionState = "Neutrale";

const affectAnalyzer = new AffectAnalyzer();
const eca = new ECAController('eca-container');
const gazeCalibrator = new GazeCalibrator();
const uiFilter = new OneEuroFilter(60, 0.1, 0.001, 1.0);
const gazeEstimator = new GazeEstimator();

let calibrationUI;
let voiceInput;

const lumaCanvas = document.createElement('canvas');
lumaCanvas.width = 32;
lumaCanvas.height = 32;
const lumaCtx = lumaCanvas.getContext('2d', { willReadFrequently: true });

// ─── Aggiunge un messaggio alla chat ───
function addChatMessage(sender, text) {
    const chatBox = document.getElementById('chat-box');
    if (!chatBox) return;
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('chat-message');
    msgDiv.classList.add(sender === 'user' ? 'msg-user' : 'msg-ai');
    msgDiv.innerText = text;
    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
    chatHistory.push({ role: sender === 'user' ? 'user' : 'model', parts: [{ text: text }] });
}

// ─── Wrapper anti-eco: blocca il microfono mentre l'ECA parla ───
async function parlaECA(frase) {
    ignoreMicUntil = Infinity;
    try {
        await eca.speak(frase);
    } catch (e) {
        console.error("Errore TTS:", e);
    } finally {
        ignoreMicUntil = performance.now() + 1500;
    }
}
// ────────────────────────────────────

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
    });

    // ─── GESTIONE DEL MICROFONO UTENTE E ANIMAZIONE ASCOLTO ───
    voiceInput = new VoiceInput((interim, final) => {

        // 🟢 Se il PDF non è caricato, AURA IGNORA TUTTO QUELLO CHE DICI
        if (!isPdfLoaded || performance.now() < ignoreMicUntil || isAnsweringUser || isProactiveInterventionActive) {
            return;
        }

        const voiceDiv = document.getElementById('val-voice');

        // Se sento rumore e l'ECA è in IDLE, mettilo in LISTENING!
        if ((interim || final) && eca.currentState === 'IDLE') {
            eca.setState('LISTENING');
        }

        // Pulisco eventuali timeout di ritorno a IDLE vecchi
        if (userMicTimeout) clearTimeout(userMicTimeout);

        if (final) {
            currentFinalTranscript = final.trim();
            voiceDiv.innerHTML = `<span style="color: #0f172a; font-weight: 600;">${final}</span>`;
            console.log(`[UTENTE]: "${currentFinalTranscript}"`);

            const textLower = currentFinalTranscript.toLowerCase();
            const isWakeWordSpoken = textLower.startsWith("aura") || textLower.includes("aura");

            if (currentFinalTranscript.length > 5 && isWakeWordSpoken) {
                // L'utente ha fatto una domanda vera: partiamo con il flusso!
                gestisciDomandaUtente(currentFinalTranscript);
            } else {
                // Falso allarme (es. un colpo di tosse o parola breve), torna in IDLE
                eca.setState('IDLE');
            }

            setTimeout(() => { if (currentFinalTranscript === final.trim()) currentFinalTranscript = ""; }, 3000);

        } else if (interim) {
            voiceDiv.innerHTML = `<span style="font-style: italic;">${interim}...</span>`;

            // Imposta un timer di sicurezza: se entro 1.5s non arriva una parola nuova o una conferma (final), 
            // significa che l'utente ha smesso di parlare senza generare una frase valida. Torna in IDLE.
            userMicTimeout = setTimeout(() => {
                if (eca.currentState === 'LISTENING') {
                    eca.setState('IDLE');
                }
            }, 1500);
        }
    }, (statusMessage) => {
        document.getElementById('voice-status').innerText = statusMessage;
    });

    // Avvio microfono qui per avere i permessi, ma bloccato dalla logica sopra!
    voiceInput.start();

    try {
        await eca.loadModel('./models/personaggio.fbx');
        console.log("✅ Modello 3D caricato con successo!");
    } catch (error) {
        console.error("❌ Errore caricamento modello 3D:", error);
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    video.addEventListener('loadeddata', loop);
}

// ─── Chiamata a Gemini ───
async function fetchGeminiResponse(userText) {
    if (!GEMINI_API_KEY) return "Errore API KEY.";

    const slideContextLine = currentSlideContext.trim()
        ? `L'utente sta guardando questa parte della slide: "${currentSlideContext.trim()}".`
        : '';
    const systemPrompt = `Sei Aura, un tutor didattico virtuale.
L'utente sembra: ${currentEmotionState}.
${slideContextLine}
REGOLE: 1. NON presentarti mai. 2. Vai dritto al sodo. 3. Sii concisa (max 3 frasi). 4. Solo testo semplice. ASSOLUTAMENTE NESSUNA FORMATTAZIONE. Vietato usare LaTeX, vietati gli asterischi (*), vietato il simbolo del dollaro ($). Scrivi le formule matematiche a parole in italiano (es. "a al quadrato più b al quadrato uguale c al quadrato").`;

    const url = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${GEMINI_API_KEY}`;

    const contents = [
        { role: "user", parts: [{ text: systemPrompt }] },
        { role: "model", parts: [{ text: "Ricevuto." }] },
        ...chatHistory.slice(-10),
        { role: "user", parts: [{ text: userText }] }
    ];

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ contents: contents })
        });
        const data = await response.json();
        return data.candidates[0].content.parts[0].text;
    } catch (e) {
        return "Errore di connessione.";
    }
}

// ─── FLUSSO CONVERSAZIONALE LLM ───
async function gestisciDomandaUtente(domandaText) {
    if (isAnsweringUser || isProactiveInterventionActive) return;

    isAnsweringUser = true;

    addChatMessage('user', domandaText);
    // Risponde per dire "Ho capito"
    await parlaECA("Certo, dammi un secondo.");

    // ORA entra in THINKING mentre simula la ricerca su LLM
    console.log("🧠 [AURA] Sto pensando alla risposta...");
    eca.setState('THINKING');

    const aiResponseTesto = await fetchGeminiResponse(domandaText);
    addChatMessage('ai', aiResponseTesto);
    await parlaECA(aiResponseTesto);

    lastProactiveIntervention = performance.now();
    isAnsweringUser = false;
    eca.setState('IDLE');
}

function extractGazedText() {
    if (pdfTextData.length === 0) return "";
    const canvases = document.querySelectorAll('.pdf-slide');
    for (let i = 0; i < canvases.length; i++) {
        const rect = canvases[i].getBoundingClientRect();
        if (currentSmoothPos.x >= rect.left && currentSmoothPos.x <= rect.right &&
            currentSmoothPos.y >= rect.top  && currentSmoothPos.y <= rect.bottom) {
            if (!pdfTextData[i]) return "";
            const localX = currentSmoothPos.x - rect.left;
            const localY = currentSmoothPos.y - rect.top;
            const RADIUS = 80;
            const nearby = pdfTextData[i].items
                .filter(item => Math.hypot(item.x - localX, item.y - localY) < RADIUS)
                .map(item => item.text.trim())
                .filter(t => t.length > 0);
            return nearby.join(' ');
        }
    }
    return "";
}

function drawFaceMeshSegments(landmarks) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const drawPath = (indices, color, closePath = false) => {
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        for (let i = 0; i < indices.length; i++) {
            const pt = landmarks[indices[i]];
            const x = pt.x * canvas.width;
            const y = pt.y * canvas.height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        if (closePath) ctx.closePath();
        ctx.stroke();
    };

    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.brows, 'rgba(43, 87, 151, 0.8)');
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.leftEye, 'rgba(0, 180, 0, 0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.rightEye, 'rgba(0, 180, 0, 0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.outerLips, 'rgba(185, 29, 71, 0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.innerLips, 'rgba(185, 29, 71, 0.8)', true);
}

async function loop() {
    const ts = performance.now();

    if (video.currentTime !== lastVideoTime) {
        const dtSec = (ts - lastFrameTimeMs) / 1000.0;
        lastFrameTimeMs = ts;
        lastVideoTime = video.currentTime;

        const dtSecClamped = Math.min(dtSec, 0.1);

        eca.update(dtSecClamped);

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const results = faceLandmarker.detectForVideo(video, ts);

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            drawFaceMeshSegments(landmarks);

            const rawMetrics = FaceMetricsExtractor.extractRawMetrics(landmarks);
            if (!rawMetrics) {
                requestAnimationFrame(loop);
                return;
            }

            currentNormalizedIris = gazeEstimator.getRobustGazeVector(landmarks);

            if (!isGazeCalibrating && gazeCalibrator.regressionModel) {
                const depthScale = 0.20 / rawMetrics.iod;
                const depthCorrectedX = currentNormalizedIris.x * depthScale;
                const depthCorrectedY = currentNormalizedIris.y * depthScale;

                const rawPos = gazeCalibrator.predict(depthCorrectedX, depthCorrectedY);

                // 🟢 FIX CRASH: Se rawPos è valido, muovi il pallino
                if (rawPos) {
                    const smoothPos = uiFilter.filter(rawPos.x, rawPos.y, ts);
                    currentSmoothPos = smoothPos;

                    gazeDot.style.left = `${smoothPos.x}px`;
                    gazeDot.style.top = `${smoothPos.y}px`;
                    document.getElementById('val-gaze').innerText = `X: ${Math.round(smoothPos.x)}, Y: ${Math.round(smoothPos.y)}`;

                    if (isPdfLoaded && ts - lastGazeContextTs > 500) {
                        lastGazeContextTs = ts;
                        currentSlideContext = extractGazedText();
                    }
                }
            }

            if (affectAnalyzer.isCalibrating) {
                const done = affectAnalyzer.processCalibrationSample(rawMetrics);
                if (done) {
                    calOverlay.style.display = 'none';
                    document.getElementById('val-status').innerText = "Calibrazione completata. Rilevamento attivo.";
                    console.log("✅ Calibrazione emotiva completata!");

                    setTimeout(async () => {
                        isProactiveInterventionActive = true;
                        try {
                            const saluto = "Calibrazione completata. Ciao, io sono Aura. Sono un sistema di assistenza proattivo. Rilevo la tua attenzione e se vedo che sei in difficoltà interverrò per darti una mano. Per qualsiasi domanda, chiedi pure.";
                            addChatMessage('ai', saluto);
                            await parlaECA(saluto);
                        } finally {
                            isProactiveInterventionActive = false;
                            console.log("✅ [AURA] Presentazione conclusa, in ascolto.");
                        }
                    }, 1000);
                }
            } else if (affectAnalyzer.isCalibrated) {
                const state = affectAnalyzer.update(rawMetrics, dtSecClamped);

                if (ts - lastLogTs > 1000) {
                    lastLogTs = ts;
                }

                // Aggiorna stato emotivo per Gemini
                if (state.isFrustrated) currentEmotionState = "Molto Frustrato";
                else if (state.isConfused) currentEmotionState = "Confuso";
                else currentEmotionState = "Attento";

                document.getElementById('val-au4').innerText = `${state.zCorrugator.toFixed(2)} σ`;
                document.getElementById('val-ear').innerText = `${state.zEar.toFixed(2)} σ`;

                const asymEl = document.getElementById('val-asym');
                if (asymEl) asymEl.innerText = `${rawMetrics.browAsymmetry.toFixed(3)}`;

                const cardStatus = document.getElementById('card-status');
                cardStatus.classList.remove('alert', 'warning', 'info');

                if (!isProactiveInterventionActive && !isAnsweringUser) {
                    if (state.isSpeaking) {
                        cardStatus.classList.add('info');
                        document.getElementById('val-status').innerText = "VAD: Movimento Labiale...";
                    } else if (state.isFrustrated) {
                        cardStatus.classList.add('alert');
                        document.getElementById('val-status').innerText = `SOVRACCARICO (${state.stressPercentage.toFixed(0)}%)`;
                    } else if (state.isConfused) {
                        cardStatus.classList.add('warning');
                        document.getElementById('val-status').innerText = `CONFUSIONE RILEVATA`;
                    } else if (state.isBored) {
                        cardStatus.classList.add('info');
                        document.getElementById('val-status').innerText = `NOIA (${state.boredomPercentage.toFixed(0)}%)`;
                    } else {
                        document.getElementById('val-status').innerText = `Attivo e Regolare (${state.stressPercentage.toFixed(0)}%)`;
                    }
                }

                // ─── INTERVENTO PROATTIVO ───
                const timeSinceLastIntervention = ts - lastProactiveIntervention;
                const cooldownPassato = timeSinceLastIntervention > PROACTIVE_COOLDOWN_MS;
                const puoIntervenireTempo = !primoInterventoFatto || cooldownPassato;

                // 🟢 AURA INTERVIENE SOLO SE LE SLIDE SONO CARICATE (isPdfLoaded)
                if (isPdfLoaded && (state.isFrustrated || state.isConfused) && puoIntervenireTempo && !isProactiveInterventionActive && !isAnsweringUser) {
                    console.log(`🤖 [AURA] Intervento INNESCATO!`);

                    isProactiveInterventionActive = true;
                    primoInterventoFatto = true;
                    lastProactiveIntervention = ts;

                    const fraseIntervento = state.isConfused
                        ? "Vedo che sei un po' confuso. C'è qualche concetto di questo argomento che vorresti che ti rispiegassi?"
                        : "Sembri frustrato. Vuoi che facciamo una piccola pausa o preferisci affrontare questo passaggio insieme?";

                    addChatMessage('ai', fraseIntervento);
                    parlaECA(fraseIntervento).then(() => {
                        console.log("⏳ [AURA] Intervento concluso.");
                        isProactiveInterventionActive = false;
                    });
                }

                if (gazeCalibrator.regressionModel) {
                    sessionData.push({
                        timestamp: ts.toFixed(2),
                        dtSec: dtSecClamped.toFixed(4),
                        smoothGazeX: currentSmoothPos.x.toFixed(2),
                        smoothGazeY: currentSmoothPos.y.toFixed(2),
                        zAU4: state.zCorrugator.toFixed(2),
                        zEar: state.zEar.toFixed(2),
                        stressPercentage: state.stressPercentage.toFixed(2),
                        boredomPercentage: state.boredomPercentage.toFixed(2),
                        isFrustrated: state.isFrustrated ? 1 : 0,
                        isBored: state.isBored ? 1 : 0,
                        isConfused: state.isConfused ? 1 : 0,
                        isSpeaking: state.isSpeaking ? 1 : 0,
                        spokenText: currentFinalTranscript
                    });
                }
            }
        }
    }
    requestAnimationFrame(loop);
}

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
        gazeCalibrator.reset();
        uiFilter.reset();
        isGazeCalibrating = true;
        calibrationUI.start();
    };

    btnCalEmotion.onclick = () => {
        btnCalEmotion.blur();

        eca.currentAudio.src = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";
        eca.currentAudio.play().catch(() => { });

        calOverlay.style.display = 'block';
        document.getElementById('val-status').innerText = "Acquisizione Baseline...";
        affectAnalyzer.startCalibration();
        console.log("🎯 Calibrazione emotiva avviata...");
    };

    // ── TASTO SPAZIO: registra punto di calibrazione sguardo ──
    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && isGazeCalibrating) {
            e.preventDefault();
            if (!e.repeat && currentNormalizedIris) {
                const target = calibrationUI.getNextPointCoords();
                gazeCalibrator.recordDataPoint(
                    currentNormalizedIris.x, currentNormalizedIris.y,
                    target.x, target.y
                );
                calibrationUI.advance();
            }
        }
    });

    const btnExport = document.getElementById('btn-export');
    if (btnExport) {
        btnExport.onclick = () => {
            btnExport.blur();
            if (sessionData.length === 0) {
                alert("Nessun dato registrato.");
                return;
            }
            const headers = Object.keys(sessionData[0]).join(",");
            const rows = sessionData.map(row => {
                if (row.spokenText) row.spokenText = row.spokenText.replace(/,/g, ';');
                return Object.values(row).join(",");
            }).join("\n");

            const csvContent = "data:text/csv;charset=utf-8," + headers + "\n" + rows;
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", "aura_test_data.csv");
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        };
    }

    // ── Bottoni vista PDF / Webcam ──
    const btnShowPdf = document.getElementById('btn-show-pdf');
    const btnShowCam = document.getElementById('btn-show-cam');

    if (btnShowPdf) {
        btnShowPdf.onclick = () => {
            document.getElementById('video-wrapper').style.display = 'none';
            document.getElementById('pdf-wrapper').style.display = 'block';
            btnShowPdf.style.display = 'none';
            btnShowCam.style.display = 'inline-block';
            document.querySelectorAll('.metric-card').forEach(el => el.style.display = 'none');
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) chatContainer.style.display = 'flex';
        };
    }

    if (btnShowCam) {
        btnShowCam.onclick = () => {
            document.getElementById('pdf-wrapper').style.display = 'none';
            document.getElementById('video-wrapper').style.display = 'block';
            btnShowCam.style.display = 'none';
            btnShowPdf.style.display = 'inline-block';
            document.querySelectorAll('.metric-card').forEach(el => el.style.display = 'block');
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) chatContainer.style.display = 'none';
        };
    }

    // ── Gestione PDF ──
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

    function handlePdfUpload(file) {
        if (!file || file.type !== "application/pdf") return alert("Carica un PDF.");

        const pdfjsLib = window['pdfjs-dist/build/pdf'];
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

        // Mostra spinner PDF e nascondi drop zone
        if (dropZone) dropZone.style.display = 'none';
        if (slidesContainer) slidesContainer.style.display = 'none';
        if (pdfSpinner) pdfSpinner.style.display = 'block';

        // Voce in parallelo — non aspettiamo che finisca
        const auraPromise = parlaECA("Sto caricando le slide e ci vorrà poco tempo.");

        const fileReader = new FileReader();
        fileReader.onload = async function () {
            try {
                const pdf = await pdfjsLib.getDocument(new Uint8Array(this.result)).promise;
                slidesContainer.innerHTML = '';
                slidesContainer.style.display = 'block';
                pdfTextData = [];
                currentSlideContext = "";

                for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                    const page = await pdf.getPage(pageNum);
                    const viewport = page.getViewport({ scale: 1.5 });
                    const slideCanvas = document.createElement('canvas');
                    slideCanvas.id = "slide_" + pageNum;
                    slideCanvas.className = 'pdf-slide';
                    slideCanvas.height = viewport.height;
                    slideCanvas.width = viewport.width;
                    slidesContainer.appendChild(slideCanvas);
                    await page.render({ canvasContext: slideCanvas.getContext('2d'), viewport: viewport }).promise;

                    const textContent = await page.getTextContent();
                    const items = textContent.items
                        .filter(item => item.str && item.str.trim().length > 0)
                        .map(item => {
                            const tx = item.transform[4];
                            const ty = item.transform[5];
                            const vx = tx * viewport.scale;
                            const vy = viewport.height - ty * viewport.scale;
                            return { text: item.str, x: vx, y: vy };
                        });
                    pdfTextData.push({ items });
                }

                // Aspetta che l'avatar finisca di dire la prima frase, se non l'ha già fatto
                await auraPromise;

                // ── Spinner si nasconde SOLO dopo tutte le slide nel DOM ──
                if (pdfSpinner) pdfSpinner.style.display = 'none';

                const msg = `Caricamento completato. Ci sono ${pdf.numPages} pagine. Chiamami dicendo "Aura" per farmi una domanda.`;
                addChatMessage('ai', msg);
                await parlaECA(msg);

                // 🟢 SBLOCCA IL MICROFONO E GLI INTERVENTI PROATTIVI
                isPdfLoaded = true;
                lastProactiveIntervention = performance.now(); // Resetta il cooldown per evitare spam immediato
                console.log("✅ PDF caricato. Aura è in ascolto e pronta a intervenire.");

            } catch (error) {
                console.error(error);
                if (pdfSpinner) pdfSpinner.style.display = 'none';
                alert("Errore durante il caricamento del PDF.");
            }
        };
        fileReader.readAsArrayBuffer(file);
    }

    init();
});