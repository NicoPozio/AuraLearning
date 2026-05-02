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

let faceLandmarker;
let lastVideoTime = -1;
let lastFrameTimeMs = performance.now();
let isGazeCalibrating = false;
let currentNormalizedIris = null;

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

// ─── 🛡️ FUNZIONE WRAPPER ANTI-ECO ───
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

        if (performance.now() < ignoreMicUntil || isAnsweringUser || isProactiveInterventionActive) {
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

            if (currentFinalTranscript.length > 5) {
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

// ─── FLUSSO CONVERSAZIONALE LLM ───
async function gestisciDomandaUtente(domandaText) {
    if (isAnsweringUser || isProactiveInterventionActive) return;

    isAnsweringUser = true;

    // Risponde per dire "Ho capito"
    await parlaECA("Certo, dammi un secondo che elaboro le informazioni.");

    // ORA entra in THINKING mentre simula la ricerca su LLM
    console.log("🧠 [AURA] Sto pensando alla risposta...");
    eca.setState('THINKING');

    await new Promise(resolve => setTimeout(resolve, 3000));

    console.log("💡 [AURA] Risposta trovata!");
    await parlaECA(`Eccomi. Riguardo a quello che mi hai chiesto... facciamo finta che ti stia spiegando perfettamente l'argomento! Sono sempre a disposizione.`);

    lastProactiveIntervention = performance.now();
    isAnsweringUser = false;
    eca.setState('IDLE');
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
                const smoothPos = uiFilter.filter(rawPos.x, rawPos.y, ts);

                currentSmoothPos = smoothPos;

                gazeDot.style.left = `${smoothPos.x}px`;
                gazeDot.style.top = `${smoothPos.y}px`;
                document.getElementById('val-gaze').innerText = `X: ${Math.round(smoothPos.x)}, Y: ${Math.round(smoothPos.y)}`;
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
                            await parlaECA("Calibrazione completata. Ciao, io sono Aura. Sono un sistema di assistenza proattivo. Rilevo la tua attenzione e se vedo che sei in difficoltà interverrò per darti una mano. Per qualsiasi domanda, chiedi pure.");
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

                document.getElementById('val-au4').innerText = `${state.zCorrugator.toFixed(2)} σ`;
                document.getElementById('val-ear').innerText = `${state.zEar.toFixed(2)} σ`;
                document.getElementById('val-asym').innerText = `${rawMetrics.browAsymmetry.toFixed(3)}`;

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

                // RIMOSSO IL BLOCCO DI CODICE CHE CAUSAVA IL CONFLITTO SULLO STATO LISTENING
                // Adesso è solo il microfono (in cima al file) a decidere quando l'ECA va in LISTENING.

                // ─── INTERVENTO PROATTIVO ───
                const timeSinceLastIntervention = ts - lastProactiveIntervention;
                const cooldownPassato = timeSinceLastIntervention > PROACTIVE_COOLDOWN_MS;
                const puoIntervenireTempo = !primoInterventoFatto || cooldownPassato;

                if ((state.isFrustrated || state.isConfused) && puoIntervenireTempo && !isProactiveInterventionActive && !isAnsweringUser) {
                    console.log(`🤖 [AURA] Intervento INNESCATO!`);

                    isProactiveInterventionActive = true;
                    primoInterventoFatto = true;
                    lastProactiveIntervention = ts;

                    const fraseIntervento = state.isConfused
                        ? "Vedo che sei un po' confuso. C'è qualche concetto di questo argomento che vorresti che ti rispiegassi?"
                        : "Sembri frustrato. Vuoi che facciamo una piccola pausa o preferisci affrontare questo passaggio insieme?";

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

    init();
});