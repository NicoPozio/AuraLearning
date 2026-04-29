import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import { GazeEstimator } from "./sensors/GazeEstimator.js";
import { FaceMetricsExtractor } from "./sensors/FaceMetricsExtractor.js";
import { AffectAnalyzer } from "./analysis/AffectAnalyzer.js";
import { GazeCalibrator } from "./calibration/GazeCalibrator.js";
import { CalibrationUI } from "./calibration/CalibrationUI.js";
import { OneEuroFilter } from "./utils/OneEuroFilter.js";
import { VoiceInput } from "./sensors/VoiceInput.js"; // <-- NUOVO IMPORT
import { ECAController } from "./eca/ECAController.js";

let video, canvas, ctx, gazeDot;
let btnCalGaze, btnCalEmotion, calOverlay;

let sessionData = [];
let currentSmoothPos = { x: 0, y: 0 };
let currentFinalTranscript = ""; // <-- Variabile per salvare l'ultima frase nel CSV

let faceLandmarker;
let lastVideoTime = -1;
let lastFrameTimeMs = performance.now();
let isGazeCalibrating = false;
let currentNormalizedIris = null;

const affectAnalyzer = new AffectAnalyzer();
const eca = new ECAController('eca-container');
const gazeCalibrator = new GazeCalibrator();
const uiFilter = new OneEuroFilter(60, 0.1, 0.001, 1.0);
const gazeEstimator = new GazeEstimator();

let calibrationUI;
let voiceInput; // <-- Istanza globale

const lumaCanvas = document.createElement('canvas');
lumaCanvas.width = 32;
lumaCanvas.height = 32;
const lumaCtx = lumaCanvas.getContext('2d', { willReadFrequently: true });

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

    // ─── INIZIALIZZAZIONE MODULO VOCALE ───
    voiceInput = new VoiceInput((interim, final) => {
        const voiceDiv = document.getElementById('val-voice');
        if (final) {
            currentFinalTranscript = final.trim();
            voiceDiv.innerHTML = `<span style="color: #0f172a; font-weight: 600;">${final}</span>`;
            // Resetta la frase dopo 3 secondi nel CSV per non duplicarla all'infinito
            setTimeout(() => { if (currentFinalTranscript === final.trim()) currentFinalTranscript = ""; }, 3000);
        } else if (interim) {
            voiceDiv.innerHTML = `<span style="font-style: italic;">${interim}...</span>`;
        }
    }, (statusMessage) => {
        document.getElementById('voice-status').innerText = statusMessage;
    });

    // Avvia l'ascolto appena la webcam è pronta
    voiceInput.start();

    // ─── CARICAMENTO MODELLO 3D (ECA) ───
    try {
        await eca.loadModel('./models/avatar.glb');
        console.log("Modello 3D caricato con successo!");
    } catch (error) {
        console.error("Errore durante il caricamento del modello 3D:", error);
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: true }); // Aggiunto audio: true
    video.srcObject = stream;
    video.addEventListener('loadeddata', loop);
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

        // --- CHIAMATA DI AGGIORNAMENTO ECA ---
        eca.update(dtSecClamped);
        // -------------------------------------

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

            const dx = landmarks[263].x - landmarks[33].x;
            const dy = landmarks[263].y - landmarks[33].y;
            const dzYaw = landmarks[263].z - landmarks[33].z;

            const roll = Math.atan2(dy, dx) * (180 / Math.PI);
            const yaw = Math.atan2(dzYaw, dx) * (180 / Math.PI);

            const dyPitch = landmarks[152].y - landmarks[10].y;
            const dzPitch = landmarks[152].z - landmarks[10].z;
            const pitch = Math.atan2(dzPitch, dyPitch) * (180 / Math.PI);

            lumaCtx.drawImage(video, 0, 0, 32, 32);
            const lumaData = lumaCtx.getImageData(0, 0, 32, 32).data;
            let lumaSum = 0;
            for (let i = 0; i < lumaData.length; i += 4) {
                lumaSum += (lumaData[i] * 0.299 + lumaData[i + 1] * 0.587 + lumaData[i + 2] * 0.114);
            }
            const averageLuma = lumaSum / 1024;

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
                }
            } else if (affectAnalyzer.isCalibrated) {
                const state = affectAnalyzer.update(rawMetrics, dtSecClamped);

                document.getElementById('val-au4').innerText = `${state.zCorrugator.toFixed(2)} σ`;
                document.getElementById('val-ear').innerText = `${state.zEar.toFixed(2)} σ`;
                document.getElementById('val-asym').innerText = `${rawMetrics.browAsymmetry.toFixed(3)}`;

                const cardStatus = document.getElementById('card-status');
                cardStatus.classList.remove('alert', 'warning', 'info');

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
                        spokenText: currentFinalTranscript // <-- NUOVO: Esportazione della frase pronunciata
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
        calOverlay.style.display = 'block';
        document.getElementById('val-status').innerText = "Acquisizione Baseline...";
        affectAnalyzer.startCalibration();
    };

    const btnExport = document.getElementById('btn-export');
    if (btnExport) {
        btnExport.onclick = () => {
            btnExport.blur();
            if (sessionData.length === 0) {
                alert("Nessun dato registrato. Esegui prima le calibrazioni.");
                return;
            }
            const headers = Object.keys(sessionData[0]).join(",");
            const rows = sessionData.map(row => {
                // Sostituisce le virgole nelle frasi parlate con un punto e virgola per non rompere il CSV
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

    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && isGazeCalibrating) {
            e.preventDefault();
            if (!e.repeat && currentNormalizedIris) {
                const target = calibrationUI.getNextPointCoords();
                gazeCalibrator.recordDataPoint(currentNormalizedIris.x, currentNormalizedIris.y, target.x, target.y);
                calibrationUI.advance();
            }
        }

        // --- NUOVO: Tasto 'T' per testare il Lip-Sync ---
        if (e.code === 'KeyT') {
            eca.speak("Ciao! Sono il tuo assistente Aura. Sto testando il movimento della mia bocca in tempo reale.");
        }
        // ------------------------------------------------
    });

    init();
});