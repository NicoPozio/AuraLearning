import { FilesetResolver, FaceLandmarker } from "@mediapipe/tasks-vision";
import { GazeEstimator } from "./sensors/GazeEstimator.js";
import { FaceMetricsExtractor } from "./sensors/FaceMetricsExtractor.js";
import { FrustrationAnalyzer } from "./analysis/FrustrationAnalyzer.js";
import { GazeCalibrator } from "./calibration/GazeCalibrator.js";
import { CalibrationUI } from "./calibration/CalibrationUI.js";
import { OneEuroFilter } from "./utils/OneEuroFilter.js";

// 1. Dichiarazione delle variabili a livello di modulo (senza inizializzazione sincrona)
let video, canvas, ctx, gazeDot;
let btnCalGaze, btnCalEmotion, calOverlay;

let faceLandmarker;
let lastVideoTime = -1;
let lastFrameTimeMs = performance.now();
let isGazeCalibrating = false;
let currentNormalizedIris = null;

// Istanziamento dei motori logici
const emotionAnalyzer = new FrustrationAnalyzer();
const gazeCalibrator = new GazeCalibrator();
const uiFilter = new OneEuroFilter(60, 0.5, 0.01, 1.0);
let calibrationUI;

// 2. Inizializzazione asincrona dei tensori
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

    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
    video.srcObject = stream;
    video.addEventListener('loadeddata', loop);
}

// 3. Motore di rendering topologico
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

    // Rendering vettoriale delle aree di interesse
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.brows, 'rgba(43, 87, 151, 0.8)');
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.leftEye, 'rgba(0, 180, 0, 0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.rightEye, 'rgba(0, 180, 0, 0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.outerLips, 'rgba(185, 29, 71, 0.8)', true);
    drawPath(FaceMetricsExtractor.RENDER_SEGMENTS.innerLips, 'rgba(185, 29, 71, 0.8)', true);
}

// 4. Ciclo di elaborazione principale (Event Loop)
async function loop() {
    const ts = performance.now();

    if (video.currentTime !== lastVideoTime) {
        const dtSec = (ts - lastFrameTimeMs) / 1000.0;
        lastFrameTimeMs = ts;
        lastVideoTime = video.currentTime;
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const results = faceLandmarker.detectForVideo(video, ts);

        if (results.faceLandmarks && results.faceLandmarks.length > 0) {
            const landmarks = results.faceLandmarks[0];
            drawFaceMeshSegments(landmarks);
            
            // Gestione Sguardo
            currentNormalizedIris = GazeEstimator.getRobustGazeVector(landmarks);
            
            if (!isGazeCalibrating && gazeCalibrator.regressionModel) {
                const rawPos = gazeCalibrator.predict(currentNormalizedIris.x, currentNormalizedIris.y);
                const smoothPos = uiFilter.filter(rawPos.x, rawPos.y, ts);
                
                gazeDot.style.left = `${smoothPos.x}px`;
                gazeDot.style.top = `${smoothPos.y}px`;
                document.getElementById('val-gaze').innerText = `X: ${Math.round(smoothPos.x)}, Y: ${Math.round(smoothPos.y)}`;
            }

            // Gestione Emotiva
            const rawMetrics = FaceMetricsExtractor.extractRawMetrics(landmarks);

            if (emotionAnalyzer.isCalibrating) {
                const done = emotionAnalyzer.processCalibrationSample(rawMetrics);
                if (done) {
                    calOverlay.style.display = 'none';
                    document.getElementById('val-status').innerText = "Calibrazione completata. Rilevamento attivo.";
                }
            } else if (emotionAnalyzer.isCalibrated) {
                const state = emotionAnalyzer.update(rawMetrics, dtSec);
                
                document.getElementById('val-au4').innerText = `${state.zCorrugator.toFixed(2)} σ`;
                document.getElementById('val-ear').innerText = `${state.zEar.toFixed(2)} σ`;
                document.getElementById('val-lip').innerText = `${state.zLip.toFixed(2)} σ`;

                const cardStatus = document.getElementById('card-status');
                if (state.isFrustrated) {
                    cardStatus.classList.add('alert');
                    document.getElementById('val-status').innerText = `SOVRACCARICO (${state.stressPercentage.toFixed(0)}%)`;
                } else {
                    cardStatus.classList.remove('alert');
                    document.getElementById('val-status').innerText = `Normale (${state.stressPercentage.toFixed(0)}%)`;
                }
            }
        }
    }
    requestAnimationFrame(loop);
}

// 5. Inizializzazione sicura ancorata al DOM
document.addEventListener("DOMContentLoaded", () => {
    // Acquisizione riferimenti fisici post-rendering
    video = document.getElementById('webcam');
    canvas = document.getElementById('output-canvas');
    ctx = canvas.getContext('2d');
    gazeDot = document.getElementById('gaze-dot');

    btnCalGaze = document.getElementById('btn-cal-gaze');
    btnCalEmotion = document.getElementById('btn-cal-emotion');
    calOverlay = document.getElementById('cal-overlay');

    // Validazione strutturale
    if (!btnCalGaze || !btnCalEmotion) {
        console.error("ERRORE ARCHITETTURALE: Nodi DOM mancanti. Assicurarsi che il file index.html sia aggiornato e la cache del browser svuotata.");
        return;
    }

    // Configurazione Event Listeners con inibizione del focus (blur)
    btnCalGaze.onclick = () => {
        btnCalGaze.blur();
        isGazeCalibrating = true;
        calibrationUI.start();
    };

    btnCalEmotion.onclick = () => {
        btnCalEmotion.blur();
        calOverlay.style.display = 'block';
        document.getElementById('val-status').innerText = "Acquisizione Baseline...";
        emotionAnalyzer.startCalibration();
    };

    window.addEventListener('keydown', (e) => {
        if (e.code === 'Space' && isGazeCalibrating) {
            e.preventDefault(); // Previene lo scroll della pagina
            if (!e.repeat && currentNormalizedIris) {
                const target = calibrationUI.getNextPointCoords();
                gazeCalibrator.recordDataPoint(currentNormalizedIris.x, currentNormalizedIris.y, target.x, target.y);
                calibrationUI.advance();
            }
        }
    });

    // Avvio della pipeline computazionale
    init();
});