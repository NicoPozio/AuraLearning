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
let documentContextText = "";
let sessionData = [];
let currentSmoothPos = { x: 0, y: 0 };
let currentFinalTranscript = "";
let currentGazedText = ""; 
let textExtractionInterval = null;

let faceLandmarker;
let lastVideoTime = -1;
let lastFrameTimeMs = performance.now();
let isGazeCalibrating = false;
let currentNormalizedIris = null;

let isPdfLoaded = false;

//PROACTIVE AND CONVERSATIONAL PHASE VARIABLES ---
let lastProactiveIntervention = 0;
const PROACTIVE_COOLDOWN_MS = 30000;
let isProactiveInterventionActive = false;
let firstInterventionDone = false;

let isAnsweringUser = false;

// Filters and Smoothing for Mic and Echo
let ignoreMicUntil = 0;
let userMicTimeout = null;

let lastLogTs = 0;

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

// Adds a message to the chat
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

// Anti-echo wrapper to prevent ECA's voice from being captured
async function speakECA(text) {
    ignoreMicUntil = Infinity;
    try {
        await eca.speak(text);
    } catch (e) {
        console.error("TTS Error:", e);
    } finally {
        ignoreMicUntil = performance.now() + 1500;
    }
}

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

    // Microphone management
    voiceInput = new VoiceInput((interim, final) => {

        // If PDF is not loaded, the assistant ignores everything
        if (!isPdfLoaded || performance.now() < ignoreMicUntil || isAnsweringUser || isProactiveInterventionActive) {
            return;
        }

        const voiceDiv = document.getElementById('val-voice');

        // If conditions are met, set assistant to listening
        if ((interim || final) && eca.currentState === 'IDLE') {
            eca.setState('LISTENING');
        }

        // Cleanup any timeouts
        if (userMicTimeout) clearTimeout(userMicTimeout);

        if (final) {
            currentFinalTranscript = final.trim();
            voiceDiv.innerHTML = `<span style="color: #0f172a; font-weight: 600;">${final}</span>`;
            console.log(`[USER]: "${currentFinalTranscript}"`);

            const textLower = currentFinalTranscript.toLowerCase();
            const isWakeWordSpoken = textLower.startsWith("aura") || textLower.includes("aura");

            if (currentFinalTranscript.length > 5 && isWakeWordSpoken) {
                // User asked a question
                manageUserQuestion(currentFinalTranscript);
            } else {
                // False alarm
                eca.setState('IDLE');
            }

            setTimeout(() => { if (currentFinalTranscript === final.trim()) currentFinalTranscript = ""; }, 3000);

        } else if (interim) {
            voiceDiv.innerHTML = `<span style="font-style: italic;">${interim}...</span>`;

            // Security timer: if no new word or final confirmation arrives within 1.5s,
            // it means the user stopped speaking without generating a valid sentence. Return to IDLE.
            userMicTimeout = setTimeout(() => {
                if (eca.currentState === 'LISTENING') {
                    eca.setState('IDLE');
                }
            }, 1500);
        }
    }, (statusMessage) => {
        document.getElementById('voice-status').innerText = statusMessage;
    });

    try {
        await eca.loadModel('./models/personaggio.fbx');
        console.log("3D Model loaded successfully!");
    } catch (error) {
        console.error("Error loading 3D model:", error);
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
    video.srcObject = stream;
    video.addEventListener('loadeddata', loop);

    // --- RECOVERY GAZE CALIBRATION IN RAM ---
    if (gazeCalibrator.loadFromStorage && gazeCalibrator.loadFromStorage()) {
        console.log("[AURA] Gaze calibration successfully recovered from session.");
        gazeDot.style.display = 'block';
    } else {
        console.log("[AURA] No previous calibration found in this session.");
    }
}

async function fetchLLMResponse(userText) {
    const url = "http://localhost:8000/api/chat";
    
    // Secure payload preparation
    const payload = {
        user_text: userText,
        emotion_state: currentEmotionState,
        slide_context: (isPdfLoaded && documentContextText !== "") ? documentContextText : ""
    };

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) throw new Error("Network error with Proxy");
        const data = await response.json();
        
        return data.text;
    } catch (e) {
        console.error(e);
        return "Connection error with central system.";
    }
}

// LLM CONVERSATIONAL FLOW
async function manageUserQuestion(questionText) {
    if (isAnsweringUser || isProactiveInterventionActive) return;

    isAnsweringUser = true;

    addChatMessage('user', questionText);
    // Responds to acknowledge
    await speakECA("Certo, dammi un secondo.");

    // NOW enters THINKING while simulating LLM search
    console.log("[AURA] Thinking about the response...");
    eca.setState('THINKING');

    const llmResponseText = await fetchLLMResponse(questionText);
    addChatMessage('ai', llmResponseText);
    await speakECA(llmResponseText);

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

                if (rawPos) {
                    const smoothPos = uiFilter.filter(rawPos.x, rawPos.y, ts);
                    currentSmoothPos = smoothPos;

                    gazeDot.style.left = `${smoothPos.x}px`;
                    gazeDot.style.top = `${smoothPos.y}px`;
                    document.getElementById('val-gaze').innerText = `X: ${Math.round(smoothPos.x)}, Y: ${Math.round(smoothPos.y)}`;
                }
            }

            if (affectAnalyzer.isCalibrating) {
                const done = affectAnalyzer.processCalibrationSample(rawMetrics);
                if (done) {
                    calOverlay.style.display = 'none';
                    document.getElementById('val-status').innerText = "Calibrazione completata. Rilevamento attivo.";
                    console.log("Emotional calibration completed!");

                    setTimeout(async () => {
                        isProactiveInterventionActive = true;
                        try {
                            const greetingMessage = "Calibrazione completata. Ciao, io sono Aura. Sono un sistema di assistenza proattivo. Rilevo la tua attenzione e se vedo che sei in difficoltà interverrò per darti una mano. Per qualsiasi domanda, chiedi pure.";
                            addChatMessage('ai', greetingMessage);
                            await speakECA(greetingMessage);
                        } finally {
                            isProactiveInterventionActive = false;
                            console.log("[AURA] Presentation concluded, listening.");
                        }
                    }, 1000);
                }
            } else if (affectAnalyzer.isCalibrated) {
                const state = affectAnalyzer.update(rawMetrics, dtSecClamped);

                if (ts - lastLogTs > 1000) {
                    lastLogTs = ts;
                }

                // Update emotional state for LLM
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

                const timeSinceLastIntervention = ts - lastProactiveIntervention;
                const isCooldownPassed = timeSinceLastIntervention > PROACTIVE_COOLDOWN_MS;
                const canInterveneTime = !firstInterventionDone || isCooldownPassed;

                if (isPdfLoaded && (state.isFrustrated || state.isConfused) && canInterveneTime && !isProactiveInterventionActive && !isAnsweringUser) {
                    console.log(`[AURA] Intervention TRIGGERED! Generating intelligent response...`);
                    isProactiveInterventionActive = true;
                    firstInterventionDone = true;
                    lastProactiveIntervention = ts;
                    
                    eca.setState('THINKING');

                    const hiddenPrompt = state.isConfused
                        ? "L'utente mostra confusione guardando l'attuale slide. Formula una brevissima domanda (massimo 15 parole) proponendo di rispiegare un termine tecnico specifico presente nel testo."
                        : "L'utente mostra segni di frustrazione o sovraccarico cognitivo. Fagli una brevissima proposta empatica (massimo 15 parole) citando l'argomento della slide per aiutarlo a sbloccarsi.";

                    fetchLLMResponse(hiddenPrompt).then(interventionPhrase => {
                        addChatMessage('ai', interventionPhrase);
                        speakECA(interventionPhrase).then(() => {
                            console.log("[AURA] Intervention concluded.");
                            isProactiveInterventionActive = false;
                        });
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

        if (voiceInput && !voiceInput.isListening) {
            voiceInput.start();
        }

        eca.currentAudio.src = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA";
        eca.currentAudio.play().catch(() => { });

        calOverlay.style.display = 'block';
        document.getElementById('val-status').innerText = "Acquisizione Baseline...";
        affectAnalyzer.startCalibration();
        console.log("Emotional calibration started...");
    };

    // --- SPACEBAR: records gaze calibration point ---
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

    // --- PDF / Webcam View Buttons ---
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

        if (dropZone) dropZone.style.display = 'none';
        if (slidesContainer) slidesContainer.style.display = 'none';
        if (pdfSpinner) pdfSpinner.style.display = 'block';

        const auraPromise = speakECA("Sto analizzando le slide, preparo il livello semantico.");

        const fileReader = new FileReader();
        fileReader.onload = async function () {
            try {
                const pdf = await pdfjsLib.getDocument(new Uint8Array(this.result)).promise;
                slidesContainer.innerHTML = '';
                slidesContainer.style.display = 'block';
                
                // Resetta il contesto precedente
                documentContextText = "";

                for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                    const page = await pdf.getPage(pageNum);
                    const viewport = page.getViewport({ scale: 1.5 });

                    // Creazione del wrapper per Canvas + TextLayer
                    const pageWrapper = document.createElement('div');
                    pageWrapper.style.position = 'relative';
                    pageWrapper.style.margin = '0 auto 30px auto';
                    pageWrapper.style.width = `${viewport.width}px`;
                    pageWrapper.style.height = `${viewport.height}px`;
                    pageWrapper.className = 'pdf-slide-wrapper';
                    
                    const slideCanvas = document.createElement('canvas');
                    slideCanvas.className = 'pdf-slide-canvas';
                    slideCanvas.height = viewport.height;
                    slideCanvas.width = viewport.width;
                    slideCanvas.style.display = 'block';
                    
                    pageWrapper.appendChild(slideCanvas);
                    slidesContainer.appendChild(pageWrapper);
                    
                    const renderTask = page.render({ canvasContext: slideCanvas.getContext('2d'), viewport: viewport }).promise;
                    
                    // 2. Creazione del TEXT LAYER invisibile per l'Hit Testing
                    const textContent = await page.getTextContent();
                    
                    const textLayerDiv = document.createElement('div');
                    textLayerDiv.className = 'textLayer';
                    textLayerDiv.style.position = 'absolute';
                    textLayerDiv.style.left = '0';
                    textLayerDiv.style.top = '0';
                    textLayerDiv.style.right = '0';
                    textLayerDiv.style.bottom = '0';
                    textLayerDiv.style.overflow = 'hidden';
                    textLayerDiv.style.opacity = '0'; // Invisibile ma interattivo
                    pageWrapper.appendChild(textLayerDiv);

                    await renderTask;

                    pdfjsLib.renderTextLayer({
                        textContentSource: textContent,
                        container: textLayerDiv,
                        viewport: viewport,
                        textDivs: []
                    });
                }

                await auraPromise;
                
                if (pdfSpinner) pdfSpinner.style.display = 'none';
                const msg = `Caricamento completato. Ora posso capire esattamente quale frase stai leggendo.`;
                addChatMessage('ai', msg);
                await speakECA(msg);
                
                isPdfLoaded = true;
                lastProactiveIntervention = performance.now();

                // 3. Avvia il polling per l'Hit Testing semantico (Gaze-Contingency)
                if (textExtractionInterval) clearInterval(textExtractionInterval);
                textExtractionInterval = setInterval(extractGazedText, 500); // Controlla cosa guardi ogni 500ms
                
            } catch (error) {
                console.error(error);
                if (pdfSpinner) pdfSpinner.style.display = 'none';
                alert("Errore critico durante il parsing del PDF.");
            }
        };
        fileReader.readAsArrayBuffer(file);
    }

    // ── Funzione Gaze-Contingent: Estrae il testo sotto gli occhi ──
    function extractGazedText() {
        if (!isPdfLoaded || isGazeCalibrating) return;

        // Disabilita temporaneamente il pointer-events del puntino rosso 
        // per non intercettare se stesso durante l'elementFromPoint
        const wasGazeDotPointerEvents = gazeDot.style.pointerEvents;
        gazeDot.style.pointerEvents = 'none';

        // Hit Testing
        const element = document.elementFromPoint(currentSmoothPos.x, currentSmoothPos.y);
        
        // Se l'elemento è uno span generato dal TextLayer di PDF.js
        if (element && element.parentNode && element.parentNode.classList.contains('textLayer')) {
            // Non prendiamo solo la singola parola, ma unisciamo i testi vicini 
            // per dare all'LLM una frase sensata (Contesto locale)
            const siblings = Array.from(element.parentNode.childNodes);
            const index = siblings.indexOf(element);
            
            // Prende le 3 parole prima e le 6 parole dopo per formare un costrutto logico
            const start = Math.max(0, index - 3);
            const end = Math.min(siblings.length, index + 6);
            
            const contextSnippet = siblings.slice(start, end).map(el => el.textContent).join(' ');
            
            if (contextSnippet.trim().length > 5) {
                currentGazedText = contextSnippet;
                // Debug opzionale: console.log("Gazed Context:", currentGazedText);
            }
        }

        gazeDot.style.pointerEvents = wasGazeDotPointerEvents;
    }

    // Infine, aggiorniamo il payload verso l'LLM (sia proattivo che conversazionale)
    async function fetchLLMResponse(userText) {
        const url = "http://localhost:8000/api/chat";
        
        // Invia ESATTAMENTE il testo che l'utente sta fissando in quel momento
        const payload = {
            user_text: userText,
            emotion_state: currentEmotionState,
            slide_context: (isPdfLoaded && currentGazedText !== "") ? currentGazedText : ""
        };

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) throw new Error("Network error with Proxy");
            const data = await response.json();
            
            return data.text;
        } catch (e) {
            console.error(e);
            return "Connection error with central system.";
        }
    }

    init();
});