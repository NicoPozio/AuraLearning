export class GazeEstimator {
    // Riferimenti Occhio Sinistro
    static L_OUTER = 33;
    static L_INNER = 133;
    static L_TOP = 159;
    static L_BOTTOM = 145;

    // Riferimenti Occhio Destro
    static R_INNER = 362;
    static R_OUTER = 263;
    static R_TOP = 386;
    static R_BOTTOM = 374;

    // Landmark di riferimento per la posa della testa (nasione e zigomi)
    // Usati per stimare yaw e roll e sottrarne la componente dal gaze.
    static NOSE_TIP = 1;
    static NOSE_BASE = 168;
    static L_CHEEK = 234;
    static R_CHEEK = 454;
    static CHIN = 152;
    static FOREHEAD = 10;

    #lastValidGaze = { x: 0, y: 0 };

    getRobustGazeVector(landmarks) {
        if (!landmarks || landmarks.length < 478) return this.#lastValidGaze;

        // Centroidi Iridi (Media dei 4 punti)
        const lIris = [474, 475, 476, 477].reduce(
            (s, i) => ({ x: s.x + landmarks[i].x / 4, y: s.y + landmarks[i].y / 4 }),
            { x: 0, y: 0 }
        );
        const rIris = [469, 470, 471, 472].reduce(
            (s, i) => ({ x: s.x + landmarks[i].x / 4, y: s.y + landmarks[i].y / 4 }),
            { x: 0, y: 0 }
        );

        // --- CALCOLO OCCHIO SINISTRO ---
        const lOuter = landmarks[GazeEstimator.L_OUTER];
        const lInner = landmarks[GazeEstimator.L_INNER];
        const lTop = landmarks[GazeEstimator.L_TOP];
        const lBottom = landmarks[GazeEstimator.L_BOTTOM];

        const lWidth = Math.hypot(lInner.x - lOuter.x, lInner.y - lOuter.y);
        const lHeight = Math.hypot(lBottom.x - lTop.x, lBottom.y - lTop.y);
        const lCenterX = (lOuter.x + lInner.x) / 2;
        const lCenterY = (lTop.y + lBottom.y) / 2;

        // --- CALCOLO OCCHIO DESTRO ---
        const rInner = landmarks[GazeEstimator.R_INNER];
        const rOuter = landmarks[GazeEstimator.R_OUTER];
        const rTop = landmarks[GazeEstimator.R_TOP];
        const rBottom = landmarks[GazeEstimator.R_BOTTOM];

        const rWidth = Math.hypot(rOuter.x - rInner.x, rOuter.y - rInner.y);
        const rHeight = Math.hypot(rBottom.x - rTop.x, rBottom.y - rTop.y);
        const rCenterX = (rInner.x + rOuter.x) / 2;
        const rCenterY = (rTop.y + rBottom.y) / 2;

        if (lWidth < 1e-5 || rWidth < 1e-5) return this.#lastValidGaze;

        // Normalizziamo l'iride RISPETTO AL CENTRO DELL'OCCHIO (invarianza spaziale)
        const lGazeX = (lIris.x - lCenterX) / lWidth;
        const lGazeY = (lIris.y - lCenterY) / lHeight;
        const rGazeX = (rIris.x - rCenterX) / rWidth;
        const rGazeY = (rIris.y - rCenterY) / rHeight;

        // Media binoculare grezza
        const rawX = (lGazeX + rGazeX) / 2;
        const rawY = (lGazeY + rGazeY) / 2;

        // ─────────────────────────────────────────────────────────────────────
        // [FIX #1 — BUG CRITICO] Compensazione della posa della testa (Yaw + Roll)
        //
        // PROBLEMA ORIGINALE: correlazione rawGazeX ↔ headYaw = -0.90 nel CSV.
        // Il gaze stimolato rispecchiava quasi interamente la rotazione della testa,
        // non il movimento dell'iride. Causa: quando giri la testa a destra,
        // il centro proiettato degli occhi si sposta a destra, ma i punti iris
        // si spostano MENO (sono quasi al centro della pupilla che non si muove).
        // Il risultato normalizzato (iris - centro) / larghezza sembrava un gaze
        // verso sinistra anche senza nessun movimento oculare reale.
        //
        // SOLUZIONE: stimiamo yaw e roll dalla geometria della testa (zigomi, mento)
        // e li sottraiamo come offset lineare dal gaze grezzo.
        // Il fattore di scala (yawGain, rollGain) è empirico — calibrato sul CSV.
        // ─────────────────────────────────────────────────────────────────────

        // Yaw: asimmetria sinistra/destra della larghezza facciale
        // Quando la testa è frontale: lCheekDist ≈ rCheekDist → yaw ≈ 0
        // Quando gira a destra: il lato sinistro appare più stretto → yawSignal > 0
        const noseTip = landmarks[GazeEstimator.NOSE_TIP];
        const lCheek = landmarks[GazeEstimator.L_CHEEK];
        const rCheek = landmarks[GazeEstimator.R_CHEEK];
        const lCheekDist = Math.abs(noseTip.x - lCheek.x);
        const rCheekDist = Math.abs(noseTip.x - rCheek.x);
        const faceWidth = lCheekDist + rCheekDist;

        // yawSignal ∈ [-0.5, +0.5]: positivo = testa girata a sinistra (verso cam)
        // Nota: il segno è invertito rispetto all'asse schermo per matchare rawGazeX
        const yawSignal = faceWidth > 1e-5 ? (rCheekDist - lCheekDist) / faceWidth : 0;

        // Roll: inclinazione laterale della testa (angolo inter-occhi)
        const eyeLineAngle = Math.atan2(rCenterY - lCenterY, rCenterX - lCenterX);
        // rollSignal in radianti: piccolo (±0.2 rad tipicamente)
        const rollSignal = eyeLineAngle;

        // Fattori di guadagno empirici (calibrati sul CSV dove yaw spiegava il 90% di rawGazeX)
        // yawGain: quanto yawSignal contribuisce all'offset di gazeX
        // rollGain: quanto il roll contribuisce all'offset di gazeY
        const YAW_GAIN = 0.65;
        const ROLL_GAIN = 0.20;

        const correctedX = rawX - (yawSignal * YAW_GAIN);
        const correctedY = rawY - (rollSignal * ROLL_GAIN);

        // ─────────────────────────────────────────────────────────────────────
        // [FIX #2] Compensazione della distanza dalla webcam (Pitch + IOD)
        //
        // PROBLEMA ORIGINALE: rawGazeY aveva una media di -0.12 (sistematicamente
        // negativa) perché quando sei vicino alla webcam, il centro geometrico
        // degli occhi si trova in una posizione diversa rispetto a quando sei lontano.
        // La variazione IOD nel CSV era del 24.5% (range 0.114 - 0.274) — enorme.
        //
        // SOLUZIONE: usiamo la posizione relativa naso/mento sull'asse Y per
        // stimare il pitch (inclinazione verticale della testa) e sottrarlo da gazeY.
        // ─────────────────────────────────────────────────────────────────────
        const chin = landmarks[GazeEstimator.CHIN];
        const forehead = landmarks[GazeEstimator.FOREHEAD];
        const nosePitchY = landmarks[GazeEstimator.NOSE_BASE];

        // Posizione verticale relativa del naso nell'asse viso (0 = alto, 1 = basso)
        const faceHeightY = Math.abs(chin.y - forehead.y);
        const noseRelY = faceHeightY > 1e-5
            ? (nosePitchY.y - forehead.y) / faceHeightY
            : 0.5;

        // Offset verticale: deviazione da centro fisiologico (≈ 0.48 sperimentale)
        const PITCH_CENTER = 0.48;
        const PITCH_GAIN = 0.30;
        const pitchOffset = (noseRelY - PITCH_CENTER) * PITCH_GAIN;

        const finalX = correctedX;
        const finalY = correctedY - pitchOffset;

        // Clipping di sicurezza (per evitare output folli durante i blink)
        const MAX_OFFSET = 0.5;
        this.#lastValidGaze = {
            x: Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, finalX)),
            y: Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, finalY))
        };

        return this.#lastValidGaze;
    }
}