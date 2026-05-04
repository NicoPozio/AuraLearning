export class GazeEstimator {
    static L_OUTER = 33;
    static L_INNER = 133;
    static L_TOP = 159;
    static L_BOTTOM = 145;

    static R_INNER = 362;
    static R_OUTER = 263;
    static R_TOP = 386;
    static R_BOTTOM = 374;

    static NOSE_TIP = 1;
    static NOSE_BASE = 168;
    static L_CHEEK = 234;
    static R_CHEEK = 454;
    static CHIN = 152;
    static FOREHEAD = 10;

    #lastValidGaze = { x: 0, y: 0 };

    getRobustGazeVector(landmarks) {
        if (!landmarks || landmarks.length < 478) return this.#lastValidGaze;

        const lOuter = landmarks[GazeEstimator.L_OUTER];
        const lInner = landmarks[GazeEstimator.L_INNER];
        const lTop = landmarks[GazeEstimator.L_TOP];
        const lBottom = landmarks[GazeEstimator.L_BOTTOM];

        const lWidth = Math.hypot(lInner.x - lOuter.x, lInner.y - lOuter.y);
        const lHeight = Math.hypot(lBottom.x - lTop.x, lBottom.y - lTop.y);
        const lCenterX = (lOuter.x + lInner.x) / 2;
        const lCenterY = (lTop.y + lBottom.y) / 2;

        const rInner = landmarks[GazeEstimator.R_INNER];
        const rOuter = landmarks[GazeEstimator.R_OUTER];
        const rTop = landmarks[GazeEstimator.R_TOP];
        const rBottom = landmarks[GazeEstimator.R_BOTTOM];

        const rWidth = Math.hypot(rOuter.x - rInner.x, rOuter.y - rInner.y);
        const rHeight = Math.hypot(rBottom.x - rTop.x, rBottom.y - rTop.y);
        const rCenterX = (rInner.x + rOuter.x) / 2;
        const rCenterY = (rTop.y + rBottom.y) / 2;

        if (lWidth < 1e-5 || rWidth < 1e-5) return this.#lastValidGaze;

        // Prevenzione Glitch da Ammiccamento (Blink Suspension / Bell's Phenomenon)
        const earLeft = lHeight / lWidth;
        const earRight = rHeight / rWidth;
        const ear = (earLeft + earRight) / 2;
        
        // Se l'occhio è chiuso o in fase di battito, interrompi il calcolo vettoriale
        if (ear < 0.15) {
            return this.#lastValidGaze; 
        }

        const lIris = [474, 475, 476, 477].reduce((s, i) => ({ x: s.x + landmarks[i].x / 4, y: s.y + landmarks[i].y / 4 }), { x: 0, y: 0 });
        const rIris = [469, 470, 471, 472].reduce((s, i) => ({ x: s.x + landmarks[i].x / 4, y: s.y + landmarks[i].y / 4 }), { x: 0, y: 0 });

        const lGazeX = (lIris.x - lCenterX) / lWidth;
        const lGazeY = (lIris.y - lCenterY) / lHeight;
        const rGazeX = (rIris.x - rCenterX) / rWidth;
        const rGazeY = (rIris.y - rCenterY) / rHeight;

        const rawX = (lGazeX + rGazeX) / 2;
        const rawY = (lGazeY + rGazeY) / 2;

        const noseTip = landmarks[GazeEstimator.NOSE_TIP];
        const lCheek = landmarks[GazeEstimator.L_CHEEK];
        const rCheek = landmarks[GazeEstimator.R_CHEEK];
        const lCheekDist = Math.abs(noseTip.x - lCheek.x);
        const rCheekDist = Math.abs(noseTip.x - rCheek.x);
        const faceWidth = lCheekDist + rCheekDist;

        const yawSignal = faceWidth > 1e-5 ? (rCheekDist - lCheekDist) / faceWidth : 0;
        const rollSignal = Math.atan2(rCenterY - lCenterY, rCenterX - lCenterX);

        const YAW_GAIN = 0.65;
        const ROLL_GAIN = 0.20;

        // Compensazione Non-Lineare (Trigonometrica) per la rotazione di Yaw
        const correctedX = rawX - (Math.sin(yawSignal * Math.PI) * YAW_GAIN);
        const correctedY = rawY - (rollSignal * ROLL_GAIN);

        const chin = landmarks[GazeEstimator.CHIN];
        const forehead = landmarks[GazeEstimator.FOREHEAD];
        const nosePitchY = landmarks[GazeEstimator.NOSE_BASE];

        const faceHeightY = Math.abs(chin.y - forehead.y);
        const noseRelY = faceHeightY > 1e-5 ? (nosePitchY.y - forehead.y) / faceHeightY : 0.5;

        const PITCH_CENTER = 0.48;
        const PITCH_GAIN = 0.30;
        const pitchOffset = (noseRelY - PITCH_CENTER) * PITCH_GAIN;

        const finalX = correctedX;
        const finalY = correctedY - pitchOffset;

        const MAX_OFFSET = 0.5;
        this.#lastValidGaze = {
            x: Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, finalX)),
            y: Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, finalY))
        };

        return this.#lastValidGaze;
    }
}