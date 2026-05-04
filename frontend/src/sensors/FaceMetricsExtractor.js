export class FaceMetricsExtractor {
    static LEFT_INNER_BROW  = 55;
    static RIGHT_INNER_BROW = 285;

    static LEFT_BROW_PEAK   = 65;
    static RIGHT_BROW_PEAK  = 295;

    static LEFT_EYE_CENTER  = 468;
    static RIGHT_EYE_CENTER = 473;

    static LEFT_OUTER_EYE  = 33;
    static RIGHT_OUTER_EYE = 263;

    static L_EYE_TOP    = 159;
    static L_EYE_BOTTOM = 145;
    static L_EYE_INNER  = 133;

    static R_EYE_TOP    = 386;
    static R_EYE_BOTTOM = 374;
    static R_EYE_INNER  = 362;
    static R_EYE_OUTER  = 263;

    static LIP_UPPER_OUTER = 0;
    static LIP_LOWER_OUTER = 17;
    static LIP_UPPER_INNER = 13;
    static LIP_LOWER_INNER = 14;
    static LIP_LEFT_CORNER  = 61;
    static LIP_RIGHT_CORNER = 291;

    static NOSE_TIP = 1;

    // AU9 proxy: lati del naso (alae nasi) vs radice del naso
    // Quando AU9 (naso arricciato) si attiva, le ali del naso si alzano
    // e si avvicinano alla radice (landmark 168)
    static NOSE_ROOT      = 168;
    static NOSE_LEFT_ALA  = 129;  // ala sinistra
    static NOSE_RIGHT_ALA = 358;  // ala destra

    static RENDER_SEGMENTS = {
        brows:     [46, 53, 52, 65, 55, 285, 295, 282, 283, 276],
        leftEye:   [33, 160, 158, 133, 153, 144, 33],
        rightEye:  [362, 385, 387, 263, 373, 380, 362],
        outerLips: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 314, 17, 84, 181, 91, 61],
        innerLips: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    };

    static _dist(p1, p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
    }

    static extractRawMetrics(landmarks) {
        if (!landmarks || landmarks.length < 478) return null;

        const iod = this._dist(landmarks[this.LEFT_OUTER_EYE], landmarks[this.RIGHT_OUTER_EYE]);
        if (iod < 1e-5) return null;

        // ── AU4 (corrugatore) ─────────────────────────────────────────────
        const leftBrowDrop  = this._dist(landmarks[this.LEFT_INNER_BROW],  landmarks[this.LEFT_EYE_CENTER]);
        const rightBrowDrop = this._dist(landmarks[this.RIGHT_INNER_BROW], landmarks[this.RIGHT_EYE_CENTER]);
        const corrugator    = ((leftBrowDrop + rightBrowDrop) / 2) / iod;
        const browAsymmetry = Math.abs(leftBrowDrop - rightBrowDrop) / iod;

        // ── AU1 (inner brow raise) ────────────────────────────────────────
        // Misura la distanza VERTICALE (asse Y) tra l'apice mediale del sopracciglio
        // e il bordo superiore dell'occhio, normalizzata per IOD.
        // In MediaPipe Y cresce verso il basso, quindi un sopracciglio ALZATO
        // ha Y più piccola → differenza (browPeak.y - eyeTop.y) diventa più negativa.
        // Invertiamo il segno: innerBrowRaise > 0 = sopracciglio alzato.
        const leftBrowRaise  = -(landmarks[this.LEFT_BROW_PEAK].y  - landmarks[this.L_EYE_TOP].y) / iod;
        const rightBrowRaise = -(landmarks[this.RIGHT_BROW_PEAK].y - landmarks[this.R_EYE_TOP].y) / iod;
        const innerBrowRaise = (leftBrowRaise + rightBrowRaise) / 2;

        // ── EAR ──────────────────────────────────────────────────────────
        const lEyeH  = this._dist(landmarks[this.L_EYE_TOP],    landmarks[this.L_EYE_BOTTOM]);
        const lEyeW  = this._dist(landmarks[this.L_EYE_INNER],  landmarks[this.LEFT_OUTER_EYE]);
        const rEyeH  = this._dist(landmarks[this.R_EYE_TOP],    landmarks[this.R_EYE_BOTTOM]);
        const rEyeW  = this._dist(landmarks[this.R_EYE_INNER],  landmarks[this.R_EYE_OUTER]);
        const earLeft  = lEyeW > 1e-5 ? lEyeH / lEyeW : 0;
        const earRight = rEyeW > 1e-5 ? rEyeH / rEyeW : 0;
        const ear = (earLeft + earRight) / 2;

        // ── EAR per singolo occhio (utile per asimmetria nel blink) ───────
        const earAsymmetry = Math.abs(earLeft - earRight);

        // ── AU24 (lip press) ──────────────────────────────────────────────
        const outerLipH = this._dist(landmarks[this.LIP_UPPER_OUTER], landmarks[this.LIP_LOWER_OUTER]);
        const innerLipH = this._dist(landmarks[this.LIP_UPPER_INNER], landmarks[this.LIP_LOWER_INNER]);
        const lipPress  = outerLipH > 1e-5 ? innerLipH / outerLipH : 0;

        // ── AU12 (sorriso) ────────────────────────────────────────────────
        const mouthWidth   = this._dist(landmarks[this.LIP_LEFT_CORNER], landmarks[this.LIP_RIGHT_CORNER]);
        const smileIntensity = mouthWidth / iod;

        // ── AU9 proxy (naso arricciato) ───────────────────────────────────
        // Distanza tra le ali del naso e la radice del naso, normalizzata.
        // Quando AU9 si attiva, le ali si alzano → distanza verticale diminuisce.
        const leftAlaRaise  = (landmarks[this.NOSE_ROOT].y - landmarks[this.NOSE_LEFT_ALA].y)  / iod;
        const rightAlaRaise = (landmarks[this.NOSE_ROOT].y - landmarks[this.NOSE_RIGHT_ALA].y) / iod;
        const noseWrinkle   = (leftAlaRaise + rightAlaRaise) / 2;  // più alto = più arricciato

        // ── Head roll ─────────────────────────────────────────────────────
        // Angolo di rotazione laterale della testa (roll), in radianti.
        // Calcolato dal vettore tra i centri degli occhi (iris landmarks).
        // |headRoll| > ~0.08rad (~4.5°) è percepibile come inclinazione.
        const lIris = landmarks[this.LEFT_EYE_CENTER];
        const rIris = landmarks[this.RIGHT_EYE_CENTER];
        const headRoll = Math.atan2(rIris.y - lIris.y, rIris.x - lIris.x);

        const noseTip = landmarks[this.NOSE_TIP];

        return {
            corrugator,
            ear,
            earLeft,
            earRight,
            earAsymmetry,
            lipPress,
            iod,
            browAsymmetry,
            smileIntensity,
            innerBrowRaise,   // AU1
            noseWrinkle,      // AU9 proxy
            headRoll,         // roll laterale testa
            noseX: noseTip.x,
            noseY: noseTip.y
        };
    }
}