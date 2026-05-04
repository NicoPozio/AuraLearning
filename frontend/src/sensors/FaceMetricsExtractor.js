export class FaceMetricsExtractor {
    static LEFT_INNER_BROW = 55;
    static RIGHT_INNER_BROW = 285;

    static LEFT_EYE_CENTER = 468;
    static RIGHT_EYE_CENTER = 473;

    static LEFT_OUTER_EYE = 33;
    static RIGHT_OUTER_EYE = 263;

    static L_EYE_TOP = 159;
    static L_EYE_BOTTOM = 145;
    static L_EYE_INNER = 133;

    static R_EYE_TOP = 386;
    static R_EYE_BOTTOM = 374;
    static R_EYE_INNER = 362;
    static R_EYE_OUTER = 263;

    static LIP_UPPER_OUTER = 0;
    static LIP_LOWER_OUTER = 17;
    static LIP_UPPER_INNER = 13;
    static LIP_LOWER_INNER = 14;

    // Aggiunto per il rilevamento del Sorriso (AU12)
    static LIP_LEFT_CORNER = 61;
    static LIP_RIGHT_CORNER = 291;

    static NOSE_TIP = 1;

    static RENDER_SEGMENTS = {
        brows: [46, 53, 52, 65, 55, 285, 295, 282, 283, 276],
        leftEye: [33, 160, 158, 133, 153, 144, 33],
        rightEye: [362, 385, 387, 263, 373, 380, 362],
        outerLips: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 314, 17, 84, 181, 91, 61],
        innerLips: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    };

    static _dist(p1, p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
    }

    static extractRawMetrics(landmarks) {
        if (!landmarks || landmarks.length < 478) return null;

        const iod = this._dist(
            landmarks[this.LEFT_OUTER_EYE],
            landmarks[this.RIGHT_OUTER_EYE]
        );

        if (iod < 1e-5) return null;

        const leftBrowDrop = this._dist(landmarks[this.LEFT_INNER_BROW], landmarks[this.LEFT_EYE_CENTER]);
        const rightBrowDrop = this._dist(landmarks[this.RIGHT_INNER_BROW], landmarks[this.RIGHT_EYE_CENTER]);

        const corrugator = ((leftBrowDrop + rightBrowDrop) / 2) / iod;
        const browAsymmetry = Math.abs(leftBrowDrop - rightBrowDrop) / iod;

        const lEyeH = this._dist(landmarks[this.L_EYE_TOP], landmarks[this.L_EYE_BOTTOM]);
        const lEyeW = this._dist(landmarks[this.L_EYE_INNER], landmarks[this.LEFT_OUTER_EYE]);
        const rEyeH = this._dist(landmarks[this.R_EYE_TOP], landmarks[this.R_EYE_BOTTOM]);
        const rEyeW = this._dist(landmarks[this.R_EYE_INNER], landmarks[this.R_EYE_OUTER]);

        const earLeft = lEyeW > 1e-5 ? lEyeH / lEyeW : 0;
        const earRight = rEyeW > 1e-5 ? rEyeH / rEyeW : 0;
        const ear = (earLeft + earRight) / 2;

        const outerLipH = this._dist(landmarks[this.LIP_UPPER_OUTER], landmarks[this.LIP_LOWER_OUTER]);
        const innerLipH = this._dist(landmarks[this.LIP_UPPER_INNER], landmarks[this.LIP_LOWER_INNER]);
        const lipPress = outerLipH > 1e-5 ? innerLipH / outerLipH : 0;

        // Estrazione Sorriso
        const mouthWidth = this._dist(landmarks[this.LIP_LEFT_CORNER], landmarks[this.LIP_RIGHT_CORNER]);
        const smileIntensity = mouthWidth / iod;

        const noseTip = landmarks[this.NOSE_TIP];

        return { 
            corrugator, 
            ear, 
            lipPress, 
            iod, 
            browAsymmetry,
            smileIntensity,
            noseX: noseTip.x,
            noseY: noseTip.y 
        };
    }
}