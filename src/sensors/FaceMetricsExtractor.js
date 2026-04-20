export class FaceMetricsExtractor {
    static LEFT_INNER_BROW = 55;
    static RIGHT_INNER_BROW = 285;
    static LEFT_OUTER_EYE = 33;
    static RIGHT_OUTER_EYE = 263;
    
    static L_EYE_TOP = 159;
    static L_EYE_BOTTOM = 145;
    static L_EYE_INNER = 133;
    
    static LIP_UPPER_OUTER = 0;
    static LIP_LOWER_OUTER = 17;
    static LIP_UPPER_INNER = 13;
    static LIP_LOWER_INNER = 14;

    // Strutture dati esportate per il rendering vettoriale nel canvas
    static RENDER_SEGMENTS = {
        brows: [46, 53, 52, 65, 55, 285, 295, 282, 283, 276],
        leftEye: [33, 160, 158, 133, 153, 144, 33],
        rightEye: [362, 385, 387, 263, 373, 380, 362],
        outerLips: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 314, 17, 84, 181, 91, 61],
        innerLips: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    };

    static _dist(p1, p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2) + Math.pow(p1.z - p2.z, 2));
    }

    static extractRawMetrics(landmarks) {
        const iod = this._dist(landmarks[this.LEFT_OUTER_EYE], landmarks[this.RIGHT_OUTER_EYE]);
        const browDist = this._dist(landmarks[this.LEFT_INNER_BROW], landmarks[this.RIGHT_INNER_BROW]);
        
        const eyeHeight = this._dist(landmarks[this.L_EYE_TOP], landmarks[this.L_EYE_BOTTOM]);
        const eyeWidth = this._dist(landmarks[this.L_EYE_INNER], landmarks[this.LEFT_OUTER_EYE]);

        const outerLipH = this._dist(landmarks[this.LIP_UPPER_OUTER], landmarks[this.LIP_LOWER_OUTER]);
        const innerLipH = this._dist(landmarks[this.LIP_UPPER_INNER], landmarks[this.LIP_LOWER_INNER]);

        return {
            corrugator: browDist / iod, 
            ear: eyeHeight / eyeWidth,
            lipPress: innerLipH / outerLipH
        };
    }
}