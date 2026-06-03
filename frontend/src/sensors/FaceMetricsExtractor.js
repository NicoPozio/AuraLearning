/**
 * FaceMetricsExtractor — converts the 478 MediaPipe facial landmarks into
 * a compact set of biomechanically meaningful, IOD-normalised metrics that
 * the AffectAnalyzer turns into Z-Scores.
 *
 * Two design principles:
 *
 *   1) Normalisation by IOD (Inter-Ocular Distance). Every facial distance
 *     is divided by the 3D distance between the two outer eye corners
 *     (landmarks 33 and 263). This produces ratios that are invariant to
 *     face size and to camera distance, so the same metric is comparable
 *     across different users and across leaning forward / backward.
 *
 *   2) Purely geometric features. No machine-learning classifier; each
 *     metric is a single interpretable ratio with a known link to a FACS
 *     Action Unit. Cheap (O(1) per frame), debuggable, and free of
 *     training-data dependencies.
 *
 * The output frame is still camera space; the AffectAnalyzer applies the
 * third (statistical) normalisation layer — the personal Z-Score against
 * the 120-frame neutral baseline.
 */
export class FaceMetricsExtractor {

    // Landmark indices follow the MediaPipe FaceMesh / FaceLandmarker topology.

    // Eyebrows
    static LEFT_INNER_BROW  = 55;
    static RIGHT_INNER_BROW = 285;
    static LEFT_BROW_PEAK   = 65;
    static RIGHT_BROW_PEAK  = 295;

    // Iris and eye corners. LEFT_OUTER_EYE and RIGHT_OUTER_EYE double as
    // the IOD anchors.
    static LEFT_EYE_CENTER  = 468;
    static RIGHT_EYE_CENTER = 473;
    static LEFT_OUTER_EYE   = 33;
    static RIGHT_OUTER_EYE  = 263;
    static L_EYE_TOP        = 159;
    static L_EYE_BOTTOM     = 145;
    static L_EYE_INNER      = 133;
    static R_EYE_TOP        = 386;
    static R_EYE_BOTTOM     = 374;
    static R_EYE_INNER      = 362;
    static R_EYE_OUTER      = 263;  // same as RIGHT_OUTER_EYE, kept for readability

    // Lips
    static LIP_UPPER_OUTER  = 0;
    static LIP_LOWER_OUTER  = 17;
    static LIP_UPPER_INNER  = 13;
    static LIP_LOWER_INNER  = 14;
    static LIP_LEFT_CORNER  = 61;
    static LIP_RIGHT_CORNER = 291;

    // Nose
    static NOSE_ROOT      = 168;
    static NOSE_LEFT_ALA  = 129;
    static NOSE_RIGHT_ALA = 358;

    // Polyline indices used by main.js to draw the landmark wireframe on
    // top of the webcam feed. Presentation-only metadata.
    static RENDER_SEGMENTS = {
        brows:      [46, 53, 52, 65, 55, 285, 295, 282, 283, 276],
        leftEye:    [33, 160, 158, 133, 153, 144, 33],
        rightEye:   [362, 385, 387, 263, 373, 380, 362],
        outerLips:  [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 314, 17, 84, 181, 91, 61],
        innerLips:  [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    };

    /**
     * 3D Euclidean distance between two MediaPipe landmarks. MediaPipe
     * returns an estimated z alongside x and y; using the full 3D form
     * keeps distances slightly more stable under head rotation than the
     * 2D projection would.
     */
    static _dist(p1, p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
    }

    /**
     * Extract every biomechanical metric from a single landmark frame.
     * Output keys correspond directly to the Z-Score names produced by
     * the AffectAnalyzer (e.g. `corrugator` → `zCorrugator`).
     *
     * Returns null when:
     *   - the input is missing or has fewer than 478 landmarks
     *     (iris indices 468–477 would be undefined), or
     *   - the IOD is degenerate (≈ 0), which would divide by zero in
     *     every ratio and contaminate the baseline.
     *
     * @param {Array<{x:number,y:number,z:number}>} landmarks
     * @returns {Object|null}
     */
    static extractRawMetrics(landmarks) {
        if (!landmarks || landmarks.length < 478) return null;

        // IOD: reference length for all subsequent ratios.
        const iod = this._dist(landmarks[this.LEFT_OUTER_EYE], landmarks[this.RIGHT_OUTER_EYE]);
        if (iod < 1e-5) return null;

        // AU4 — corrugator (brow furrow).
        // We use the VERTICAL drop of each inner brow relative to the
        // eye centre underneath, not the horizontal distance between
        // the two inner brows. A strong side light casts a shadow on
        // the nose bridge that brings the two brow landmarks closer
        // in X but does not move them vertically — the vertical drop
        // is robust to that artifact.
        const leftBrowDrop  = this._dist(landmarks[this.LEFT_INNER_BROW],  landmarks[this.LEFT_EYE_CENTER]);
        const rightBrowDrop = this._dist(landmarks[this.RIGHT_INNER_BROW], landmarks[this.RIGHT_EYE_CENTER]);
        const corrugator = ((leftBrowDrop + rightBrowDrop) / 2) / iod;

        // AU1 — inner brow raise. Distance from each brow peak to the
        // upper eyelid, negated because MediaPipe's Y axis grows
        // downwards: a higher brow has a smaller y, so flipping the
        // sign makes the metric grow when the brow rises.
        const leftBrowRaise  = -(landmarks[this.LEFT_BROW_PEAK].y  - landmarks[this.L_EYE_TOP].y) / iod;
        const rightBrowRaise = -(landmarks[this.RIGHT_BROW_PEAK].y - landmarks[this.R_EYE_TOP].y) / iod;
        const innerBrowRaise = (leftBrowRaise + rightBrowRaise) / 2;

        // EAR (Eye Aspect Ratio) — AU43/46. Eye height over eye width,
        // averaged across both eyes. Drops on squinting, blinking, or
        // visual fatigue. Not divided by IOD because the width already
        // lives in the same image scale.
        const lEyeH = this._dist(landmarks[this.L_EYE_TOP],   landmarks[this.L_EYE_BOTTOM]);
        const lEyeW = this._dist(landmarks[this.L_EYE_INNER], landmarks[this.LEFT_OUTER_EYE]);
        const rEyeH = this._dist(landmarks[this.R_EYE_TOP],   landmarks[this.R_EYE_BOTTOM]);
        const rEyeW = this._dist(landmarks[this.R_EYE_INNER], landmarks[this.R_EYE_OUTER]);
        const earLeft  = lEyeW > 1e-5 ? lEyeH / lEyeW : 0;
        const earRight = rEyeW > 1e-5 ? rEyeH / rEyeW : 0;
        const ear = (earLeft + earRight) / 2;

        // AU24 — lip press (tension, holding back a reaction). Ratio of
        // the inner lip aperture to the outer one. When the lips are
        // pressed together the inner aperture collapses while the outer
        // height stays roughly constant, so the ratio drops sharply.
        const outerLipH = this._dist(landmarks[this.LIP_UPPER_OUTER], landmarks[this.LIP_LOWER_OUTER]);
        const innerLipH = this._dist(landmarks[this.LIP_UPPER_INNER], landmarks[this.LIP_LOWER_INNER]);
        const lipPress = outerLipH > 1e-5 ? innerLipH / outerLipH : 0;

        // mouthOpen — absolute outer-lip opening normalised by IOD.
        // Distinct from lipPress: lipPress is a relative inner/outer
        // compression ratio (low = closed); mouthOpen measures the
        // actual opening (high = jaw dropped).
        const mouthOpen = outerLipH / iod;

        // AU12 — smile intensity. Mouth width (between commissures)
        // divided by IOD. Grows when the corners are pulled outward.
        const mouthWidth = this._dist(landmarks[this.LIP_LEFT_CORNER], landmarks[this.LIP_RIGHT_CORNER]);
        const smileIntensity = mouthWidth / iod;

        // AU15 — mouth-corner depression (frown). How far the corners
        // sit below the centre of the upper lip. Y grows downwards, so
        // cornersMeanY > upperLipY means corners are lower than the
        // upper lip — an inverted-U shape. During speech the corners
        // oscillate but do not systematically drop, so this metric
        // stays reasonably specific to negative affect.
        const cornersMeanY = (landmarks[this.LIP_LEFT_CORNER].y + landmarks[this.LIP_RIGHT_CORNER].y) / 2;
        const mouthCurvature = (cornersMeanY - landmarks[this.LIP_UPPER_OUTER].y) / iod;

        // AU9 — nose wrinkle (disgust / disappointment). Vertical lift
        // of each nostril wing (ala) relative to the nose root. Y grows
        // downwards, so (rootY - alaY) is positive when the wings move
        // up — consistent with the muscular action of AU9.
        const leftAlaRaise  = (landmarks[this.NOSE_ROOT].y - landmarks[this.NOSE_LEFT_ALA].y)  / iod;
        const rightAlaRaise = (landmarks[this.NOSE_ROOT].y - landmarks[this.NOSE_RIGHT_ALA].y) / iod;
        const noseWrinkle = (leftAlaRaise + rightAlaRaise) / 2;

        return {
            corrugator,       // AU4   — brow furrow
            ear,              // AU43/46 — eye aperture
            lipPress,         // AU24  — lip press
            mouthOpen,        //       — jaw drop / surprise
            mouthCurvature,   // AU15  — mouth-corner depression
            iod,              //       — reference length, used by main.js for depth correction
            smileIntensity,   // AU12  — smile
            innerBrowRaise,   // AU1   — inner brow raise
            noseWrinkle       // AU9   — nose wrinkle
        };
    }
}