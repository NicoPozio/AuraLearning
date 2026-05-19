/**
 * FaceMetricsExtractor — converts the 478 MediaPipe facial landmarks into a
 * compact set of biomechanically meaningful, IOD-normalised metrics that
 * the AffectAnalyzer can later turn into Z-Scores.
 *
 * Two design principles guide every metric in this class:
 *
 *   1. NORMALISATION BY IOD (Inter-Ocular Distance) — every facial distance
 *      is divided by the 3D distance between the two outer eye corners
 *      (landmarks 33 and 263). This produces ratios that are invariant to
 *      face size and to camera distance, so the same metric is comparable
 *      across different users and across leaning forward / backward.
 *
 *   2. PURELY GEOMETRIC FEATURES — no machine-learning classifier is used
 *      here; each metric is a single, interpretable geometric ratio with
 *      a known link to a FACS Action Unit. This keeps the pipeline cheap
 *      (O(1) per frame), debuggable, and independent of training data.
 *
 * The output coordinate frame is still the camera frame; the AffectAnalyzer
 * applies the third (statistical) normalisation layer — the personal
 * Z-Score against a 120-frame neutral baseline.
 */
export class FaceMetricsExtractor {

    // ──────────────────────────────────────────────────────────────────────
    // LANDMARK INDICES (MediaPipe FaceMesh / FaceLandmarker topology)
    // ──────────────────────────────────────────────────────────────────────

    // ── Eyebrows ──
    static LEFT_INNER_BROW = 55;   // inner end of the left brow (closest to the nose)
    static RIGHT_INNER_BROW = 285;  // inner end of the right brow
    static LEFT_BROW_PEAK = 65;   // arc peak of the left brow (highest point)
    static RIGHT_BROW_PEAK = 295;  // arc peak of the right brow

    // ── Iris / eyes ──
    static LEFT_EYE_CENTER = 468;  // iris centre of the left eye (MediaPipe iris model)
    static RIGHT_EYE_CENTER = 473;  // iris centre of the right eye
    static LEFT_OUTER_EYE = 33;   // outer canthus of the left eye  → used as one anchor of IOD
    static RIGHT_OUTER_EYE = 263;  // outer canthus of the right eye → used as the other anchor of IOD
    static L_EYE_TOP = 159;
    static L_EYE_BOTTOM = 145;
    static L_EYE_INNER = 133;
    static R_EYE_TOP = 386;
    static R_EYE_BOTTOM = 374;
    static R_EYE_INNER = 362;
    static R_EYE_OUTER = 263;       // same point as RIGHT_OUTER_EYE — duplicated for readability

    // ── Lips ──
    static LIP_UPPER_OUTER = 0;     // top of the outer upper lip
    static LIP_LOWER_OUTER = 17;    // bottom of the outer lower lip
    static LIP_UPPER_INNER = 13;    // top of the inner upper lip
    static LIP_LOWER_INNER = 14;    // bottom of the inner lower lip
    static LIP_LEFT_CORNER = 61;   // left mouth corner (commissure)
    static LIP_RIGHT_CORNER = 291;  // right mouth corner (commissure)

    // ── Nose ──
    static NOSE_TIP = 1;      // tip of the nose
    static NOSE_ROOT = 168;    // root / bridge between the eyes
    static NOSE_LEFT_ALA = 129;    // left nostril wing
    static NOSE_RIGHT_ALA = 358;    // right nostril wing

    // ──────────────────────────────────────────────────────────────────────
    // RENDER SEGMENTS — polyline indices used by main.js to draw the
    // landmark wireframe on top of the webcam feed. Pure presentation
    // metadata; the analyser does not use these.
    // ──────────────────────────────────────────────────────────────────────
    static RENDER_SEGMENTS = {
        brows: [46, 53, 52, 65, 55, 285, 295, 282, 283, 276],
        leftEye: [33, 160, 158, 133, 153, 144, 33],
        rightEye: [362, 385, 387, 263, 373, 380, 362],
        outerLips: [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 314, 17, 84, 181, 91, 61],
        innerLips: [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
    };

    /**
     * 3D Euclidean distance between two MediaPipe landmarks.
     *
     * MediaPipe returns a z-coordinate alongside x and y (estimated depth
     * relative to the centre of the face), so we use the full 3D form
     * rather than the 2D projection — this makes distances slightly more
     * stable under head rotation.
     *
     * @param {{x:number,y:number,z:number}} p1
     * @param {{x:number,y:number,z:number}} p2
     * @returns {number} Euclidean distance in normalised image units.
     * @private
     */
    static _dist(p1, p2) {
        return Math.hypot(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
    }

    /**
     * Extract every biomechanical metric from a single landmark frame.
     *
     * Output object keys correspond directly to the Z-Score names produced
     * by the AffectAnalyzer (e.g. `corrugator` → `zCorrugator`).
     *
     * Returns `null` and bails out if:
     *   • the input is missing or has fewer than 478 landmarks
     *     (MediaPipe iris indices 468–477 would be undefined), or
     *   • the IOD is degenerate (≈ 0), which would cause a division-by-zero
     *     in every ratio and contaminate the baseline.
     *
     * @param {Array<{x:number,y:number,z:number}>} landmarks
     *        478 facial landmarks from MediaPipe FaceLandmarker.
     * @returns {Object|null} Dictionary of raw, IOD-normalised metrics, or
     *        `null` when the frame cannot be processed.
     */
    static extractRawMetrics(landmarks) {
        // Validate the input mesh has the expected size (iris model present)
        if (!landmarks || landmarks.length < 478) return null;

        // Reference length for all subsequent ratios. Computed once per frame
        // because every metric below depends on it.
        const iod = this._dist(landmarks[this.LEFT_OUTER_EYE], landmarks[this.RIGHT_OUTER_EYE]);
        // Degenerate-geometry guard: a near-zero IOD would blow up the ratios
        if (iod < 1e-5) return null;

        // ── AU4: corrugator (brow furrow) ────────────────────────────────
        // We measure the VERTICAL drop of each inner brow relative to the
        // centre of the eye underneath, NOT the horizontal distance between
        // the two inner brows. This choice makes the metric immune to side
        // lighting: a strong lateral light casts a shadow on the bridge of
        // the nose that visually brings the two brow landmarks closer in
        // the X direction, but does not move them vertically.
        const leftBrowDrop = this._dist(landmarks[this.LEFT_INNER_BROW], landmarks[this.LEFT_EYE_CENTER]);
        const rightBrowDrop = this._dist(landmarks[this.RIGHT_INNER_BROW], landmarks[this.RIGHT_EYE_CENTER]);
        const corrugator = ((leftBrowDrop + rightBrowDrop) / 2) / iod;
        // Asymmetry between the two sides — useful to filter false positives
        // due to one-sided shadows or partial occlusion
        const browAsymmetry = Math.abs(leftBrowDrop - rightBrowDrop) / iod;

        // ── AU1: inner brow raise (perplexity) ──────────────────────────
        // Distance from each brow peak to the upper eyelid, with a NEGATED
        // sign because MediaPipe's Y axis grows downwards: a higher brow
        // has a smaller (more negative) y, so flipping the sign makes the
        // metric grow when the brow rises.
        const leftBrowRaise = -(landmarks[this.LEFT_BROW_PEAK].y - landmarks[this.L_EYE_TOP].y) / iod;
        const rightBrowRaise = -(landmarks[this.RIGHT_BROW_PEAK].y - landmarks[this.R_EYE_TOP].y) / iod;
        const innerBrowRaise = (leftBrowRaise + rightBrowRaise) / 2;

        // ── EAR (Eye Aspect Ratio) — AU43/46 ─────────────────────────────
        // Eye height divided by eye width, averaged across the two eyes.
        // Drops on squinting, blinking, or visual fatigue. NOT divided by
        // IOD because the eye width already lives in the same image scale.
        const lEyeH = this._dist(landmarks[this.L_EYE_TOP], landmarks[this.L_EYE_BOTTOM]);
        const lEyeW = this._dist(landmarks[this.L_EYE_INNER], landmarks[this.LEFT_OUTER_EYE]);
        const rEyeH = this._dist(landmarks[this.R_EYE_TOP], landmarks[this.R_EYE_BOTTOM]);
        const rEyeW = this._dist(landmarks[this.R_EYE_INNER], landmarks[this.R_EYE_OUTER]);
        const earLeft = lEyeW > 1e-5 ? lEyeH / lEyeW : 0;
        const earRight = rEyeW > 1e-5 ? rEyeH / rEyeW : 0;
        const ear = (earLeft + earRight) / 2;

        // ── AU24: lip press (tension, holding back a reaction) ──────────
        // Ratio between the height of the INNER lip aperture and the
        // height of the OUTER lip aperture. When the lips are pressed
        // together the inner aperture collapses while the outer height
        // remains roughly constant, so the ratio drops sharply.
        const outerLipH = this._dist(landmarks[this.LIP_UPPER_OUTER], landmarks[this.LIP_LOWER_OUTER]);
        const innerLipH = this._dist(landmarks[this.LIP_UPPER_INNER], landmarks[this.LIP_LOWER_INNER]);
        // Guard against outerLipH ≈ 0 (extreme lip occlusion / weird mouth shape)
        const lipPress = outerLipH > 1e-5 ? innerLipH / outerLipH : 0;

        // ── Mouth open (surprise / confusion indicator) ─────────────────
        // Absolute outer-lip opening, normalised by IOD. Distinct from
        // lipPress: lipPress measures relative inner/outer compression
        // (low = closed), while mouthOpen measures the actual opening
        // (high = jaw dropped).
        const mouthOpen = outerLipH / iod;

        // ── AU12: smile intensity (positive emotion) ────────────────────
        // Width of the mouth (distance between commissures) divided by IOD.
        // Grows when the corners are pulled outward in a smile.
        const mouthWidth = this._dist(landmarks[this.LIP_LEFT_CORNER], landmarks[this.LIP_RIGHT_CORNER]);
        const smileIntensity = mouthWidth / iod;

        // ── AU15: mouth corner depression (contrariness / disgust) ──────
        // How far the corners sit BELOW the centre of the upper lip.
        // MediaPipe's Y grows downwards, therefore:
        //   cornersMeanY > upperLipY  → corners lower than the upper lip
        //                              → "inverted U" shape, i.e. a frown
        // Normalised by IOD. Important corner case: during speech the
        // corners oscillate but do NOT systematically drop, so this
        // metric remains reasonably specific to negative affect.
        const cornersMeanY = (landmarks[this.LIP_LEFT_CORNER].y + landmarks[this.LIP_RIGHT_CORNER].y) / 2;
        const mouthCurvature = (cornersMeanY - landmarks[this.LIP_UPPER_OUTER].y) / iod;

        // ── AU9: nose wrinkle (disgust / disappointment) ────────────────
        // Vertical lift of each nostril wing (ala) relative to the nose
        // root. Y grows downwards, so subtracting (alaY - rootY) gives a
        // value that increases when the wings move UP — consistent with
        // the muscular action of AU9.
        const leftAlaRaise = (landmarks[this.NOSE_ROOT].y - landmarks[this.NOSE_LEFT_ALA].y) / iod;
        const rightAlaRaise = (landmarks[this.NOSE_ROOT].y - landmarks[this.NOSE_RIGHT_ALA].y) / iod;
        const noseWrinkle = (leftAlaRaise + rightAlaRaise) / 2;

        // ── Head roll (auxiliary signal for the orchestrator) ───────────
        // Angle of the line connecting the two iris centres. Same idea as
        // GazeEstimator's rollSignal but computed from iris centres rather
        // than eye corners, which is slightly more stable on small heads.
        const lIris = landmarks[this.LEFT_EYE_CENTER];
        const rIris = landmarks[this.RIGHT_EYE_CENTER];
        const headRoll = Math.atan2(rIris.y - lIris.y, rIris.x - lIris.x);

        // Cached for telemetry; not used by the affect classifier itself
        const noseTip = landmarks[this.NOSE_TIP];

        // ── Output bundle ───────────────────────────────────────────────
        // Field names match the AffectAnalyzer's expected input keys.
        return {
            corrugator,        // AU4   — brow furrow
            ear, earLeft, earRight, // AU43/46 — eye closure (with per-eye debug values)
            lipPress,          // AU24  — lip press
            mouthOpen,         //        — jaw drop / surprise
            mouthCurvature,    // AU15  — mouth-corner depression
            iod,               //        — reference length, exported for downstream depth correction
            browAsymmetry,     //        — left/right brow imbalance (debug / filtering)
            smileIntensity,    // AU12  — smile
            innerBrowRaise,    // AU1   — inner brow raise
            noseWrinkle,       // AU9   — nose wrinkle
            headRoll,          //        — head tilt around the camera Z axis
            noseX: noseTip.x,  //        — nose-tip image coordinates (CSV telemetry only)
            noseY: noseTip.y
        };
    }
}