/**
 * GazeEstimator — turns raw MediaPipe facial landmarks into a stable,
 * head-pose-corrected gaze vector expressed in normalised camera space.
 *
 * The output is NOT yet a pixel on the screen: it is the direction of the
 * gaze in the webcam frame, in the range [-0.5, +0.5] on both axes.
 * Mapping to actual screen pixels is the responsibility of GazeCalibrator
 * (Thin-Plate Spline) further down the pipeline.
 *
 * Internal flow (executed every frame):
 *   1. Compute the binocular eye geometry (centre + width + height).
 *   2. Bail out if the user is blinking (Eye Aspect Ratio below 0.15),
 *      returning the last valid sample to avoid Bell's-phenomenon spikes.
 *   3. Compute a binocular iris centroid by averaging the 4 iris landmarks
 *      of each eye, then express it relative to each eye's centre and
 *      normalise by eye size (a spatial low-pass filter against jitter).
 *   4. Apply non-linear head-pose compensation for yaw (horizontal head
 *      rotation), roll (head tilt) and pitch (head nodding).
 *   5. Clamp the result to [-0.5, +0.5] and remember it as the last
 *      valid sample for the blink-suspension fallback.
 */
export class GazeEstimator {

    // ── MediaPipe landmark indices (FaceLandmarker / FaceMesh topology) ──

    // Left eye corners: outer (33, temple side), inner (133, nose side),
    // top eyelid (159), bottom eyelid (145).
    static L_OUTER = 33;
    static L_INNER = 133;
    static L_TOP = 159;
    static L_BOTTOM = 145;

    // Right eye corners: inner (362, nose side), outer (263, temple side),
    // top eyelid (386), bottom eyelid (374).
    static R_INNER = 362;
    static R_OUTER = 263;
    static R_TOP = 386;
    static R_BOTTOM = 374;

    // Facial reference points for head-pose estimation
    static NOSE_TIP = 1;     // tip of the nose, used for the yaw asymmetry signal
    static NOSE_BASE = 168;  // bridge of the nose, used for the pitch signal
    static L_CHEEK = 234;    // left cheekbone (zygomatic)
    static R_CHEEK = 454;    // right cheekbone (zygomatic)
    static CHIN = 152;       // bottom of the chin
    static FOREHEAD = 10;    // top of the forehead

    // Cached last valid gaze sample. Used both as the blink-suspension
    // fallback and as the bail-out value when landmarks are missing.
    // Private field (#) — not exposed on the public API.
    #lastValidGaze = { x: 0, y: 0 };

    /**
     * Compute the head-pose-corrected gaze vector for the current frame.
     *
     * @param {Array<{x:number,y:number,z:number}>} landmarks
     *        The 478 facial landmarks returned by MediaPipe FaceLandmarker,
     *        in normalised image coordinates ([0, 1] on both axes).
     * @returns {{x:number,y:number}} Gaze vector in normalised camera space,
     *        clamped to [-0.5, +0.5] on each axis. Returns the last valid
     *        sample on bad input, on a blink, or on numerically degenerate
     *        geometry.
     */
    getRobustGazeVector(landmarks) {
        // Defensive guard: MediaPipe must return the full 478-landmark mesh,
        // otherwise iris indices (469-477) would be out of bounds
        if (!landmarks || landmarks.length < 478) return this.#lastValidGaze;

        // ── Left eye geometry ────────────────────────────────────────────
        const lOuter = landmarks[GazeEstimator.L_OUTER];
        const lInner = landmarks[GazeEstimator.L_INNER];
        const lTop = landmarks[GazeEstimator.L_TOP];
        const lBottom = landmarks[GazeEstimator.L_BOTTOM];

        // Euclidean width/height of the eye and centre of its bounding box.
        // Math.hypot is used instead of sqrt(a² + b²) for numerical robustness.
        const lWidth = Math.hypot(lInner.x - lOuter.x, lInner.y - lOuter.y);
        const lHeight = Math.hypot(lBottom.x - lTop.x, lBottom.y - lTop.y);
        const lCenterX = (lOuter.x + lInner.x) / 2;
        const lCenterY = (lTop.y + lBottom.y) / 2;

        // ── Right eye geometry (mirrored layout of inner/outer corners) ──
        const rInner = landmarks[GazeEstimator.R_INNER];
        const rOuter = landmarks[GazeEstimator.R_OUTER];
        const rTop = landmarks[GazeEstimator.R_TOP];
        const rBottom = landmarks[GazeEstimator.R_BOTTOM];

        const rWidth = Math.hypot(rOuter.x - rInner.x, rOuter.y - rInner.y);
        const rHeight = Math.hypot(rBottom.x - rTop.x, rBottom.y - rTop.y);
        const rCenterX = (rInner.x + rOuter.x) / 2;
        const rCenterY = (rTop.y + rBottom.y) / 2;

        // Degenerate geometry guard: a zero-width eye would cause a
        // division by zero in the normalisation step below
        if (lWidth < 1e-5 || rWidth < 1e-5) return this.#lastValidGaze;

        // ── Blink suspension (defends against Bell's phenomenon) ────────
        // During a blink the eyeball deviates slightly upwards before the
        // eyelid fully closes; that micro-saccade would propagate as a
        // false vertical jump in the gaze output. We detect blinks via the
        // Eye Aspect Ratio (height/width) and freeze the gaze on the last
        // valid sample whenever EAR drops below 0.15.
        const earLeft = lHeight / lWidth;
        const earRight = rHeight / rWidth;
        const ear = (earLeft + earRight) / 2;

        // Eye too closed → trust the cached value instead of recomputing
        if (ear < 0.15) {
            return this.#lastValidGaze;
        }

        // ── Binocular iris centroid ──────────────────────────────────────
        // MediaPipe returns 4 landmarks around each iris (469-472 right,
        // 474-477 left, in compass-like N/E/S/W order). Averaging the four
        // points instead of using any single one acts as a spatial low-pass
        // filter: stray corneal reflections affect at most one of the four
        // points, the other three keep the centroid stable.
        const lIris = [474, 475, 476, 477].reduce((s, i) => ({ x: s.x + landmarks[i].x / 4, y: s.y + landmarks[i].y / 4 }), { x: 0, y: 0 });
        const rIris = [469, 470, 471, 472].reduce((s, i) => ({ x: s.x + landmarks[i].x / 4, y: s.y + landmarks[i].y / 4 }), { x: 0, y: 0 });

        // Express each iris centroid as an offset from its eye centre,
        // normalised by the eye dimensions. This makes the signal scale
        // independent of the apparent face size in the image.
        const lGazeX = (lIris.x - lCenterX) / lWidth;
        const lGazeY = (lIris.y - lCenterY) / lHeight;
        const rGazeX = (rIris.x - rCenterX) / rWidth;
        const rGazeY = (rIris.y - rCenterY) / rHeight;

        // Average both eyes to get the raw binocular gaze signal
        const rawX = (lGazeX + rGazeX) / 2;
        const rawY = (lGazeY + rGazeY) / 2;

        // ── Head-pose signals ────────────────────────────────────────────
        // Yaw is estimated from the asymmetry between the distances from
        // the nose tip to each cheekbone: when the head turns to the right,
        // the right cheek "shrinks" in the image and the left one "grows".
        const noseTip = landmarks[GazeEstimator.NOSE_TIP];
        const lCheek = landmarks[GazeEstimator.L_CHEEK];
        const rCheek = landmarks[GazeEstimator.R_CHEEK];
        const lCheekDist = Math.abs(noseTip.x - lCheek.x);
        const rCheekDist = Math.abs(noseTip.x - rCheek.x);
        const faceWidth = lCheekDist + rCheekDist;

        // yawSignal ∈ [-1, +1]: 0 means perfectly frontal, positive means
        // the head is turned to the right (right cheek closer to the nose)
        const yawSignal = faceWidth > 1e-5 ? (rCheekDist - lCheekDist) / faceWidth : 0;

        // Roll is the angle of the line connecting the two eye centres:
        // 0 rad when the head is upright, positive when tilted clockwise
        const rollSignal = Math.atan2(rCenterY - lCenterY, rCenterX - lCenterX);

        // Empirically tuned gains: how strongly yaw/roll bias the raw gaze
        const YAW_GAIN = 0.65;
        const ROLL_GAIN = 0.20;

        // ── Non-linear head-pose correction ──────────────────────────────
        // Yaw is corrected with a sinusoidal term so that the compensation
        // saturates near ±90°, where the iris would no longer be visible
        // anyway. Roll uses a linear correction on Y.
        const correctedX = rawX - (Math.sin(yawSignal * Math.PI) * YAW_GAIN);
        const correctedY = rawY - (rollSignal * ROLL_GAIN);

        // ── Pitch estimation (head nodding up/down) ─────────────────────
        // Pitch is inferred from the vertical position of the nose base
        // relative to the forehead-to-chin axis. A small face-height tweak
        // moves the nose base up or down inside that range as the head pitches.
        const chin = landmarks[GazeEstimator.CHIN];
        const forehead = landmarks[GazeEstimator.FOREHEAD];
        const nosePitchY = landmarks[GazeEstimator.NOSE_BASE];

        const faceHeightY = Math.abs(chin.y - forehead.y);
        // noseRelY ∈ [0, 1]: 0 means nose at the forehead, 1 means at the chin
        const noseRelY = faceHeightY > 1e-5 ? (nosePitchY.y - forehead.y) / faceHeightY : 0.5;

        // 0.48 is the empirical "neutral" position of the nose base in a
        // frontal pose. Deviations from it indicate pitch and are scaled
        // by the gain before being subtracted from the Y gaze coordinate.
        const PITCH_CENTER = 0.48;
        const PITCH_GAIN = 0.30;
        const pitchOffset = (noseRelY - PITCH_CENTER) * PITCH_GAIN;

        const finalX = correctedX;
        const finalY = correctedY - pitchOffset;

        // ── Output clamping ──────────────────────────────────────────────
        // The gaze vector is bounded to [-0.5, +0.5] on each axis to avoid
        // feeding extreme outliers (e.g. caused by a tracking glitch on a
        // single landmark) to the downstream TPS calibrator.
        const MAX_OFFSET = 0.5;
        this.#lastValidGaze = {
            x: Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, finalX)),
            y: Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, finalY))
        };

        return this.#lastValidGaze;
    }
}