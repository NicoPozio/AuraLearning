/**
 * GazeEstimator — turns raw MediaPipe facial landmarks into a stable,
 * head-pose-corrected gaze vector expressed in normalised camera space.
 *
 * The output is NOT yet a pixel on the screen: it is the gaze direction
 * in the webcam frame, in [-0.5, +0.5] on both axes. Mapping to actual
 * screen pixels is the responsibility of GazeCalibrator (TPS).
 *
 * Per-frame flow:
 *   1. Binocular eye geometry (centre + width + height).
 *   2. Bail out on a blink (EAR < 0.15), returning the last valid sample
 *      to avoid Bell's-phenomenon spikes.
 *   3. Binocular iris centroid: average the 4 iris landmarks of each eye,
 *      express it relative to each eye's centre, normalise by eye size.
 *   4. Non-linear head-pose compensation (yaw, roll, pitch).
 *   5. Clamp to [-0.5, +0.5] and remember the result as the new "last
 *      valid sample" for the blink-suspension fallback.
 */
export class GazeEstimator {

    // Left eye: outer corner (temple side), inner corner (nose side),
    // top and bottom eyelid landmarks.
    static L_OUTER  = 33;
    static L_INNER  = 133;
    static L_TOP    = 159;
    static L_BOTTOM = 145;

    // Right eye: mirrored layout of inner/outer corners.
    static R_INNER  = 362;
    static R_OUTER  = 263;
    static R_TOP    = 386;
    static R_BOTTOM = 374;

    // Facial reference points for head-pose estimation.
    static NOSE_TIP  = 1;
    static NOSE_BASE = 168;
    static L_CHEEK   = 234;
    static R_CHEEK   = 454;
    static CHIN      = 152;
    static FOREHEAD  = 10;

    // Cached last valid gaze sample. Used as the blink-suspension
    // fallback and as the bail-out value when landmarks are missing.
    #lastValidGaze = { x: 0, y: 0 };

    /**
     * Compute the head-pose-corrected gaze vector for the current frame.
     *
     * @param {Array<{x:number,y:number,z:number}>} landmarks
     *        478 facial landmarks from MediaPipe FaceLandmarker, in
     *        normalised image coordinates ([0, 1] on each axis).
     * @returns {{x:number,y:number}} Gaze vector in normalised camera
     *        space, clamped to [-0.5, +0.5] per axis. Returns the last
     *        valid sample on bad input, on a blink, or on numerically
     *        degenerate geometry.
     */
    getRobustGazeVector(landmarks) {
        // Defensive guard: iris indices 469–477 require the full mesh.
        if (!landmarks || landmarks.length < 478) return this.#lastValidGaze;

        // Left eye geometry. Math.hypot is preferred over sqrt(a²+b²)
        // for numerical robustness.
        const lOuter = landmarks[GazeEstimator.L_OUTER];
        const lInner = landmarks[GazeEstimator.L_INNER];
        const lTop = landmarks[GazeEstimator.L_TOP];
        const lBottom = landmarks[GazeEstimator.L_BOTTOM];
        const lWidth = Math.hypot(lInner.x - lOuter.x, lInner.y - lOuter.y);
        const lHeight = Math.hypot(lBottom.x - lTop.x, lBottom.y - lTop.y);
        const lCenterX = (lOuter.x + lInner.x) / 2;
        const lCenterY = (lTop.y + lBottom.y) / 2;

        // Right eye geometry.
        const rInner = landmarks[GazeEstimator.R_INNER];
        const rOuter = landmarks[GazeEstimator.R_OUTER];
        const rTop = landmarks[GazeEstimator.R_TOP];
        const rBottom = landmarks[GazeEstimator.R_BOTTOM];
        const rWidth = Math.hypot(rOuter.x - rInner.x, rOuter.y - rInner.y);
        const rHeight = Math.hypot(rBottom.x - rTop.x, rBottom.y - rTop.y);
        const rCenterX = (rInner.x + rOuter.x) / 2;
        const rCenterY = (rTop.y + rBottom.y) / 2;

        // Degenerate geometry guard: a zero-width eye would divide by
        // zero in the normalisation step below.
        if (lWidth < 1e-5 || rWidth < 1e-5) return this.#lastValidGaze;

        // Blink suspension (defends against Bell's phenomenon). During
        // a blink the eyeball deviates slightly upwards before the
        // eyelid fully closes; that micro-saccade would otherwise
        // propagate as a false vertical jump. EAR (height/width) drops
        // sharply on a blink — below 0.15 we trust the cached sample.
        const ear = ((lHeight / lWidth) + (rHeight / rWidth)) / 2;
        if (ear < 0.15) return this.#lastValidGaze;

        // Binocular iris centroid. MediaPipe returns 4 landmarks around
        // each iris (469–472 right, 474–477 left, in N/E/S/W compass
        // order). Averaging the four points instead of using any single
        // one acts as a spatial low-pass filter: a stray corneal
        // reflection affects at most one of the four landmarks, the
        // other three keep the centroid stable.
        const lIris = [474, 475, 476, 477].reduce(
            (s, i) => ({ x: s.x + landmarks[i].x / 4, y: s.y + landmarks[i].y / 4 }),
            { x: 0, y: 0 });
        const rIris = [469, 470, 471, 472].reduce(
            (s, i) => ({ x: s.x + landmarks[i].x / 4, y: s.y + landmarks[i].y / 4 }),
            { x: 0, y: 0 });

        // Iris offset from each eye centre, normalised by eye size,
        // averaged across both eyes. Scale-independent of face size.
        const lGazeX = (lIris.x - lCenterX) / lWidth;
        const lGazeY = (lIris.y - lCenterY) / lHeight;
        const rGazeX = (rIris.x - rCenterX) / rWidth;
        const rGazeY = (rIris.y - rCenterY) / rHeight;
        const rawX = (lGazeX + rGazeX) / 2;
        const rawY = (lGazeY + rGazeY) / 2;

        // Yaw signal. Asymmetry between nose-tip-to-cheekbone distances:
        // when the head turns right, the right cheek "shrinks" in the
        // image and the left one "grows". The signal is in [-1, +1], 0
        // meaning frontal, positive meaning the head is turned right.
        const noseTip = landmarks[GazeEstimator.NOSE_TIP];
        const lCheek = landmarks[GazeEstimator.L_CHEEK];
        const rCheek = landmarks[GazeEstimator.R_CHEEK];
        const lCheekDist = Math.abs(noseTip.x - lCheek.x);
        const rCheekDist = Math.abs(noseTip.x - rCheek.x);
        const faceWidth = lCheekDist + rCheekDist;
        const yawSignal = faceWidth > 1e-5 ? (rCheekDist - lCheekDist) / faceWidth : 0;

        // Roll: angle of the line connecting the two eye centres.
        // 0 rad when upright, positive when tilted clockwise.
        const rollSignal = Math.atan2(rCenterY - lCenterY, rCenterX - lCenterX);

        // Empirically tuned gains. Yaw uses a sinusoidal correction so
        // the compensation saturates near ±90°, where the iris would no
        // longer be visible anyway. Roll uses a linear correction on Y.
        const YAW_GAIN = 0.65;
        const ROLL_GAIN = 0.20;
        const correctedX = rawX - (Math.sin(yawSignal * Math.PI) * YAW_GAIN);
        const correctedY = rawY - (rollSignal * ROLL_GAIN);

        // Pitch: inferred from the vertical position of the nose base
        // relative to the forehead-to-chin axis. 0.48 is the empirical
        // "neutral" position in a frontal pose; deviations indicate
        // pitch and are scaled by PITCH_GAIN before being subtracted
        // from the Y gaze coordinate.
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

        // Clamp to [-0.5, +0.5]. Extreme outliers from single-landmark
        // tracking glitches never reach the downstream TPS calibrator.
        const MAX_OFFSET = 0.5;
        this.#lastValidGaze = {
            x: Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, finalX)),
            y: Math.max(-MAX_OFFSET, Math.min(MAX_OFFSET, finalY))
        };
        return this.#lastValidGaze;
    }
}