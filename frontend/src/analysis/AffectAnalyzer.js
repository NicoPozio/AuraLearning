/**
 * AffectAnalyzer — Micro-expression detector.
 *
 * ARCHITECTURE (simpler than the previous accumulator-based version):
 *
 * Each micro-expression (ME) carries its own independent state:
 *   • zThresh      — z-score threshold relative to the personal baseline
 *   • durationMs   — minimum sustained time above zThresh to "activate"
 *   • deactivateMs — minimum sustained time below zThresh to "deactivate"
 *
 * The student is classified as "In difficoltà" (struggling) as soon as
 * AT LEAST ONE ME is active. Recovery to "Normale" requires RECOVERY_MS
 * milliseconds with every ME below threshold — explicit timer, no
 * stress accumulator.
 *
 * MICRO-EXPRESSIONS MONITORED (all features come from FaceMetricsExtractor):
 *   1. browFurrow   — AU4, brow furrow                          → ≥ 1.8 s
 *   2. eyeSquint    — AU43/46, low EAR (squint or fatigue)       → ≥ 1.4 s
 *   3. lipPress     — AU24, lip compression                      → ≥ 1.5 s
 *   4. mouthOpen    — open jaw (surprise / confusion)            → ≥ 1.5 s
 *   5. noseWrinkle  — AU9, nose wrinkle (disgust)                → ≥ 1.5 s
 *   6. mouthFrown   — AU15, mouth-corner depression              → ≥ 1.5 s
 *   7. browRaise    — AU1, inner brow raise (perplexity)         → ≥ 1.5 s
 *   8. faceAbsent   — face not detected / covered (hands etc.)   → ≥ 1.5 s
 *
 * GAZE-AWAY (frequency-based, fed from outside via notifyGazeAway):
 *   Activated when the gaze leaves the PDF GAZE_AWAY_COUNT times within
 *   GAZE_AWAY_WINDOW_MS — count, not duration.
 *
 * Baseline: 120 frames (~4 s) of neutral expression, persisted in
 * sessionStorage. Adaptively updated at a very slow rate (~0.002·dt per
 * second) only when neutrality is "certain", to track slow drift in the
 * user's resting expression without contaminating the baseline with
 * negative-emotion samples.
 */
export class AffectAnalyzer {

    /**
     * Build the analyser. The logger is optional; when omitted the class
     * is completely silent, which keeps production runs free of noise.
     *
     * @param {(msg:string, level?:string)=>void} [loggerCallback]
     */
    constructor(loggerCallback) {
        this.log = loggerCallback || (() => { });

        this.isCalibrating = false;
        this.isCalibrated = false;
        this.samples = [];                  // baseline-acquisition buffer
        this.baseline = this._defaultBaseline();

        // ── Micro-expressions configuration table ────────────────────────
        // Per-entry fields:
        //   zThresh      — activation threshold on the z-score
        //   durationMs   — sustained time above zThresh required to ACTIVATE  (slow hysteresis)
        //   deactivateMs — sustained time below zThresh required to DEACTIVATE (fast hysteresis)
        //   _sinceMs     — running timer for the activation side  (signal ON)
        //   _offSinceMs  — running timer for the deactivation side (signal OFF after activation)
        //   active       — current latched state of the ME
        //
        // Asymmetric hysteresis is intentional: we want activation to be
        // hard to trigger (avoid false positives) but deactivation to be
        // quick enough to feel responsive once the user calms down.
        this._MEs = {
            browFurrow: { zThresh: 1.0, durationMs: 1800, deactivateMs: 1200, _sinceMs: 0, _offSinceMs: 0, active: false },
            eyeSquint: { zThresh: 1.1, durationMs: 1400, deactivateMs: 1000, _sinceMs: 0, _offSinceMs: 0, active: false },
            lipPress: { zThresh: 1.2, durationMs: 1500, deactivateMs: 1000, _sinceMs: 0, _offSinceMs: 0, active: false },
            mouthOpen: { zThresh: 1.5, durationMs: 1500, deactivateMs: 800, _sinceMs: 0, _offSinceMs: 0, active: false },
            mouthFrown: { zThresh: 1.0, durationMs: 1500, deactivateMs: 1200, _sinceMs: 0, _offSinceMs: 0, active: false },
            noseWrinkle: { zThresh: 1.0, durationMs: 1500, deactivateMs: 800, _sinceMs: 0, _offSinceMs: 0, active: false },
            browRaise: { zThresh: 1.0, durationMs: 1500, deactivateMs: 1000, _sinceMs: 0, _offSinceMs: 0, active: false },
            faceAbsent: { zThresh: 0, durationMs: 1500, deactivateMs: 800, _sinceMs: 0, _offSinceMs: 0, active: false },
        };

        // A spontaneous smile is interpreted as a moment of relief or
        // comprehension and clears every active ME at once. Threshold is
        // set high (z = 2σ) so that ambiguous smiles do not reset the state.
        this._SMILE_RESET_THRESH = 2.0;


        // ── Recovery timer ───────────────────────────────────────────────
        // The student goes back to "Normale" only after this many ms with
        // every ME below threshold — prevents flicker around the boundary.
        this._RECOVERY_MS = 4000;
        this._recoverySince = 0; // timestamp from which we have been continuously neutral

        // ── Global state ─────────────────────────────────────────────────
        this.isInDifficulty = false;
        this.activeExpressions = []; // names of currently active MEs (debug / log / prompt)

        // ── Gaze-away (frequency-based) ──────────────────────────────────
        this._GAZE_AWAY_WINDOW_MS = 30000;
        this._GAZE_AWAY_COUNT = 4;     // N events inside the window → active
        this._gazeAwayTimestamps = [];
        this._gazeAwayActive = false;

        // ── Per-frame EMA smoothing on the raw metrics ──────────────────
        // Tames single-frame outliers from MediaPipe before they reach the
        // z-score computation. α = 0.20 → effective averaging window ≈ 5 frames.
        this._SMOOTH_ALPHA = 0.20;
        this.smoothedMetrics = null;

        // ── Blink rate (logged in CSV, not used by the classifier) ───────
        this._BLINK_THRESH = 0.18;
        this._inBlink = false;
        this.blinkTimestamps = [];
        this.blinkRate = 0;
    }

    /**
     * Identity baseline used before calibration completes. Every metric
     * has mean = 0 and std = 1 so that z-scores effectively pass through
     * the raw values and nothing diverges.
     *
     * @returns {Object} A neutral baseline object.
     * @private
     */
    _defaultBaseline() {
        return {
            corrugator: { mean: 0, std: 1 },
            ear: { mean: 0, std: 1 },
            lipPress: { mean: 0, std: 1 },
            mouthOpen: { mean: 0, std: 1 },
            mouthCurvature: { mean: 0, std: 1 },
            noseWrinkle: { mean: 0, std: 1 },
            innerBrowRaise: { mean: 0, std: 1 },
            smileIntensity: { mean: 0, std: 1 },
            iod: 0.20
        };
    }

    // ── Calibration ──────────────────────────────────────────────────────

    /**
     * Begin the 120-frame neutral-expression acquisition. Resets every
     * ME timer, the gaze-away buffer and the recovery state so previous
     * sessions cannot leak into the new baseline.
     */
    startCalibration() {
        this.isCalibrating = true;
        this.isCalibrated = false;
        this.samples = [];
        this.smoothedMetrics = null;
        this.isInDifficulty = false;
        this.activeExpressions = [];
        this._recoverySince = 0;
        this._gazeAwayTimestamps = [];
        this._gazeAwayActive = false;
        // Clear every ME's timers and latched state
        for (const me of Object.values(this._MEs)) { me._sinceMs = 0; me._offSinceMs = 0; me.active = false; }
    }

    /**
     * Add the current frame's metrics to the baseline buffer. When 120
     * samples have been collected, compute robust per-metric statistics
     * (mean and std over the IQR-filtered samples), persist them in
     * sessionStorage, and switch to the operating regime.
     *
     * @param {Object} metrics - Output of FaceMetricsExtractor.extractRawMetrics.
     * @returns {boolean} true when the baseline has just been finalised.
     */
    processCalibrationSample(metrics) {
        if (!this.isCalibrating) return false;

        this.samples.push(metrics);

        if (this.samples.length >= 120) {
            // IOD baseline = plain mean over the 120 samples (no outliers expected)
            this.baseline.iod = this.samples.reduce((s, m) => s + m.iod, 0) / this.samples.length;

            /**
             * Robust per-metric mean/std with an IQR filter:
             *   • Sort the values
             *   • Discard everything outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]
             *   • Compute mean/std on what remains
             *   • Floor the std at `minStd` to ensure a physiologically
             *     plausible noise level — protects against the case in
             *     which the user sits unnaturally still during the 2 s
             *     baseline window and σ collapses to zero.
             */
            const stat = (key, minStd) => {
                const vals = this.samples.map(m => m[key]).sort((a, b) => a - b);
                const q1 = vals[Math.floor(vals.length * 0.25)];
                const q3 = vals[Math.floor(vals.length * 0.75)];
                const iqr = q3 - q1;
                const filtered = vals.filter(v => v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr);
                const n = filtered.length || 1;
                const mean = filtered.reduce((s, v) => s + v, 0) / n;
                const std = Math.sqrt(filtered.reduce((s, v) => s + (v - mean) ** 2, 0) / n);
                return { mean, std: Math.max(std, minStd) };
            };

            // Per-metric minimum sigma values chosen empirically so that a
            // single-frame muscle twitch maps to roughly a 1σ z-score
            this.baseline.corrugator = stat('corrugator', 0.008);
            this.baseline.ear = stat('ear', 0.020);
            this.baseline.lipPress = stat('lipPress', 0.010);
            this.baseline.mouthOpen = stat('mouthOpen', 0.015);
            this.baseline.mouthCurvature = stat('mouthCurvature', 0.010);
            this.baseline.noseWrinkle = stat('noseWrinkle', 0.008);
            this.baseline.innerBrowRaise = stat('innerBrowRaise', 0.010);
            this.baseline.smileIntensity = stat('smileIntensity', 0.010);

            this.isCalibrating = false;
            this.isCalibrated = true;
            // Persist the baseline so a page refresh doesn't force the user to recalibrate
            try {
                sessionStorage.setItem('aura_baseline', JSON.stringify(this.baseline));
                this.log('Baseline salvata in sessionStorage.', 'INFO');
            } catch (e) {
                this.log('sessionStorage non disponibile.', 'WARN');
            }

            return true;
        }
        return false;
    }

    /**
     * Calibration progress as a value in [0, 1], suitable for binding to
     * a progress bar. Kept around even though main.js currently does not
     * render it directly.
     */
    get calibrationProgress() {
        return Math.min(this._calValidCount / this._CAL_MIN_SAMPLES, 1.0);
    }

    /**
     * Try to restore a baseline saved by a previous session. Missing keys
     * (older versions of the schema) are filled with the default identity
     * baseline instead of being left undefined, which would crash the
     * z-score computation.
     *
     * @returns {boolean} true on successful restoration.
     */
    loadBaselineFromStorage() {
        try {
            const saved = sessionStorage.getItem('aura_baseline');
            if (!saved) return false;
            const parsed = JSON.parse(saved);
            // Merge with defaults: any missing key (e.g. from an older
            // version) falls back to { mean:0, std:1 } instead of `undefined`
            this.baseline = { ...this._defaultBaseline(), ...parsed };
            this.isCalibrated = true;
            this.log('Baseline caricata da sessionStorage.', 'INFO');
            return true;
        } catch (e) {
            return false;
        }
    }

    // ── Gaze-away notification from main.js ──────────────────────────────

    /**
     * Register one "gaze left the PDF" event. Keeps a rolling window of
     * timestamps and activates the gazeAway signal when their count
     * reaches GAZE_AWAY_COUNT inside the window. Called by main.js
     * whenever the live gaze context goes stale.
     */
    notifyGazeAway() {
        const now = performance.now();
        this._gazeAwayTimestamps.push(now);
        // Drop events that fell outside the current window
        const cutoff = now - this._GAZE_AWAY_WINDOW_MS;
        this._gazeAwayTimestamps = this._gazeAwayTimestamps.filter(t => t > cutoff);
        const wasActive = this._gazeAwayActive;
        this._gazeAwayActive = this._gazeAwayTimestamps.length >= this._GAZE_AWAY_COUNT;
        // Log only the rising edge to keep the trace clean
        if (this._gazeAwayActive && !wasActive) {
            this.log(`Gaze-away: ${this._gazeAwayTimestamps.length} volte in ${(this._GAZE_AWAY_WINDOW_MS / 1000)}s`, 'ALERT');
        }
    }

    // ── Update ────────────────────────────────────────────────────────────

    /**
     * Per-frame update. Smooths the metrics, computes z-scores against
     * the personal baseline, runs each ME's hysteresis timer, updates
     * the overall "in difficulty" flag and the adaptive baseline.
     *
     * @param {Object} metrics - Output of FaceMetricsExtractor.extractRawMetrics.
     * @param {number} dtSec   - Seconds elapsed since the previous update.
     * @returns {Object|null} Affective state for this frame, or null if
     *                        the analyser is not calibrated yet.
     */
    update(metrics, dtSec) {
        if (!this.isCalibrated) return null;

        const nowMs = performance.now();

        // ── EMA smoothing of the raw metrics ─────────────────────────────
        // First call seeds the EMA with the current sample so we don't
        // bias the first few frames towards zero
        if (!this.smoothedMetrics) {
            this.smoothedMetrics = { ...metrics };
        } else {
            const a = this._SMOOTH_ALPHA;
            for (const k of ['corrugator', 'ear', 'lipPress', 'mouthOpen', 'mouthCurvature', 'noseWrinkle', 'innerBrowRaise', 'smileIntensity']) {
                if (metrics[k] !== undefined)
                    this.smoothedMetrics[k] = a * metrics[k] + (1 - a) * this.smoothedMetrics[k];
            }
        }
        const sm = this.smoothedMetrics;

        // ── Blink rate (counted via EAR threshold crossings) ────────────
        const eyeClosed = sm.ear < this._BLINK_THRESH;
        if (eyeClosed && !this._inBlink) {
            // Rising edge: start of a new blink
            this._inBlink = true;
            this.blinkTimestamps.push(nowMs);
        } else if (!eyeClosed) { this._inBlink = false; }
        // Keep only the last 10 seconds of blinks → rate is blinks per second
        this.blinkTimestamps = this.blinkTimestamps.filter(t => t > nowMs - 10000);
        this.blinkRate = this.blinkTimestamps.length / 10;

        // ── Z-scores against the personal baseline ──────────────────────
        // Closure with a guard: missing baseline keys (version mismatch)
        // collapse to 0 instead of NaN/Infinity
        const z = (key, smKey) => {
            const b = this.baseline[key];
            if (!b) return 0;
            const val = sm[smKey ?? key];
            if (val === undefined) return 0;
            return (val - b.mean) / (b.std || 1);
        };
        // Sign conventions:
        //   • Negate when the metric DECREASES on activation (e.g. an
        //     angry brow reduces the corrugator distance), so a positive
        //     z-score always means "muscle activated".
        const zCorrugator = -z('corrugator');       // negated: baseline > sample = brow furrowed
        const zEar = -z('ear');              // negated: baseline > sample = eye squinted
        const zLipPress = -z('lipPress');         // negated: baseline > sample = lips pressed
        const zMouthOpen = z('mouthOpen');        // positive: mouth more open than baseline
        const zNoseWrinkle = z('noseWrinkle');      // positive: nose more wrinkled
        const zBrowRaise = z('innerBrowRaise');   // positive: inner brow raised
        const zSmile = z('smileIntensity');
        const zMouthCurvature = z('mouthCurvature');  // positive = corners pulled down = frown

        // ── Map z-scores to instantaneous boolean signals ───────────────
        const signals = {
            browFurrow: zCorrugator > this._MEs.browFurrow.zThresh,
            eyeSquint: zEar > this._MEs.eyeSquint.zThresh,
            lipPress: zLipPress > this._MEs.lipPress.zThresh,
            mouthOpen: zMouthOpen > this._MEs.mouthOpen.zThresh,
            noseWrinkle: zNoseWrinkle > this._MEs.noseWrinkle.zThresh,
            browRaise: zBrowRaise > this._MEs.browRaise.zThresh,
            mouthFrown: zMouthCurvature > this._MEs.mouthFrown.zThresh,
            faceAbsent: false, // owned by updateFaceAbsent()
        };

        // ── Smile-driven full reset ─────────────────────────────────────
        // A genuine smile clears every ME timer, the latched states and
        // the gaze-away flag — interpreted as a "the student got it" moment
        if (zSmile > this._SMILE_RESET_THRESH) {
            for (const me of Object.values(this._MEs)) { me._sinceMs = 0; me._offSinceMs = 0; me.active = false; }
            this._gazeAwayActive = false;
            this.log('Sorriso rilevato: reset microespressioni.', 'INFO');
        }

        // ── Per-ME hysteresis update ────────────────────────────────────
        const dtMs = dtSec * 1000;
        const newlyActivated = [];

        for (const [name, me] of Object.entries(this._MEs)) {
            if (name === 'faceAbsent') continue; // handled by updateFaceAbsent()

            if (signals[name]) {
                // Signal ON: charge the activation timer, drain the deactivation timer
                me._sinceMs += dtMs;
                me._offSinceMs = 0;
                // Cross the activation threshold → latch ME on
                if (!me.active && me._sinceMs >= me.durationMs) {
                    me.active = true;
                    newlyActivated.push(name);
                    this.log(`ME attivata: ${name} (${(me._sinceMs / 1000).toFixed(1)}s)`, 'ALERT');
                }
            } else {
                // Signal OFF: reset the activation timer
                me._sinceMs = 0;
                if (me.active) {
                    // While latched on, charge the deactivation timer
                    me._offSinceMs += dtMs;
                    if (me._offSinceMs >= me.deactivateMs) {
                        me.active = false;
                        me._offSinceMs = 0;
                        this.log(`ME disattivata: ${name}`, 'INFO');
                    }
                }
            }
        }

        // ── Aggregate active list ────────────────────────────────────────
        this.activeExpressions = Object.entries(this._MEs)
            .filter(([, me]) => me.active)
            .map(([name]) => name);
        // Gaze-away is independent from the per-ME machine but contributes
        // to the same overall "in difficulty" decision
        if (this._gazeAwayActive) this.activeExpressions.push('gazeAway');

        const anyActive = this.activeExpressions.length > 0; // includes mouthFrown and gazeAway

        // ── Recovery / difficulty state machine ─────────────────────────
        if (!anyActive) {
            // Start (or continue) measuring how long we've been fully neutral
            if (this._recoverySince === 0) this._recoverySince = nowMs;
            const recoveredMs = nowMs - this._recoverySince;
            if (recoveredMs >= this._RECOVERY_MS && this.isInDifficulty) {
                this.isInDifficulty = false;
                this.log('Ritorno alla neutralità confermato.', 'INFO');
            }
        } else {
            // Anything active → reset the recovery timer
            this._recoverySince = 0;
            if (!this.isInDifficulty) {
                this.isInDifficulty = true;
                this.log(`In difficoltà: [${this.activeExpressions.join(', ')}]`, 'ALERT');
            }
        }

        // ── Slow adaptive baseline update ───────────────────────────────
        // Only updated when neutrality is CERTAIN: nothing active, not in
        // difficulty, and we've been neutral for at least RECOVERY_MS + 2 s.
        // Prevents negative-emotion samples from contaminating the baseline.
        const isNeutralCertain = !anyActive && !this.isInDifficulty && this._recoverySince > 0
            && (nowMs - this._recoverySince) > this._RECOVERY_MS + 2000;

        if (isNeutralCertain) {
            // α = 0.002·dt → ~500 s half-life: glacial drift tracking
            const alpha = 0.002 * dtSec;
            for (const [key, smKey] of [
                ['corrugator', 'corrugator'], ['ear', 'ear'], ['lipPress', 'lipPress'],
                ['mouthOpen', 'mouthOpen'], ['mouthCurvature', 'mouthCurvature'], ['noseWrinkle', 'noseWrinkle'], ['innerBrowRaise', 'innerBrowRaise']
            ]) {
                const val = sm[smKey];
                if (val !== undefined) {
                    this.baseline[key].mean = (1 - alpha) * this.baseline[key].mean + alpha * val;
                }
            }
        }

        // ── Frame-level affective state returned to main.js ─────────────
        return {
            isInDifficulty: this.isInDifficulty,
            activeExpressions: this.activeExpressions,
            // Raw values exported for CSV telemetry
            zCorrugator,
            zEar,
            zLipPress,
            zMouthOpen,
            zNoseWrinkle,
            zBrowRaise,
            blinkRate: this.blinkRate,
            gazeAwayCount: this._gazeAwayTimestamps.length,
            zMouthCurvature,
            // Instantaneous threshold-crossing booleans (debug sidebar)
            debugSignals: {
                browFurrow: zCorrugator > this._MEs.browFurrow.zThresh,
                eyeSquint: zEar > this._MEs.eyeSquint.zThresh,
                lipPress: zLipPress > this._MEs.lipPress.zThresh,
                mouthOpen: zMouthOpen > this._MEs.mouthOpen.zThresh,
                mouthFrown: zMouthCurvature > this._MEs.mouthFrown.zThresh,
                noseWrinkle: zNoseWrinkle > this._MEs.noseWrinkle.zThresh,
                browRaise: zBrowRaise > this._MEs.browRaise.zThresh,
            },
            // Full set of raw z-scores (calibration / diagnostics)
            rawZ: { zCorrugator, zEar, zLipPress, zMouthOpen, zMouthCurvature, zNoseWrinkle, zBrowRaise },
        };
    }

    /**
     * Update the latched state of the faceAbsent ME. Called from main.js
     * every frame, with `isAbsent = true` when MediaPipe found no face.
     * Uses the same activation / deactivation hysteresis as the other MEs.
     *
     * @param {boolean} isAbsent - true when no face was detected this frame.
     * @param {number}  dtSec    - Seconds elapsed since the previous call.
     */
    updateFaceAbsent(isAbsent, dtSec) {
        const me = this._MEs.faceAbsent;
        const dtMs = dtSec * 1000;
        if (isAbsent) {
            me._sinceMs += dtMs;
            me._offSinceMs = 0;
            if (!me.active && me._sinceMs >= me.durationMs) {
                me.active = true;
                this.log('Faccia non rilevata (mani sul viso?)', 'ALERT');
            }
        } else {
            me._sinceMs = 0;
            if (me.active) {
                me._offSinceMs += dtMs;
                if (me._offSinceMs >= me.deactivateMs) {
                    me.active = false;
                    me._offSinceMs = 0;
                }
            }
        }
    }
}