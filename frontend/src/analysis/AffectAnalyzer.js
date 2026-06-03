/**
 * AffectAnalyzer — Micro-expression detector.
 *
 * Each micro-expression (ME) carries three independent parameters:
 *   zThresh      — z-score threshold relative to the personal baseline
 *   durationMs   — minimum sustained time above threshold to activate
 *   deactivateMs — minimum sustained time below threshold to deactivate
 *
 * The student is "In difficoltà" as soon as any ME is active. Recovery
 * to "Normale" requires every ME below threshold for RECOVERY_MS.
 *
 * Micro-expressions monitored (all from FaceMetricsExtractor):
 *   browFurrow, eyeSquint, lipPress, mouthOpen, noseWrinkle,
 *   mouthFrown, browRaise, faceAbsent.
 *
 * Baseline: 120 frames (~4 s) of neutral expression, persisted in
 * sessionStorage. Adaptively updated at a very slow rate (α = 0.002·dt)
 * only when neutrality is certain, to track slow drift without
 * contaminating it with negative-emotion samples.
 */
export class AffectAnalyzer {

    /**
     * Build the analyser. The logger is optional; without it the class
     * is silent, which keeps production runs free of noise.
     *
     * @param {(msg:string, level?:string)=>void} [loggerCallback]
     */
    constructor(loggerCallback) {
        this.log = loggerCallback || (() => { });

        this.isCalibrating = false;
        this.isCalibrated = false;
        this.samples = [];                  // baseline-acquisition buffer
        this.baseline = this._defaultBaseline();

        // Micro-expressions configuration table.
        // Asymmetric hysteresis is intentional: activation is hard to
        // trigger (avoid false positives) and deactivation is faster
        // (feel responsive when the user calms down).
        this._MEs = {
            browFurrow:  { zThresh: 1.0, durationMs: 1800, deactivateMs: 1200, _sinceMs: 0, _offSinceMs: 0, active: false },
            eyeSquint:   { zThresh: 1.5, durationMs: 1400, deactivateMs: 1000, _sinceMs: 0, _offSinceMs: 0, active: false },
            lipPress:    { zThresh: 1.2, durationMs: 1500, deactivateMs: 1000, _sinceMs: 0, _offSinceMs: 0, active: false },
            mouthOpen:   { zThresh: 1.5, durationMs: 1500, deactivateMs:  800, _sinceMs: 0, _offSinceMs: 0, active: false },
            mouthFrown:  { zThresh: 1.0, durationMs: 1500, deactivateMs: 1200, _sinceMs: 0, _offSinceMs: 0, active: false },
            noseWrinkle: { zThresh: 1.5, durationMs: 1500, deactivateMs:  800, _sinceMs: 0, _offSinceMs: 0, active: false },
            browRaise:   { zThresh: 1.0, durationMs: 1500, deactivateMs: 1000, _sinceMs: 0, _offSinceMs: 0, active: false },
            faceAbsent:  { zThresh: 0,   durationMs: 1500, deactivateMs:  800, _sinceMs: 0, _offSinceMs: 0, active: false },
        };

        // A spontaneous smile is interpreted as relief or comprehension
        // and clears every active ME. Threshold is set high (z = 2σ) so
        // that ambiguous smiles do not reset the state.
        this._SMILE_RESET_THRESH = 2.0;

        // Return to "Normale" requires this many ms with every ME below
        // threshold — prevents flicker around the boundary.
        this._RECOVERY_MS = 4000;
        this._recoverySince = 0;

        this.isInDifficulty = false;
        this.activeExpressions = [];

        // EMA smoothing of raw metrics. α = 0.20 gives an effective window
        // of ~5 frames, enough to tame single-frame outliers from
        // MediaPipe without introducing perceptible lag.
        this._SMOOTH_ALPHA = 0.20;
        this.smoothedMetrics = null;

        // Blink rate is logged in the CSV; the classifier does not use it.
        this._BLINK_THRESH = 0.18;
        this._inBlink = false;
        this.blinkTimestamps = [];
        this.blinkRate = 0;
    }

    /**
     * Identity baseline used before calibration completes. Every metric
     * has mean = 0 and std = 1 so z-scores pass through the raw values.
     */
    _defaultBaseline() {
        return {
            corrugator:     { mean: 0, std: 1 },
            ear:            { mean: 0, std: 1 },
            lipPress:       { mean: 0, std: 1 },
            mouthOpen:      { mean: 0, std: 1 },
            mouthCurvature: { mean: 0, std: 1 },
            noseWrinkle:    { mean: 0, std: 1 },
            innerBrowRaise: { mean: 0, std: 1 },
            smileIntensity: { mean: 0, std: 1 },
            iod: 0.20
        };
    }

    /**
     * Begin the 120-frame neutral-expression acquisition. Resets every
     * ME timer and the recovery state so previous sessions cannot leak
     * into the new baseline.
     */
    startCalibration() {
        this.isCalibrating = true;
        this.isCalibrated = false;
        this.samples = [];
        this.smoothedMetrics = null;
        this.isInDifficulty = false;
        this.activeExpressions = [];
        this._recoverySince = 0;
        for (const me of Object.values(this._MEs)) {
            me._sinceMs = 0;
            me._offSinceMs = 0;
            me.active = false;
        }
    }

    /**
     * Add the current frame's metrics to the baseline buffer. When 120
     * samples have been collected, compute robust per-metric statistics
     * (mean and std on the IQR-filtered samples), persist them in
     * sessionStorage, and switch to the operating regime.
     *
     * @param {Object} metrics - Output of FaceMetricsExtractor.extractRawMetrics.
     * @returns {boolean} true when the baseline has just been finalised.
     */
    processCalibrationSample(metrics) {
        if (!this.isCalibrating) return false;

        this.samples.push(metrics);

        if (this.samples.length >= 120) {
            // IOD baseline is the plain mean over the 120 samples
            this.baseline.iod = this.samples.reduce((s, m) => s + m.iod, 0) / this.samples.length;

            // Robust per-metric mean and std with an IQR filter. The std is
            // floored at minStd so an unnaturally still calibration cannot
            // collapse σ to zero.
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
            // Empirical per-metric floors: a single-frame muscle twitch
            // maps to roughly a 1σ z-score.
            this.baseline.corrugator     = stat('corrugator',     0.020);
            this.baseline.ear            = stat('ear',            0.065);
            this.baseline.lipPress       = stat('lipPress',       0.045);
            this.baseline.mouthOpen      = stat('mouthOpen',      0.030);
            this.baseline.mouthCurvature = stat('mouthCurvature', 0.060);
            this.baseline.noseWrinkle    = stat('noseWrinkle',    0.040);
            this.baseline.innerBrowRaise = stat('innerBrowRaise', 0.060);
            this.baseline.smileIntensity = stat('smileIntensity', 0.040);

            this.isCalibrating = false;
            this.isCalibrated = true;

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
     * Try to restore a baseline saved by a previous session. Missing keys
     * (older schema versions) fall back to the identity baseline instead
     * of being left undefined, which would crash the z-score computation.
     *
     * @returns {boolean} true on successful restoration.
     */
    loadBaselineFromStorage() {
        try {
            const saved = sessionStorage.getItem('aura_baseline');
            if (!saved) return false;
            const parsed = JSON.parse(saved);
            this.baseline = { ...this._defaultBaseline(), ...parsed };
            this.isCalibrated = true;
            this.log('Baseline caricata da sessionStorage.', 'INFO');
            return true;
        } catch (e) {
            return false;
        }
    }

    /**
     * Per-frame update. Smooths the metrics, computes z-scores against
     * the personal baseline, runs each ME's hysteresis timer, updates
     * the overall "in difficulty" flag and the adaptive baseline.
     *
     * @param {Object}  metrics        - Output of FaceMetricsExtractor.extractRawMetrics.
     * @param {number}  dtSec          - Seconds elapsed since the previous update.
     * @param {boolean} [isUserSpeaking=false] - True while the user is talking. When
     *        true the mouth-related signals (lipPress, mouthOpen) are forced
     *        to false: articulating words mechanically opens and closes the
     *        mouth, which would otherwise look like frustration or surprise.
     *        Latched ME states are NOT cleared — only the instantaneous
     *        detection is suppressed, so deactivation timers keep draining.
     * @returns {Object|null} Affective state for this frame, or null if
     *        the analyser is not calibrated yet.
     */
    update(metrics, dtSec, isUserSpeaking = false) {
        if (!this.isCalibrated) return null;

        const nowMs = performance.now();

        // EMA smoothing of raw metrics. The first call seeds the filter
        // with the current sample so the first few frames are not biased
        // towards zero.
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

        // Blink rate via EAR threshold crossings. The last 10 s of blinks
        // are kept so the rate ends up in blinks/second.
        const eyeClosed = sm.ear < this._BLINK_THRESH;
        if (eyeClosed && !this._inBlink) {
            this._inBlink = true;
            this.blinkTimestamps.push(nowMs);
        } else if (!eyeClosed) {
            this._inBlink = false;
        }
        this.blinkTimestamps = this.blinkTimestamps.filter(t => t > nowMs - 10000);
        this.blinkRate = this.blinkTimestamps.length / 10;

        // Z-scores against the personal baseline. Missing keys collapse
        // to 0 instead of NaN. Sign convention: positive z always means
        // "muscle activated", so metrics that decrease on activation
        // (corrugator, ear) are negated.
        const z = (key, smKey) => {
            const b = this.baseline[key];
            if (!b) return 0;
            const val = sm[smKey ?? key];
            if (val === undefined) return 0;
            return (val - b.mean) / (b.std || 1);
        };
        const zCorrugator     = -z('corrugator');       // baseline > sample = brow furrowed
        const zEar            = -z('ear');              // baseline > sample = eye squinted
        const zLipPress       =  z('lipPress');
        const zMouthOpen      =  z('mouthOpen');
        const zNoseWrinkle    =  z('noseWrinkle');
        const zBrowRaise      =  z('innerBrowRaise');
        const zSmile          =  z('smileIntensity');
        const zMouthCurvature =  z('mouthCurvature');   // positive = corners pulled down


        // Instantaneous boolean signals from the z-scores. lipPress and
        // mouthOpen are gated on isUserSpeaking; the others are not
        // affected by speech and stay free to fire.
        const MOUTH_OPEN_GATE_FOR_LIPPRESS = 1.0; // z-score; below mouthOpen.zThresh on purpose
        const mouthIsOpening = zMouthOpen > MOUTH_OPEN_GATE_FOR_LIPPRESS;
        const signals = {
            browFurrow:  zCorrugator     > this._MEs.browFurrow.zThresh,
            eyeSquint:   zEar            > this._MEs.eyeSquint.zThresh,
            lipPress:    isUserSpeaking ? false : (zLipPress  > this._MEs.lipPress.zThresh),
            mouthOpen:   isUserSpeaking ? false : (zMouthOpen > this._MEs.mouthOpen.zThresh),
            noseWrinkle: zNoseWrinkle    > this._MEs.noseWrinkle.zThresh,
            browRaise:   zBrowRaise      > this._MEs.browRaise.zThresh,
            mouthFrown:  zMouthCurvature > this._MEs.mouthFrown.zThresh,
            faceAbsent:  false, // owned by updateFaceAbsent()
        };

        // A genuine smile clears every ME timer and latched state — read
        // as a "the student got it" moment.
        if (zSmile > this._SMILE_RESET_THRESH) {
            for (const me of Object.values(this._MEs)) {
                me._sinceMs = 0;
                me._offSinceMs = 0;
                me.active = false;
            }
            this.log('Sorriso rilevato: reset microespressioni.', 'INFO');
        }

        // Per-ME hysteresis update.
        const dtMs = dtSec * 1000;

        for (const [name, me] of Object.entries(this._MEs)) {
            if (name === 'faceAbsent') continue; // owned by updateFaceAbsent()

            if (signals[name]) {
                // Signal ON: charge the activation timer, drain the
                // deactivation timer. Latch on once the threshold is met.
                me._sinceMs += dtMs;
                me._offSinceMs = 0;
                if (!me.active && me._sinceMs >= me.durationMs) {
                    me.active = true;
                    this.log(`ME attivata: ${name} (${(me._sinceMs / 1000).toFixed(1)}s)`, 'ALERT');
                }
            } else {
                // Signal OFF: reset the activation timer. While latched
                // on, charge the deactivation timer until it expires.
                me._sinceMs = 0;
                if (me.active) {
                    me._offSinceMs += dtMs;
                    if (me._offSinceMs >= me.deactivateMs) {
                        me.active = false;
                        me._offSinceMs = 0;
                        this.log(`ME disattivata: ${name}`, 'INFO');
                    }
                }
            }
        }

        this.activeExpressions = Object.entries(this._MEs)
            .filter(([, me]) => me.active)
            .map(([name]) => name);

        const anyActive = this.activeExpressions.length > 0;

        // Recovery / difficulty state machine. Enter "in difficulty" as
        // soon as any ME is active; leave it only after RECOVERY_MS of
        // sustained neutrality.
        if (!anyActive) {
            if (this._recoverySince === 0) this._recoverySince = nowMs;
            const recoveredMs = nowMs - this._recoverySince;
            if (recoveredMs >= this._RECOVERY_MS && this.isInDifficulty) {
                this.isInDifficulty = false;
                this.log('Ritorno alla neutralità confermato.', 'INFO');
            }
        } else {
            this._recoverySince = 0;
            if (!this.isInDifficulty) {
                this.isInDifficulty = true;
                this.log(`In difficoltà: [${this.activeExpressions.join(', ')}]`, 'ALERT');
            }
        }

        // Slow adaptive baseline update. Only fires when neutrality is
        // certain: nothing active, not in difficulty, and at least
        // RECOVERY_MS + 2 s of continuous calm. Prevents any negative
        // sample from leaking into the baseline.
        const isNeutralCertain = !anyActive
            && !this.isInDifficulty
            && this._recoverySince > 0
            && (nowMs - this._recoverySince) > this._RECOVERY_MS + 2000;

        if (isNeutralCertain) {
            // α = 0.002·dt → half-life ≈ 500 s, glacial drift tracking.
            const alpha = 0.002 * dtSec;
            for (const [key, smKey] of [
                ['corrugator',     'corrugator'],
                ['ear',            'ear'],
                ['lipPress',       'lipPress'],
                ['mouthOpen',      'mouthOpen'],
                ['mouthCurvature', 'mouthCurvature'],
                ['noseWrinkle',    'noseWrinkle'],
                ['innerBrowRaise', 'innerBrowRaise']
            ]) {
                const val = sm[smKey];
                if (val !== undefined) {
                    this.baseline[key].mean = (1 - alpha) * this.baseline[key].mean + alpha * val;
                }
            }
        }

        return {
            isInDifficulty: this.isInDifficulty,
            activeExpressions: this.activeExpressions,
            zCorrugator,
            zEar,
            zLipPress,
            zMouthOpen,
            zNoseWrinkle,
            zBrowRaise,
            zMouthCurvature,
            blinkRate: this.blinkRate,
            // Instantaneous threshold-crossing booleans for the debug
            // sidebar. Speaking-gate suppression of lipPress / mouthOpen
            // is already baked into `signals`, so the UI dots reflect it.
            debugSignals: {
                browFurrow:  signals.browFurrow,
                eyeSquint:   signals.eyeSquint,
                lipPress:    signals.lipPress,
                mouthOpen:   signals.mouthOpen,
                mouthFrown:  signals.mouthFrown,
                noseWrinkle: signals.noseWrinkle,
                browRaise:   signals.browRaise,
            },
            rawZ: { zCorrugator, zEar, zLipPress, zMouthOpen, zMouthCurvature, zNoseWrinkle, zBrowRaise },
        };
    }

    /**
     * Update the latched state of the faceAbsent ME. Called from main.js
     * every frame with isAbsent = true when MediaPipe found no face.
     * Uses the same activation / deactivation hysteresis as the others.
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