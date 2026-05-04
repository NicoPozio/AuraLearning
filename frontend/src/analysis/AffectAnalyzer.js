export class AffectAnalyzer {
    constructor(loggerCallback) {
        this.log = loggerCallback || (() => {});
        this.isCalibrating = false;
        this.isCalibrated  = false;
        this.samples  = [];
        this.baseline = {
            corrugator:     { mean: 0, std: 1 },
            ear:            { mean: 0, std: 1 },
            lipPress:       { mean: 0, std: 1 },
            browAsymmetry:  { mean: 0, std: 1 },
            smile:          { mean: 0, std: 1 },
            innerBrowRaise: { mean: 0, std: 1 },
            noseWrinkle:    { mean: 0, std: 1 },
            iod: 0.20
        };

        this.stressAccumulator  = 0.0;
        this.boredomAccumulator = 0.0;
        this.activationThreshold = 100.0;

        this.currentStateFrustrated = false;
        this.currentStateBored      = false;
        this.lastReportedState      = "";

        this.headPoseHistory = [];
        this.lipHistory      = [];

        this._SMOOTH_ALPHA   = 0.25;
        this.smoothedMetrics = null;

        this._STRESS_DEBOUNCE_FRAMES = 4;
        this.stressSignalFrames      = 0;

        this._NEUTRAL_FRAMES_REQUIRED = 90;
        this.neutralFramesCount       = 0;

        // Blink rate tracking (finestra 10s)
        this._BLINK_WINDOW_MS  = 10000;
        this._BLINK_EAR_THRESH = 0.18;
        this._inBlink          = false;
        this.blinkTimestamps   = [];
        this.blinkRate         = 0;

        // Head roll history per confusione
        this.headRollHistory = [];
    }

    startCalibration() {
        this.isCalibrating = true;
        this.isCalibrated  = false;
        this.samples       = [];
        this.stressAccumulator  = 0.0;
        this.boredomAccumulator = 0.0;
        this.currentStateFrustrated = false;
        this.currentStateBored      = false;
        this.lastReportedState      = "";
        this.smoothedMetrics        = null;
        this.stressSignalFrames     = 0;
        this.neutralFramesCount     = 0;
        this.blinkTimestamps        = [];
        this._inBlink               = false;
        this.headRollHistory        = [];
    }

    processCalibrationSample(metrics) {
        if (!this.isCalibrating) return false;
        this.samples.push(metrics);

        if (this.samples.length >= 120) {
            this.baseline.iod = this.samples.reduce((s, m) => s + m.iod, 0) / 120;

            const calculateStats = (key, minStd) => {
                const values   = this.samples.map(m => m[key]).sort((a, b) => a - b);
                const q1       = values[Math.floor(values.length * 0.25)];
                const q3       = values[Math.floor(values.length * 0.75)];
                const iqr      = q3 - q1;
                const filtered = values.filter(v => v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr);
                const n        = filtered.length || 1;
                const mean     = filtered.reduce((s, v) => s + v, 0) / n;
                const variance = filtered.reduce((s, v) => s + (v - mean) ** 2, 0) / n;
                return { mean, std: Math.max(Math.sqrt(variance), minStd) };
            };

            this.baseline.corrugator     = calculateStats('corrugator',     0.008);
            this.baseline.ear            = calculateStats('ear',            0.020);
            this.baseline.lipPress       = calculateStats('lipPress',       0.010);
            this.baseline.browAsymmetry  = calculateStats('browAsymmetry',  0.008);
            this.baseline.smile          = calculateStats('smileIntensity', 0.010);
            this.baseline.innerBrowRaise = calculateStats('innerBrowRaise', 0.010);
            this.baseline.noseWrinkle    = calculateStats('noseWrinkle',    0.008);

            this.isCalibrating = false;
            this.isCalibrated  = true;
            return true;
        }
        return false;
    }

    update(metrics, dtSec) {
        if (!this.isCalibrated) return null;

        const nowMs = performance.now();

        // EMA smoothing
        if (this.smoothedMetrics === null) {
            this.smoothedMetrics = {
                corrugator: metrics.corrugator, ear: metrics.ear,
                lipPress: metrics.lipPress, browAsymmetry: metrics.browAsymmetry,
                smileIntensity: metrics.smileIntensity,
                innerBrowRaise: metrics.innerBrowRaise,
                noseWrinkle: metrics.noseWrinkle,
            };
        } else {
            const a = this._SMOOTH_ALPHA;
            for (const k of ['corrugator','ear','lipPress','browAsymmetry','smileIntensity','innerBrowRaise','noseWrinkle']) {
                this.smoothedMetrics[k] = a * metrics[k] + (1 - a) * this.smoothedMetrics[k];
            }
        }
        const sm = this.smoothedMetrics;

        // Blink rate
        const isEyeClosed = sm.ear < this._BLINK_EAR_THRESH;
        if (isEyeClosed && !this._inBlink) {
            this._inBlink = true;
            this.blinkTimestamps.push(nowMs);
        } else if (!isEyeClosed) {
            this._inBlink = false;
        }
        const windowStart    = nowMs - this._BLINK_WINDOW_MS;
        this.blinkTimestamps = this.blinkTimestamps.filter(t => t > windowStart);
        this.blinkRate       = this.blinkTimestamps.length / (this._BLINK_WINDOW_MS / 1000);

        // Head roll history
        this.headRollHistory.push(metrics.headRoll);
        if (this.headRollHistory.length > 45) this.headRollHistory.shift();
        let headRollVariance = 0;
        if (this.headRollHistory.length > 10) {
            const meanRoll   = this.headRollHistory.reduce((a, b) => a + b, 0) / this.headRollHistory.length;
            headRollVariance = this.headRollHistory.reduce((a, b) => a + (b - meanRoll) ** 2, 0) / this.headRollHistory.length;
        }

        // Head / lip movement
        this.headPoseHistory.push({ x: metrics.noseX, y: metrics.noseY });
        if (this.headPoseHistory.length > 150) this.headPoseHistory.shift();

        this.lipHistory.push(sm.lipPress);
        if (this.lipHistory.length > 15) this.lipHistory.shift();

        let lipVariance = 0;
        if (this.lipHistory.length > 10) {
            const lipMean = this.lipHistory.reduce((a, b) => a + b, 0) / this.lipHistory.length;
            lipVariance   = this.lipHistory.reduce((a, b) => a + (b - lipMean) ** 2, 0) / this.lipHistory.length;
        }
        const isSpeaking = lipVariance > 0.0003;

        let headVariance = 0;
        if (this.headPoseHistory.length > 30) {
            const hMX    = this.headPoseHistory.reduce((a, b) => a + b.x, 0) / this.headPoseHistory.length;
            const hMY    = this.headPoseHistory.reduce((a, b) => a + b.y, 0) / this.headPoseHistory.length;
            headVariance = this.headPoseHistory.reduce((a, b) => a + (b.x - hMX) ** 2 + (b.y - hMY) ** 2, 0) / this.headPoseHistory.length;
        }
        const normalizedHeadVar = headVariance / (metrics.iod * metrics.iod + 1e-6);
        const isFaceStatic      = normalizedHeadVar < 0.015;

        // Z-scores
        const zCorrugator     = (this.baseline.corrugator.mean     - sm.corrugator)     / this.baseline.corrugator.std;
        const zEar            = (this.baseline.ear.mean            - sm.ear)            / this.baseline.ear.std;
        const zLip            = (this.baseline.lipPress.mean       - sm.lipPress)       / this.baseline.lipPress.std;
        const zAsymmetry      = (sm.browAsymmetry  - this.baseline.browAsymmetry.mean)  / this.baseline.browAsymmetry.std;
        const zSmile          = (sm.smileIntensity - this.baseline.smile.mean)          / this.baseline.smile.std;
        const zInnerBrowRaise = (sm.innerBrowRaise - this.baseline.innerBrowRaise.mean) / this.baseline.innerBrowRaise.std;
        const zNoseWrinkle    = (sm.noseWrinkle    - this.baseline.noseWrinkle.mean)    / this.baseline.noseWrinkle.std;

        const iodRatio       = metrics.iod && this.baseline.iod > 0 ? metrics.iod / this.baseline.iod : 1.0;
        const poseConfidence = Math.max(0, 1.0 - Math.abs(1.0 - iodRatio) * 1.5);
        const zEarClamped    = Math.min(Math.abs(zEar), 3.0) * Math.sign(zEar);

        // Componenti stress
        const actCorr        = zCorrugator     > 1.2 ? zCorrugator     * poseConfidence : 0;
        const actEar         = zEarClamped     > 1.5 ? zEarClamped     : 0;
        const actLip         = (zLip > 1.8 && !isSpeaking) ? zLip : 0;
        const actAU1         = zInnerBrowRaise > 1.5 ? zInnerBrowRaise * 0.5 : 0;  // preoccupazione
        const actNoseWrinkle = zNoseWrinkle    > 1.5 ? zNoseWrinkle    * 0.6 : 0;  // avversione

        let stressDelta = (actCorr * 2.0) + (actEar * 0.5) + (actLip * 0.8) + actAU1 + actNoseWrinkle;

        // Confusione — sistema a punteggio (≥2 segnali su 5)
        const confScore1 = zCorrugator     > 0.8  ? 1 : 0;  // sopracciglio aggrottato
        const confScore2 = zAsymmetry      > 1.5  ? 1 : 0;  // asimmetria sopracciglia
        const confScore3 = Math.abs(metrics.headRoll) > 0.07 ? 1 : 0;  // testa inclinata ~4°
        const confScore4 = zInnerBrowRaise > 1.2  ? 1 : 0;  // sopracciglio mediale alzato
        const confScore5 = headRollVariance > 0.003 && zCorrugator > 0.5 ? 1 : 0; // scuotimento testa
        const confusionScore = confScore1 + confScore2 + confScore3 + confScore4 + confScore5;
        const isConfused     = confusionScore >= 2;

        // Noia
        let boredomDelta = 0;
        if (isFaceStatic && zEar > 1.2 && zCorrugator < 0.5) boredomDelta = 15.0 * dtSec;
        if (this.blinkTimestamps.length >= 3 && this.blinkRate < 0.08 && stressDelta < 0.5)
            boredomDelta = Math.max(boredomDelta, 10.0 * dtSec);
        if (this.blinkRate > 0.6 && isFaceStatic)
            boredomDelta = Math.max(boredomDelta, 8.0 * dtSec);

        if (normalizedHeadVar > 0.1) this.boredomAccumulator *= 0.5;

        // Debounce stress
        if (stressDelta > 0) this.stressSignalFrames++;
        else                  this.stressSignalFrames = 0;
        const effectiveStressDelta = this.stressSignalFrames >= this._STRESS_DEBOUNCE_FRAMES
            ? stressDelta : 0;

        // Baseline adattiva con guardiano
        const isNeutralNow = this.stressAccumulator < 15 && boredomDelta === 0
            && !isSpeaking && !isConfused && stressDelta === 0;
        if (isNeutralNow) this.neutralFramesCount++;
        else              this.neutralFramesCount = 0;

        if (this.neutralFramesCount >= this._NEUTRAL_FRAMES_REQUIRED) {
            const alpha = 0.005 * dtSec;
            for (const [key, smKey] of [
                ['corrugator','corrugator'], ['ear','ear'], ['lipPress','lipPress'],
                ['innerBrowRaise','innerBrowRaise'], ['noseWrinkle','noseWrinkle']
            ]) {
                this.baseline[key].mean = (1 - alpha) * this.baseline[key].mean + alpha * sm[smKey];
            }
        }

        // Accumulo stress
        if (zSmile > 2.0) {
            if (this.stressAccumulator > 10) this.log(`Eureka (Z-Smile: ${zSmile.toFixed(1)})`, "INFO");
            this.stressAccumulator = Math.max(0, this.stressAccumulator - 40.0 * dtSec);
        } else {
            this.stressAccumulator += effectiveStressDelta * 18.0 * dtSec - 30.0 * dtSec;
            this.stressAccumulator  = Math.max(0, Math.min(this.stressAccumulator, this.activationThreshold));
        }

        this.boredomAccumulator += boredomDelta - 10.0 * dtSec;
        this.boredomAccumulator  = Math.max(0, Math.min(this.boredomAccumulator, this.activationThreshold));

        // Transizioni
        if (isConfused && this.lastReportedState !== "CONFUSED") {
            this.log(`Confusione score=${confusionScore}/5 (AU4=${confScore1} asym=${confScore2} roll=${confScore3} AU1=${confScore4} rollVar=${confScore5})`, "ALERT");
            this.lastReportedState = "CONFUSED";
        }
        if (this.stressAccumulator >= 80 && !this.currentStateFrustrated) {
            this.log("FRUSTRAZIONE confermata (>80).", "ALERT");
            this.currentStateFrustrated = true;
            this.lastReportedState = "FRUSTRATED";
        } else if (this.stressAccumulator < 35 && this.currentStateFrustrated) {
            this.log("Recupero: stress < 35.", "INFO");
            this.currentStateFrustrated = false;
            this.lastReportedState = "NORMAL";
        }
        if (this.boredomAccumulator >= 80)     this.currentStateBored = true;
        else if (this.boredomAccumulator < 40) this.currentStateBored = false;

        return {
            zCorrugator, zEar, zLip,
            zInnerBrowRaise, zNoseWrinkle,
            confusionScore, blinkRate: this.blinkRate,
            isFrustrated: this.currentStateFrustrated,
            isBored:      this.currentStateBored,
            isConfused,
            stressPercentage:  this.stressAccumulator,
            boredomPercentage: this.boredomAccumulator,
            isSpeaking
        };
    }
}