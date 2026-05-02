export class AffectAnalyzer {
    constructor() {
        this.isCalibrating = false;
        this.isCalibrated = false;
        this.samples = [];
        this.baseline = {
            corrugator: { mean: 0, std: 1 },
            ear: { mean: 0, std: 1 },
            lipPress: { mean: 0, std: 1 },
            browAsymmetry: { mean: 0, std: 1 },
            iod: 0.20
        };

        this.stressAccumulator = 0.0;
        this.boredomAccumulator = 0.0;
        this.activationThreshold = 100.0;

        this.currentStateFrustrated = false;
        this.currentStateBored = false;

        this.headPoseHistory = [];
        this.lipHistory = [];
    }

    startCalibration() {
        this.isCalibrating = true;
        this.isCalibrated = false;
        this.samples = [];
        this.stressAccumulator = 0.0;
        this.boredomAccumulator = 0.0;
        this.currentStateFrustrated = false;
        this.currentStateBored = false;
    }

    processCalibrationSample(metrics) {
        if (!this.isCalibrating) return false;

        this.samples.push(metrics);

        if (this.samples.length >= 120) {
            this.baseline.iod = this.samples.reduce((s, m) => s + m.iod, 0) / 120;

            const calculateStats = (key, minStd) => {
                const values = this.samples.map(m => m[key]).sort((a, b) => a - b);
                const q1 = values[Math.floor(values.length * 0.25)];
                const q3 = values[Math.floor(values.length * 0.75)];
                const iqr = q3 - q1;

                const filtered = values.filter(v => v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr);
                const n = filtered.length;

                if (n === 0) return { mean: values[Math.floor(values.length / 2)], std: minStd };

                const mean = filtered.reduce((s, v) => s + v, 0) / n;
                const variance = filtered.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / n;
                const std = Math.sqrt(variance);
                return { mean, std: Math.max(std, minStd) };
            };

            this.baseline.corrugator = calculateStats('corrugator', 0.008);
            this.baseline.ear = calculateStats('ear', 0.025);
            this.baseline.lipPress = calculateStats('lipPress', 0.010);
            this.baseline.browAsymmetry = calculateStats('browAsymmetry', 0.008);

            this.isCalibrating = false;
            this.isCalibrated = true;
            return true;
        }
        return false;
    }

    update(metrics, dtSec) {
        if (!this.isCalibrated) return null;

        this.headPoseHistory.push({ x: metrics.noseX, y: metrics.noseY });
        if (this.headPoseHistory.length > 150) this.headPoseHistory.shift();

        this.lipHistory.push(metrics.lipPress);
        if (this.lipHistory.length > 15) this.lipHistory.shift();

        let lipVariance = 0;
        if (this.lipHistory.length > 10) {
            const lipMean = this.lipHistory.reduce((a, b) => a + b, 0) / this.lipHistory.length;
            lipVariance = this.lipHistory.reduce((a, b) => a + Math.pow(b - lipMean, 2), 0) / this.lipHistory.length;
        }
        const isSpeaking = lipVariance > 0.0003;

        let headVariance = 0;
        if (this.headPoseHistory.length > 30) {
            const headMeanX = this.headPoseHistory.reduce((a, b) => a + b.x, 0) / this.headPoseHistory.length;
            const headMeanY = this.headPoseHistory.reduce((a, b) => a + b.y, 0) / this.headPoseHistory.length;
            headVariance = this.headPoseHistory.reduce((a, b) =>
                a + Math.pow(b.x - headMeanX, 2) + Math.pow(b.y - headMeanY, 2), 0
            ) / this.headPoseHistory.length;
        }
        const normalizedHeadVar = headVariance / (metrics.iod * metrics.iod + 1e-6);
        const isFaceStatic = normalizedHeadVar < 0.015;

        const zCorrugator = (this.baseline.corrugator.mean - metrics.corrugator) / this.baseline.corrugator.std;
        const zEar = (this.baseline.ear.mean - metrics.ear) / this.baseline.ear.std;
        const zLip = (this.baseline.lipPress.mean - metrics.lipPress) / this.baseline.lipPress.std;
        const zAsymmetry = (metrics.browAsymmetry - this.baseline.browAsymmetry.mean) / this.baseline.browAsymmetry.std;

        const poseConfidence = metrics.iod && this.baseline.iod > 0
            ? Math.min(1.0, metrics.iod / this.baseline.iod)
            : 1.0;

        // Clamping zEar: evita che valori estremi dominino
        const zEarClamped = Math.min(Math.abs(zEar), 3.0) * Math.sign(zEar);

        // Corrugatore: soglia 1.0 (era 1.2) — più reattivo alle sopracciglia aggrottate
        const actCorr = zCorrugator > 1.0 ? zCorrugator * poseConfidence : 0;

        // EAR: soglia 1.5 (era 1.8) — più reattivo allo strizzare gli occhi
        const actEar = zEarClamped > 1.5 ? zEarClamped : 0;

        // Lip/sbuffo: soglia abbassata 1.8→1.3 e NON bloccato da isSpeaking
        // perché lo sbuffo è un movimento labiale breve e deciso
        const actLip = zLip > 1.3 ? zLip : 0;

        // Pesi: corrugatore dominante, EAR moderato, labbra per sbuffo
        let stressDelta = (actCorr * 2.0) + (actEar * 0.5) + (actLip * 0.8);

        // Confusione: corrugatore > 1.0 + asimmetria > 1.8 (era 2.0)
        const isConfused = (zCorrugator > 1.0) && (zAsymmetry > 1.8);

        let boredomDelta = 0;
        if (isFaceStatic && zEar > 1.2 && zCorrugator < 0.5) {
            boredomDelta = 15.0 * dtSec;
        }

        if (normalizedHeadVar > 0.1) {
            this.boredomAccumulator *= 0.5;
        }

        if (this.stressAccumulator < 20 && boredomDelta === 0 && !isSpeaking && !isConfused) {
            const alpha = 0.1 * dtSec;
            this.baseline.corrugator.mean = (1 - alpha) * this.baseline.corrugator.mean + alpha * metrics.corrugator;
            this.baseline.ear.mean = (1 - alpha) * this.baseline.ear.mean + alpha * metrics.ear;
            this.baseline.lipPress.mean = (1 - alpha) * this.baseline.lipPress.mean + alpha * metrics.lipPress;
        }

        // Incremento 16→18, decadimento rimane 30
        const incrementoStress = stressDelta * 18.0 * dtSec;
        const decadimentoStress = 30.0 * dtSec;
        this.stressAccumulator += (incrementoStress - decadimentoStress);
        this.stressAccumulator = Math.max(0, Math.min(this.stressAccumulator, this.activationThreshold));

        const decadimentoNoia = 10.0 * dtSec;
        this.boredomAccumulator += (boredomDelta - decadimentoNoia);
        this.boredomAccumulator = Math.max(0, Math.min(this.boredomAccumulator, this.activationThreshold));

        // Isteresi: scatta a 80, si spegne sotto 35
        if (this.stressAccumulator >= 80) {
            this.currentStateFrustrated = true;
        } else if (this.stressAccumulator < 35) {
            this.currentStateFrustrated = false;
        }

        if (this.boredomAccumulator >= 80) {
            this.currentStateBored = true;
        } else if (this.boredomAccumulator < 40) {
            this.currentStateBored = false;
        }

        return {
            zCorrugator,
            zEar,
            zLip,
            isFrustrated: this.currentStateFrustrated,
            isBored: this.currentStateBored,
            isConfused: isConfused,
            stressPercentage: this.stressAccumulator,
            boredomPercentage: this.boredomAccumulator,
            isSpeaking: isSpeaking
        };
    }
}