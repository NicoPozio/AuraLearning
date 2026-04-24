// Assicurati che ci sia "export" all'inizio della riga
export class FrustrationAnalyzer {
    constructor() {
        this.isCalibrating = false;
        this.isCalibrated = false;
        this.samples = [];
        this.baseline = {
            corrugator: { mean: 0, std: 1 },
            ear: { mean: 0, std: 1 },
            lipPress: { mean: 0, std: 1 },
            iod: 0.20
        };

        this.stressAccumulator = 0.0;
        this.activationThreshold = 100.0;
    }

    startCalibration() {
        this.isCalibrating = true;
        this.isCalibrated = false;
        this.samples = [];

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

            this.baseline.corrugator = calculateStats('corrugator', 0.006);
            this.baseline.ear = calculateStats('ear', 0.020);
            this.baseline.lipPress = calculateStats('lipPress', 0.008);

            this.isCalibrating = false;
            this.isCalibrated = true;
            return true;
        }
        return false;
    }

    update(metrics, dtSec) {
        if (!this.isCalibrated) return null;

        // Formula corretta: (Media - Valore)
        const zCorrugator = (this.baseline.corrugator.mean - metrics.corrugator) / this.baseline.corrugator.std;
        const zEar = (this.baseline.ear.mean - metrics.ear) / this.baseline.ear.std;
        const zLip = (this.baseline.lipPress.mean - metrics.lipPress) / this.baseline.lipPress.std;

        const poseConfidence = metrics.iod && this.baseline.iod > 0
            ? Math.min(1.0, metrics.iod / this.baseline.iod)
            : 1.0;

        const actCorr = zCorrugator > 1.5 ? zCorrugator * poseConfidence : 0;
        const actEar = zEar > 1.5 ? zEar : 0;
        const actLip = zLip > 2.0 ? zLip : 0;

        const stressDelta = (actCorr * 2.0) + (actEar * 0.8) + (actLip * 0.5);

        const incremento = stressDelta * 25.0 * dtSec;
        const decadimento = 35.0 * dtSec;

        this.stressAccumulator += (incremento - decadimento);
        this.stressAccumulator = Math.max(0, Math.min(this.stressAccumulator, this.activationThreshold));

        return {
            zCorrugator,
            zEar,
            zLip,
            stressDelta,
            isFrustrated: this.stressAccumulator >= this.activationThreshold,
            stressPercentage: this.stressAccumulator
        };
    }
}